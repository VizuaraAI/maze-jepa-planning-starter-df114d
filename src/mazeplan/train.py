"""Entry point: train the world model (and optionally the temporal-distance head).

python -m mazeplan.train --config configs/default.yaml [key=value ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PlayDataset
from .model import WorldModel
from .tdcost import train_td_head
from .utils import get_device, load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    dev = get_device(cfg["device"])
    m, t = cfg["model"], cfg["train"]
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = PlayDataset(cfg["data"]["path"], seq_len=t["seq_len"], img_size=cfg["data"]["img_size"])
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], drop_last=True)
    wm = WorldModel(m["latent_dim"], 2, m["enc_width"], m["pred_hidden"]).to(dev)
    opt = torch.optim.AdamW(wm.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    print(f"device={dev} params={sum(p.numel() for p in wm.parameters()) / 1e6:.2f}M samples={len(ds)}")

    step, t_start, log = 0, time.time(), []
    while step < t["steps"]:
        for frames, actions, _, _ in dl:
            out = wm.loss(frames.to(dev), actions.to(dev), t["sigreg_weight"])
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            opt.step()
            if step % t["log_every"] == 0:
                row = {k: round(v.item(), 5) for k, v in out.items()} | {"step": step, "sec": round(time.time() - t_start, 1)}
                log.append(row)
                print(row)
            step += 1
            if step >= t["steps"]:
                break

    torch.save({"model": wm.state_dict(), "cfg": cfg}, run_dir / "world_model.pt")
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=1))

    if cfg["td"]["enabled"]:
        head = train_td_head(wm.encoder, ds, m["latent_dim"], steps=cfg["td"]["steps"],
                             max_offset=cfg["td"]["max_offset"], device=dev, seed=cfg["seed"])
        torch.save(head.state_dict(), run_dir / "td_head.pt")
    print(f"saved checkpoints to {run_dir}")


if __name__ == "__main__":
    main()
