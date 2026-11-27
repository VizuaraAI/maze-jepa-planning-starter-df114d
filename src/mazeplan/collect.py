"""Entry point: collect reward-free random play.  python -m mazeplan.collect --config configs/default.yaml"""
from __future__ import annotations

import argparse

from .data import collect_random_play
from .utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value overrides, e.g. data.n_transitions=100000")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    path = collect_random_play(d["family"], d["n_transitions"], d["traj_len"], d["n_layouts"],
                               cfg["seed"], d["path"], d["max_step"])
    print(f"saved {d['n_transitions']} transitions of {d['family']} play to {path}")


if __name__ == "__main__":
    main()
