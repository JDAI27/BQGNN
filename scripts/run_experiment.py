#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from bqgnn.train import run_and_save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config JSON")
    p.add_argument("--out", type=str, default="runs", help="Output directory for results")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    res = run_and_save(cfg, out_dir=args.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())

