#!/usr/bin/env python3
"""Atomically maintain the long-running fair experiment status file."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--completed", action="store_true")
    parser.add_argument("--detail", default="")
    args = parser.parse_args()
    path = Path(args.file)
    if path.is_file():
        value = json.loads(path.read_text())
    else:
        value = {"protocol": "fair_comparison_v1", "expected_runs": 26,
                 "learned_runs": 24, "completed_runs": []}
    current = {
        "stage": args.stage, "dataset": args.dataset, "model": args.model,
        "seed": int(args.seed) if args.seed else None, "detail": args.detail,
    }
    value.update({
        "state": args.state, "updated_at": datetime.now().astimezone().isoformat(),
        "current": current,
    })
    if args.completed:
        key = {k: current[k] for k in ("dataset", "model", "seed")}
        if key not in value["completed_runs"]:
            value["completed_runs"].append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
