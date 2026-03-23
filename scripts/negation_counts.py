#!/usr/bin/env python3
import json
import os
import pathlib
import sys


def _load_config(run_dir: pathlib.Path):
    cfg_path = run_dir / "hparams.json"
    if not cfg_path.is_file():
        return None
    try:
        with cfg_path.open() as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/negation_counts.py <runs_subdir>")
        sys.exit(1)

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs" / sys.argv[1]
    if not runs_dir.is_dir():
        print(f"Folder not found: {runs_dir}")
        sys.exit(1)

    neg = 0
    no_neg = 0
    missing = 0

    for entry in runs_dir.iterdir():
        if not entry.is_dir():
            continue
        cfg = _load_config(entry)
        # Handle nested run dirs with single child.
        if cfg is None:
            subdirs = [p for p in entry.iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                cfg = _load_config(subdirs[0])
                if cfg is not None:
                    entry = subdirs[0]

        if cfg is None:
            missing += 1
            continue

        if cfg.get("no_negation", False):
            no_neg += 1
        else:
            neg += 1

    total = neg + no_neg + missing
    print(f"Scanned: {total} folders in {runs_dir}")
    print(f"Negation enabled: {neg}")
    print(f"Negation disabled: {no_neg}")
    print(f"Missing/invalid hparams.json: {missing}")


if __name__ == "__main__":
    main()
