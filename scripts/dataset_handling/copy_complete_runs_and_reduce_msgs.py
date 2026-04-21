#!/usr/bin/env python3
"""
Copy only complete run folders from one runs project to another, then reduce msgs*.csv rows.

A run folder is considered complete iff it contains at least:
- one eval*.csv
- one predicate*.csv
- one msgs*.csv

After each copied run folder, this script calls:
  python -m concon.eval.predicate.reduce_msgs_rows <copied_run_dir> --apply
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_source(source: str) -> Path:
    p = Path(source)
    if p.exists():
        return p.resolve()

    runs_source = Path("runs") / source
    if runs_source.exists():
        return runs_source.resolve()

    raise FileNotFoundError(
        f"Source not found: {source!r}. Provide an existing path or a project under runs/."
    )


def _resolve_destination(destination: str) -> Path:
    p = Path(destination)
    if p.is_absolute():
        return p

    # If destination exists as provided, keep it.
    if p.exists():
        return p.resolve()

    # Convenience: bare name means runs/<name>
    if len(p.parts) == 1:
        return (Path("runs") / p).resolve()

    return p.resolve()


def _is_complete_run(run_dir: Path) -> bool:
    has_eval = any(run_dir.glob("eval*.csv"))
    has_pred = any(run_dir.glob("predicate*.csv"))
    has_msgs = any(run_dir.glob("msgs*.csv"))
    return has_eval and has_pred and has_msgs


def _copy_run(src_run_dir: Path, dst_run_dir: Path, overwrite: bool) -> bool:
    if dst_run_dir.exists() and not overwrite:
        return False
    shutil.copytree(src_run_dir, dst_run_dir, dirs_exist_ok=overwrite)
    return True


def _reduce_msgs_rows(target_run_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "concon.eval.predicate.reduce_msgs_rows",
        str(target_run_dir),
        "--apply",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="Source runs project name (resolved as runs/<source>) or an existing path.",
    )
    parser.add_argument(
        "destination",
        help="Destination path. Bare name is resolved as runs/<destination>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination run folders if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without copying or reducing.",
    )
    args = parser.parse_args()

    src_root = _resolve_source(args.source)
    dst_root = _resolve_destination(args.destination)
    dst_root.mkdir(parents=True, exist_ok=True)

    src_runs = sorted([p for p in src_root.iterdir() if p.is_dir()])
    if not src_runs:
        print(f"No run directories found in {src_root}")
        return

    considered = 0
    copied = 0
    skipped_incomplete = 0
    skipped_existing = 0

    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")

    for src_run in src_runs:
        considered += 1
        if not _is_complete_run(src_run):
            skipped_incomplete += 1
            print(f"[skip incomplete] {src_run.name}")
            continue

        dst_run = dst_root / src_run.name
        if dst_run.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"[skip existing] {dst_run}")
            continue

        print(f"[copy] {src_run} -> {dst_run}")
        if args.dry_run:
            continue

        _copy_run(src_run, dst_run, overwrite=args.overwrite)
        copied += 1

        print(f"[reduce msgs] {dst_run}")
        _reduce_msgs_rows(dst_run)

    print("\n=== Summary ===")
    print(f"considered={considered}")
    print(f"copied={copied}")
    print(f"skipped_incomplete={skipped_incomplete}")
    print(f"skipped_existing={skipped_existing}")


if __name__ == "__main__":
    main()
