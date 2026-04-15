#!/usr/bin/env python3
"""
Reduce msgs*.csv row counts while preserving per-predicate signal proportions.

For each file:
- Group rows by (pred_col, msg_col)
- For each predicate, divide signal counts by gcd(counts)
- Rewrite with those reduced multiplicities

Example:
  pred A: msg x -> 20, msg y -> 10  => gcd=10 => keep x:2, y:1
  pred B: msg z -> 7                 => gcd=7  => keep z:1

Default mode is dry-run. Use --apply to write files in place.
"""

import argparse
import csv
import math
import pathlib
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Tuple


@dataclass
class FileStats:
    path: pathlib.Path
    rows_before: int = 0
    rows_after: int = 0
    predicates: int = 0
    reducible: bool = False
    schema_ok: bool = True
    conflicts: int = 0


def _resolve_scan_root(target: str) -> pathlib.Path:
    """
    Resolve user target as:
    1) existing path (absolute or relative), OR
    2) project name under runs/<target>.
    """
    as_path = pathlib.Path(target)
    if as_path.exists():
        return as_path.resolve()

    as_project = pathlib.Path("runs") / target
    if as_project.exists():
        return as_project.resolve()

    raise FileNotFoundError(
        f"Target not found: {target!r}. Provide an existing path, "
        f"or a project name that exists under runs/."
    )


def _gcd_many(values: List[int]) -> int:
    if not values:
        return 1
    return reduce(math.gcd, values)


def _process_file(
    path: pathlib.Path,
    pred_col: str,
    msg_col: str,
    apply: bool,
) -> FileStats:
    stats = FileStats(path=path)

    by_pred_msg: Dict[str, Counter] = defaultdict(Counter)
    pred_order: List[str] = []
    msg_order: Dict[str, List[str]] = defaultdict(list)
    first_row_by_pair: Dict[Tuple[str, str], Dict[str, str]] = {}
    first_signature_by_pair: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if pred_col not in fieldnames or msg_col not in fieldnames:
            stats.schema_ok = False
            return stats

        for row in reader:
            stats.rows_before += 1
            pred = row[pred_col]
            msg = row[msg_col]
            key = (pred, msg)

            if pred not in by_pred_msg:
                pred_order.append(pred)
            if msg not in by_pred_msg[pred]:
                msg_order[pred].append(msg)

            by_pred_msg[pred][msg] += 1
            if key not in first_row_by_pair:
                first_row_by_pair[key] = row.copy()
                # Keep a lightweight signature to detect inconsistent extra columns in duplicates
                sig = tuple((k, row.get(k, "")) for k in fieldnames if k not in (pred_col, msg_col))
                first_signature_by_pair[key] = sig
            else:
                sig = tuple((k, row.get(k, "")) for k in fieldnames if k not in (pred_col, msg_col))
                if sig != first_signature_by_pair[key]:
                    stats.conflicts += 1

    stats.predicates = len(by_pred_msg)
    reduced_counts: Dict[Tuple[str, str], int] = {}
    for pred, msg_counts in by_pred_msg.items():
        g = _gcd_many(list(msg_counts.values()))
        for msg, count in msg_counts.items():
            reduced_counts[(pred, msg)] = count // g
            stats.rows_after += count // g

    stats.reducible = stats.rows_after < stats.rows_before

    if not apply:
        return stats

    # Rebuild file in a stable order using the first row observed per (pred,msg)
    with path.open("r", newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []

    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", delete=False, dir=str(path.parent)
    ) as tmp:
        tmp_path = pathlib.Path(tmp.name)
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()

        for pred in pred_order:
            for msg in msg_order[pred]:
                key = (pred, msg)
                row = first_row_by_pair[key]
                n = reduced_counts[key]
                for _ in range(n):
                    writer.writerow(row)

    tmp_path.replace(path)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help=(
            "Either project name (resolved as runs/<project>) OR an existing path "
            "(e.g. runs/<project>/<game_id>/<run>)."
        ),
    )
    parser.add_argument(
        "--glob",
        default="msgs*.csv",
        help="File glob to match recursively (default: msgs*.csv).",
    )
    parser.add_argument("--pred-col", default="pred_str", help="Predicate column name.")
    parser.add_argument("--msg-col", default="msg", help="Signal/message column name.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files in place. Without this flag, only print dry-run stats.",
    )
    args = parser.parse_args()

    try:
        root = _resolve_scan_root(args.target)
    except FileNotFoundError as e:
        parser.error(str(e))
        return

    files = sorted(root.rglob(args.glob))
    if not files:
        print(f"No files matched: {root}/**/{args.glob}")
        return

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Root: {root}")
    print(f"Files matched: {len(files)}")

    total_before = 0
    total_after = 0
    reducible_files = 0
    schema_bad = 0
    conflicts_total = 0

    for path in files:
        stats = _process_file(
            path=path,
            pred_col=args.pred_col,
            msg_col=args.msg_col,
            apply=args.apply,
        )
        if not stats.schema_ok:
            schema_bad += 1
            print(f"[SKIP schema] {path} (missing '{args.pred_col}' and/or '{args.msg_col}')")
            continue

        total_before += stats.rows_before
        total_after += stats.rows_after
        conflicts_total += stats.conflicts
        if stats.reducible:
            reducible_files += 1

        status = "reducible" if stats.reducible else "no_change"
        print(
            f"[{status}] {path} | rows {stats.rows_before} -> {stats.rows_after} | "
            f"predicates={stats.predicates} | conflicts={stats.conflicts}"
        )

    print("\n=== Summary ===")
    print(f"schema_skipped={schema_bad}")
    print(f"files_reducible={reducible_files}")
    print(f"rows_before={total_before}")
    print(f"rows_after={total_after}")
    if total_before > 0:
        reduction_pct = 100.0 * (1.0 - (total_after / total_before))
        print(f"reduction_pct={reduction_pct:.2f}")
    print(f"duplicate_nonkey_conflicts={conflicts_total}")


if __name__ == "__main__":
    main()
