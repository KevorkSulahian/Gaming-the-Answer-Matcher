#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch driver for eval_nonbinary.py
- Walks answer-generation/gpqa/* for CSVs
- Runs eval_nonbinary.py only on missing continuous outputs
- Skips:
    * gpqa_surface_gaming_1/
    * baseline MCQ CSVs
- Dry-run mode shows plan only
- Prints progress counter and per-run timing
"""

import subprocess as ss
from pathlib import Path
import re
import argparse
import time

# -------------------- paths --------------------
REPO = Path(__file__).resolve()
while REPO != REPO.parent:
    if (REPO / "answer-matching" / "inference.py").exists():
        break
    REPO = REPO.parent

ANS_GEN = REPO / "answer-generation" / "gpqa"
SCORES_ROOT = REPO / "answer-matching" / "scores" / "gpqa_cont"
EVAL_SCRIPT = REPO / "experiments" / "kev_test" / "eval_nonbinary.py"

# -------------------- helpers --------------------
def _safe_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

def discover_csvs():
    return sorted(ANS_GEN.rglob("*.csv"))

def should_skip(csv: Path) -> bool:
    """Skip unwanted CSVs"""
    if "gpqa_surface_gaming_1" in str(csv):
        return True
    if csv.parent.name == "baseline" and "mcq" in csv.name.lower():
        return True
    return False

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which files will run, skip execution"
    )
    args = ap.parse_args()

    skipped_existing = []
    skipped_manual = []
    to_run = []

    for csv in discover_csvs():
        if should_skip(csv):
            skipped_manual.append(csv)
            continue

        folder_name = csv.parent.name
        out_dir = SCORES_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{csv.stem}__continuous_scores.csv"

        if out_csv.exists():
            skipped_existing.append(out_csv)
            continue

        to_run.append((csv, out_csv))

    # ----------- summary -----------
    print("\n=== SUMMARY ===")
    print("Skipped (already exist):")
    for s in skipped_existing:
        print(f"  [skip-existing] {s}")

    print("\nSkipped (manual rules):")
    for s in skipped_manual:
        print(f"  [skip-rule] {s}")

    print("\nTo run (will generate):")
    for inp, out in to_run:
        print(f"  [run] {inp} -> {out}")
    print("================\n")

    if args.dry_run:
        print("Dry-run mode: no eval_nonbinary.py calls made.")
        return


    # ----------- run loop -----------
    total = len(to_run)
    for i, (inp, out) in enumerate(to_run, 1):
        print(f"[{i}/{total}] Running: {inp} -> {out}  (forcing --split quant)")
        start = time.time()
        ss.run(
            ["python", str(EVAL_SCRIPT), str(inp), "--split", "quant"],
            check=True
        )
        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        print(f"    ✓ Done in {mins}m{secs:02d}s")


if __name__ == "__main__":
    main()
