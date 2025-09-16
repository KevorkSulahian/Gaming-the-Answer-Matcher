# GPQA answer-matching robustness metrics (continuous version)
# ------------------------------------------------------------
# One row per CSV (each dataset/run reported separately).
#
# Usage:
#   python gpqa_continuous_metrics.py \
#     --base-dir answer-matching/scores/gpqa_cont \
#     --out answer-matching/scores/gpqa_cont/gpqa_summary_continuous.csv

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------- loading ----------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("empty CSV")
    # drop unnamed cols
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # columns check
    cols_lower = {c.lower(): c for c in df.columns}
    if "score" not in cols_lower:
        raise ValueError(f"'score' column missing in {path}")
    df[cols_lower["score"]] = pd.to_numeric(df[cols_lower["score"]], errors="coerce")
    if "question" not in cols_lower and "question_id" not in cols_lower:
        raise ValueError(f"'question' or 'question_id' column missing in {path}")
    # normalize column names to lower for downstream usage
    df.columns = [c.lower() for c in df.columns]
    return df

# ---------- path parsing ----------
_SUBVARIANT_PATTERNS = [
    "answers",
    "surface_medium",
    "surface_heavy",
    "surface_light",
    "multiple",
    "strategic",
    "verbose",
    "wrong",
]

def _find_subvariant(stem: str) -> str:
    s = stem.lower()
    for pat in _SUBVARIANT_PATTERNS:
        if pat in s:
            return pat
    return stem  # fallback to whole stem if nothing matches

def parse_meta_from_path(path: Path):
    """
    Parse model/split from filename, variant from parent folder,
    and build a unique attack_variant label using filename stem.
    Examples:
      .../baseline/gpt_quant_baseline_answers__continuous_scores.csv
      .../gpqa_surface_gaming/gpt_qual_surface_medium__continuous_scores.csv
      .../strategic/qwen_quant_strategic_answers__continuous_scores.csv
    """
    fname = path.name.lower()
    stem  = path.stem.lower()
    parent_variant = path.parent.name.lower()

    # model
    if "gpt" in fname or fname.startswith("gpt_"):
        model = "gpt"
    elif "qwen" in fname or fname.startswith("qwen_"):
        model = "qwen"
    else:
        model = "unknown"

    # split
    split = "quant" if "quant" in fname else ("qual" if "qual" in fname else "unknown")

    # baseline is just 'baseline'
    if parent_variant == "baseline":
        attack_variant = "baseline"
    else:
        sub = _find_subvariant(stem)
        # compose "folder/subvariant" for uniqueness
        attack_variant = f"{parent_variant}/{sub}"

    return model, split, attack_variant

def discover_csvs(base_dir: Path) -> pd.DataFrame:
    files = list(base_dir.rglob("*__continuous_scores.csv"))
    entries = []
    for p in files:
        model, split, attack_variant = parse_meta_from_path(p)
        entries.append(
            {
                "path": p,
                "model": model,
                "split": split,
                "attack_variant": attack_variant,
                "stem": p.stem.lower(),
                "folder": p.parent.name.lower(),
            }
        )
    df = pd.DataFrame(entries)
    if not df.empty:
        print("Discovered files:")
        print(
            df.sort_values(["model", "split", "attack_variant"])[
                [ "model", "split", "attack_variant"]
            ].to_string(index=False)
        )
    return df

# ---------- metrics ----------
def _norm_q(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )

def compute_pair_metrics(df_base: pd.DataFrame, df_attack: pd.DataFrame):
    # choose best merge key
    if "question_id" in df_base.columns and "question_id" in df_attack.columns:
        key = "question_id"
        dfb = df_base[[key, "score"]].drop_duplicates(subset=[key])
        dfa = df_attack[[key, "score"]].drop_duplicates(subset=[key])
    else:
        key = "question"
        dfb = df_base[[key, "score"]].copy()
        dfa = df_attack[[key, "score"]].copy()
        dfb[key] = _norm_q(dfb[key])
        dfa[key] = _norm_q(dfa[key])
        dfb = dfb.drop_duplicates(subset=[key])
        dfa = dfa.drop_duplicates(subset=[key])

    merged = pd.merge(dfb, dfa, on=key, how="inner", suffixes=("_base", "_atk"))
    n = len(merged)
    if n == 0:
        return None

    base_mean = merged["score_base"].mean()
    atk_mean  = merged["score_atk"].mean()
    delta     = atk_mean - base_mean

    diff = merged["score_atk"] - merged["score_base"]
    sd = diff.std(ddof=1)
    d = 0.0 if sd == 0 else diff.mean() / sd

    return {
        "n_pairs": int(n),
        "baseline_mean": float(base_mean),
        "attack_mean": float(atk_mean),
        "delta": float(delta),
        "cohens_dz": float(d),
    }

# ---------- main ----------
def main(args):
    base_dir = Path(args.base_dir)
    index = discover_csvs(base_dir)
    if index.empty:
        raise SystemExit(f"No continuous CSVs found under {base_dir}")

    # load all valid CSVs
    loaded = {}
    for row in index.itertuples(index=False):
        try:
            loaded[row.path] = load_csv(row.path)
        except Exception as e:
            print(f"[WARN] Skipping {row.path}: {e}")

    results = []
    # for each (model, split), find a baseline once
    for (model, split), group in index.groupby(["model", "split"]):
        # pick the first valid baseline file for this (model, split)
        base_rows = group[group["attack_variant"] == "baseline"]
        if base_rows.empty:
            print(f"[WARN] No baseline for {(model, split)}; skipping group")
            continue

        # choose the first baseline that actually loaded
        df_base = None
        base_path_used = None
        for p in base_rows["path"]:
            dfb = loaded.get(p)
            if dfb is not None and not dfb.empty:
                df_base = dfb
                base_path_used = p
                break
        if df_base is None:
            print(f"[WARN] No valid baseline CSV for {(model, split)}; skipping group")
            continue

        # now emit ONE ROW PER ATTACK FILE (unique by attack_variant label we built)
        for _, row in group.iterrows():
            if row["attack_variant"] == "baseline":
                continue
            df_attack = loaded.get(row["path"])
            if df_attack is None or df_attack.empty:
                print(f"[WARN] Missing/empty attack CSV for {row['path']}; skipping")
                continue
            m = compute_pair_metrics(df_base, df_attack)
            if m is None:
                print(f"[WARN] No overlap for {(model, split, row['attack_variant'])}; skipping")
                continue
            results.append(
                {
                    "model": model,
                    "split": split,
                    "attack_variant": row["attack_variant"],
                    # "baseline_path": str(base_path_used),
                    # "attack_path": str(row["path"]),
                    **m,
                }
            )

    out_df = pd.DataFrame(results).sort_values(
        ["model", "split", "attack_variant"], na_position="last"
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Wrote summary to {args.out}")

    print(out_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default="answer-matching/scores/gpqa_cont",
        help="Root folder containing scores/gpqa_cont/*/",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="gpqa_summary_continuous.csv",
        help="Where to write the summary CSV",
    )
    args = parser.parse_args()
    main(args)
