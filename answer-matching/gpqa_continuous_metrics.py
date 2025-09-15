# GPQA answer-matching robustness metrics (continuous version)
# ------------------------------------------------------------
# Usage:
#   python gpqa_continuous_metrics.py --base-dir scores/gpqa --out gpqa_summary_continuous.csv

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# --- optionally harden load_csv to skip empty files gracefully ---
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("empty CSV")
    for c in list(df.columns):
        if c.lower().startswith("unnamed:"):
            df = df.drop(columns=[c])
    if "score" not in df.columns:
        raise ValueError(f"'score' column missing in {path}")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "question" not in df.columns and "question_id" not in df.columns:
        raise ValueError(f"'question' or 'question_id' column missing in {path}")
    return df


# --- with this ---
def parse_meta_from_path(path: Path):
    """
    Parse model/split from the filename (e.g., 'gpt_qual_...'), and
    variant from the parent folder name (e.g., 'baseline', 'strategic', ...).
    """
    fname = path.name.lower()
    # model / split are encoded in the filename in your tree (gpt_qual..., qwen_quant...)
    model = "gpt" if "gpt" in fname else ("qwen" if "qwen" in fname else "unknown")
    split = "qual" if "qual" in fname else ("quant" if "quant" in fname else "unknown")
    # variant is the directory name directly under base (baseline/strategic/verbose/wrong/etc.)
    variant = path.parent.name.lower()
    return model, split, variant

def discover_csvs(base_dir: Path):
    files = list(base_dir.rglob("*__continuous_scores.csv"))
    entries = []
    for p in files:
        model, split, variant = parse_meta_from_path(p)
        entries.append({"path": p, "model": model, "split": split, "variant": variant})
    df = pd.DataFrame(entries)
    # nice to see what was picked up
    if not df.empty:
        print(df.sort_values(["model","split","variant"])[["path","model","split","variant"]].to_string(index=False))
    return df
# --- with this ---


# --- make compute_pair_metrics robust + non-crashy when no overlap ---
def _norm_q(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower())

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
        return None  # let caller skip quietly

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

def main(args):
    base_dir = Path(args.base_dir)
    index = discover_csvs(base_dir)
    if index.empty:
        raise SystemExit(f"No continuous CSVs found under {base_dir}")

    # --- in main(), keep going if some are missing/empty/no-overlap ---
    loaded = {}
    for row in index.itertuples(index=False):
        try:
            loaded[row.path] = load_csv(row.path)
        except Exception as e:
            print(f"[WARN] Skipping {row.path}: {e}")

    results = []
    for (model, split), group in index.groupby(["model","split"]):
        base_rows = group[group["variant"] == "baseline"]
        if base_rows.empty:
            print(f"[WARN] No baseline for {(model, split)}; skipping")
            continue
        base_path = base_rows.iloc[0]["path"]
        df_base = loaded.get(base_path)
        if df_base is None or df_base.empty:
            print(f"[WARN] Baseline DF empty or missing for {(model, split)}; skipping")
            continue

        for _, row in group.iterrows():
            if row["variant"] == "baseline":
                continue
            df_attack = loaded.get(row["path"])
            if df_attack is None or df_attack.empty:
                print(f"[WARN] Missing/empty attack CSV for {row['path']}; skipping")
                continue
            metrics = compute_pair_metrics(df_base, df_attack)
            if metrics is None:
                print(f"[WARN] No overlap for {(model, split, row['variant'])}; skipping")
                continue
            results.append({
                "model": model,
                "split": split,
                "attack_variant": row["variant"],
                **metrics
            })


    out_df = pd.DataFrame(results).sort_values(["model","split","attack_variant"], na_position="last")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Wrote summary to {args.out}")
    print(out_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="scores/gpqa", help="Root folder containing scores/gpqa/*/")
    parser.add_argument("--out", type=str, default="gpqa_summary_continuous.csv", help="Where to write the summary CSV")
    args = parser.parse_args()
    main(args)
