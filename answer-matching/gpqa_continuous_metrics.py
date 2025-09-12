# GPQA answer-matching robustness metrics (continuous version)
# ------------------------------------------------------------
# Usage:
#   python gpqa_continuous_metrics.py --base-dir scores/gpqa --out gpqa_summary_continuous.csv

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in list(df.columns):
        if c.lower().startswith("unnamed:"):
            df = df.drop(columns=[c])
    if "score" not in df.columns:
        raise ValueError(f"'score' column missing in {path}")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "question" not in df.columns:
        raise ValueError(f"'question' column missing in {path}")
    return df

def parse_meta_from_name(name: str):
    base = name.lower()
    model = "gpt" if "gpt" in base else ("qwen" if "qwen" in base else "unknown")
    split = "qual" if "qual" in base else ("quant" if "quant" in base else "unknown")
    if "baseline" in base:
        variant = "baseline"
    elif "backward" in base:
        variant = "backward"
    elif "forward" in base:
        variant = "forward"
    elif "strategic" in base:
        variant = "strategic"
    elif "wrong" in base:
        variant = "wrong_baseline"
    elif "surface" in base:
        variant = "surface"
    else:
        variant = base
    return model, split, variant

def discover_csvs(base_dir: Path):
    files = list(base_dir.rglob("*__continuous_scores.csv"))
    entries = []
    for p in files:
        model, split, variant = parse_meta_from_name(p.name)
        entries.append({
            "path": p,
            "model": model,
            "split": split,
            "variant": variant
        })
    return pd.DataFrame(entries)

def paired_cohens_d(baseline: pd.Series, attack: pd.Series) -> float:
    diff = attack - baseline
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else diff.mean() / sd

def compute_pair_metrics(df_base: pd.DataFrame, df_attack: pd.DataFrame, key="question"):
    dfb = df_base[[key, "score"]].drop_duplicates(subset=[key])
    dfa = df_attack[[key, "score"]].drop_duplicates(subset=[key])
    merged = pd.merge(dfb, dfa, on=key, how="inner", suffixes=("_base", "_atk"))
    n = len(merged)
    if n == 0:
        raise ValueError("No overlap between baseline and attack CSVs")

    base_mean = merged["score_base"].mean()
    atk_mean = merged["score_atk"].mean()
    delta = atk_mean - base_mean
    d = paired_cohens_d(merged["score_base"], merged["score_atk"])

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
        df_base = loaded[base_path]

        for _, row in group.iterrows():
            if row["variant"] == "baseline":
                continue
            df_attack = loaded.get(row["path"])
            if df_attack is None:
                continue
            metrics = compute_pair_metrics(df_base, df_attack)
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
