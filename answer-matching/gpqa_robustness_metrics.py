
# GPQA answer-matching robustness metrics
# ---------------------------------------
#
# This script computes baseline vs. attack robustness metrics (alignment rate,
# ASR, flips, paired Cohen's d_z, McNemar p-value, and an optional naive
# two-proportion z-test) for the GPQA CSVs. 
# Make sure to make the changes to the columns and location for MMLU
#
# Usage:
#   python gpqa_robustness_metrics.py --base-dir scores/gpqa --out gpqa_summary.csv

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import math

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop accidental index columns
    for c in list(df.columns):
        if c.lower().startswith('unnamed:'):
            df = df.drop(columns=[c])
    # Normalize score column to int {0,1}
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0).astype(int)
    else:
        raise ValueError(f"'score' column missing in {path}")
    # Ensure key column exists
    if 'question' not in df.columns:
        raise ValueError(f"'question' column missing in {path}")
    # Normalize judge field
    if 'judge' in df.columns:
        df['judge'] = df['judge'].astype(str).str.replace('Qwen/Qwen3-4B', 'Qwen3-4B', regex=False)
    # Provide defaults
    if 'df_type' not in df.columns:
        name = path.name.lower()
        df['df_type'] = 'qual' if 'qual' in name else ('quant' if 'quant' in name else 'unknown')
    if 'attack' not in df.columns:
        df['attack'] = 'none'
    return df

def parse_meta_from_name(name: str):
    base = name.lower()
    model = 'gpt' if 'gpt_' in base else ('qwen' if 'qwen_' in base else 'unknown')
    split = 'qual' if 'qual_' in base else ('quant' if 'quant_' in base else 'unknown')
    # detect variant using presence of keywords
    if 'baseline' in base:
        variant = 'baseline'
    elif 'backward' in base:
        variant = 'backward'
    elif 'forward' in base:
        variant = 'forward'
    elif 'strategic' in base:
        variant = 'strategic'
    elif 'surface_gaming' in base:
        variant = 'surface_gaming'
    elif 'surface_light' in base:
        variant = 'surface_light'
    elif 'surface_heavy' in base:
        variant = 'surface_heavy'
    else:
        variant = base
    intensity = None
    if 'surface_light' in base: intensity = 'light'
    if 'surface_heavy' in base: intensity = 'heavy'
    return model, split, variant, intensity

def discover_csvs(base_dir: Path):
    files = list(base_dir.rglob("*_matches.csv"))
    entries = []
    for p in files:
        model, split, variant, intensity = parse_meta_from_name(p.name)
        entries.append({
            "path": p,
            "model": model,
            "split": split,
            "variant": variant,
            # "intensity": intensity
        })
    return pd.DataFrame(entries)

def mcNemar_pvalue(b, c):
    """McNemar's chi-square with continuity correction (df=1)."""
    if b + c == 0:
        return 1.0, float('nan')
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    # Compute p-value using chi-square CDF with df=1 (closed form via error function)
    # CDF_chi2(x; k=1) = erf(sqrt(x/2))
    p = 1 - math.erf(math.sqrt(chi2 / 2))
    return p, chi2

def two_proportion_z_test(x1, n1, x2, n2):
    """Naive two-sample proportion z-test (ignores pairing). Returns (z, p)."""
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if denom == 0:
        return 0.0, 1.0
    z = (p1 - p2) / denom
    # two-sided p-value from normal
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p

def paired_cohens_d(baseline: pd.Series, attack: pd.Series) -> float:
    """Cohen's d_z for paired binary scores: mean(diff)/sd(diff)."""
    diff = attack.astype(float) - baseline.astype(float)
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else diff.mean() / sd

def compute_pair_metrics(df_base: pd.DataFrame, df_attack: pd.DataFrame, key='question'):
    dfb = df_base[[key, 'score']].drop_duplicates(subset=[key])
    dfa = df_attack[[key, 'score']].drop_duplicates(subset=[key])
    merged = pd.merge(dfb, dfa, on=key, how='inner', suffixes=('_base', '_atk'))
    n = len(merged)
    if n == 0:
        raise ValueError("No overlap between baseline and attack CSVs")
    base_acc = merged['score_base'].mean()
    atk_acc = merged['score_atk'].mean()
    delta = atk_acc - base_acc

    # Flips
    up = ((merged['score_base'] == 0) & (merged['score_atk'] == 1)).sum()
    down = ((merged['score_base'] == 1) & (merged['score_atk'] == 0)).sum()
    same = n - up - down

    # ASR conditional on baseline incorrect
    denom = (merged['score_base'] == 0).sum()
    asr = (up / denom) if denom > 0 else float('nan')

    p_mcnemar, chi2 = mcNemar_pvalue(down, up)
    d_z = paired_cohens_d(merged['score_base'], merged['score_atk'])

    # Naive two-proportion z-test (unpaired assumption)
    z, p_z = two_proportion_z_test(int(merged['score_base'].sum()), n, int(merged['score_atk'].sum()), n)

    return {
        "n_pairs": int(n),
        "baseline_acc": float(base_acc),
        "attack_acc": float(atk_acc),
        "delta": float(delta),
        "flips_up": int(up),
        "flips_down": int(down),
        "flips_same": int(same),
        "ASR_cond_on_base_incorrect": float(asr) if asr == asr else None,
        "mcnemar_p": float(p_mcnemar) if p_mcnemar == p_mcnemar else None,
        "mcnemar_chi2": float(chi2) if chi2 == chi2 else None,
        "cohens_dz": float(d_z),
        "ztest_p": float(p_z),
        "ztest_z": float(z),
    }

def main(args):
    base_dir = Path(args.base_dir)
    index = discover_csvs(base_dir)
    if index.empty:
        raise SystemExit(f"No '*_matches.csv' found under {base_dir}")

    # Load all CSVs
    loaded = {}
    for row in index.itertuples(index=False):
        try:
            loaded[row.path] = load_csv(row.path)
        except Exception as e:
            print(f"[WARN] Skipping {row.path}: {e}")

    # Group by (model, split), then compare baseline to each attack variant present
    results = []
    for (model, split), group in index.groupby(['model','split']):
        base_rows = group[group['variant'] == 'baseline']
        if base_rows.empty:
            print(f"[WARN] No baseline for {(model, split)}; skipping")
            continue
        base_path = base_rows.iloc[0]['path']
        df_base = loaded[base_path]

        for _, row in group.iterrows():
            if row['variant'] == 'baseline':
                continue
            df_attack = loaded.get(row['path'])
            if df_attack is None:
                continue
            metrics = compute_pair_metrics(df_base, df_attack)
            results.append({
                "model": model,
                "split": split,
                "attack_variant": row['variant'],
                # "intensity": row['intensity'],
                **metrics
            })

    out_df = pd.DataFrame(results).sort_values(['model','split','attack_variant'], na_position='last')
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Wrote summary to {args.out}")
    # Pretty print
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(out_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="scores/gpqa", help="Root folder containing baseline/, backward/, surface_gaming/, etc.")
    parser.add_argument("--out", type=str, default="gpqa_summary.csv", help="Where to write the summary CSV")
    args = parser.parse_args()
    main(args)
