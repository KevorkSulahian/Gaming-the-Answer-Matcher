import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

def mean_accuracy(df):
    # Convert to numeric, non-numeric becomes NaN
    numeric_scores = pd.to_numeric(df["score"], errors="coerce")

    # Drop NaNs
    numeric_scores = numeric_scores.dropna()

    # Optional: keep only "reasonable" values, e.g. 0 <= score <= 1
    numeric_scores = numeric_scores[(numeric_scores >= 0) & (numeric_scores <= 1)]

    accuracy = numeric_scores.mean()

    return accuracy, len(numeric_scores)

def calc_asr(path):
    #baseline with wrong answers
    wrong_base_scores = pd.read_csv(path) 
    asr = (wrong_base_scores["score"].astype(float).mean())
    return asr, len(wrong_base_scores) #asr, total

def decision_flip(baseline_path, gamed_path): #df
    baseline = pd.read_csv(baseline_path)
    gamed = pd.read_csv(gamed_path)
    # print(baseline.head(1))
    # print(gamed.head(1))

    for df in (baseline, gamed):
        df["question"] = (
            df["question"]
            .astype(str)           # force to string
            .str.strip()           # remove leading/trailing whitespace
            .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
        )

    merged = pd.merge(baseline, gamed, on="question", suffixes=("_base", "_gamed"))

    valid_base = merged['score_base'].astype(str).str.len() == 1
    valid_gamed = merged['score_gamed'].astype(str).str.len() == 1
    filtered = merged[valid_base & valid_gamed].copy()
    filtered['score_base'] = filtered['score_base'].astype(float)
    filtered['score_gamed'] = filtered['score_gamed'].astype(float)


    counter = ((filtered["score_base"] == 0) & (filtered["score_gamed"] == 1)).sum()
    asr = (counter/len(filtered))
    return asr, counter, len(filtered) #asr, successes, total

def significance(x1, n1, x2, n2): #successes, total_count
    stat, pval = proportions_ztest([x1, x2], [n1, n2])
    # print(f"Baseline ASR: {x1/n1:.3f}")
    # print(f"Attack   ASR: {x2/n2:.3f}")
    # print(f"Z-statistic: {stat:.3f}")
    # print(f"P-value: {pval:.4f}")
    zstat = stat
    pval = pval
    base_asr = x1/n1
    attack_asr = x2/n2
    return base_asr, attack_asr, zstat, pval

def cohens(baseline_path, attack_path): # csv paths
    """Given a normal scenario, how much better do the gamed answers perform?"""
    baseline = pd.read_csv(baseline_path)
    attack = pd.read_csv(attack_path)
    base_scores = pd.to_numeric(baseline['score'])
    attack_scores = pd.to_numeric(attack['score'])
    base_scores = base_scores[base_scores.isin([0,1])]
    attack_scores = attack_scores[attack_scores.isin([0,1])]

    mean_base = base_scores.mean()
    mean_attack = attack_scores.mean()
    std_base = base_scores.std(ddof=1)
    std_attack = attack_scores.std(ddof=1)
    n_base = len(base_scores)
    n_attack = len(attack_scores)

    pooled_std = np.sqrt(((n_base-1)*std_base**2 + (n_attack-1)*std_attack**2) / (n_base+n_attack-2))

    cohen_d = (mean_attack - mean_base) / pooled_std if pooled_std > 0 else np.nan

    p1, p2 = mean_base, mean_attack
    cohen_h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

    # print(f"""mean_baseline: {mean_base},
    #     mean_attack: {mean_attack},
    #     cohen's d: {cohen_d},
    #     cohen's h: {cohen_h}""")
    
    return {
        "mean_baseline": mean_base,
        "mean_attack": mean_attack,
        "cohen_d": cohen_d,
        "cohen_h": cohen_h
    }


def main():
    # Explicitly list all files by category + model
    file_paths = {
        "Baseline": {
            "Qwen3": "answer-matching/scores/mmlu/baseline_quant/gpt_mmlu_pro_baseline_quant_qwen3_binary_matcher.csv",
            "Qwen2.5": "answer-matching/scores/mmlu/baseline_quant/gpt_mmlu_pro_baseline_quant_qwen2.5_binary_matcher.csv",
            "Gemma": "answer-matching/scores/mmlu/baseline_quant/gpt_mmlu_pro_baseline_quant_gemma_binary_matcher.csv",
            "GPT":   "answer-matching/scores/mmlu/baseline_quant/gpt_mmlu_pro_baseline_quant_gpt_binary_matcher.csv",
        },
        "Forward": {
            "Qwen3": "answer-matching/scores/mmlu/forward_quant/gpt_mmlu_pro_forward_quant_qwen3_binary_matcher.csv",
            "Qwen2.5": "answer-matching/scores/mmlu/forward_quant/gpt_mmlu_pro_forward_quant_qwen2.5_binary_matcher.csv",
            "Gemma": "answer-matching/scores/mmlu/forward_quant/gpt_mmlu_pro_forward_quant_gemma_binary_matcher.csv",
            "GPT":   "answer-matching/scores/mmlu/forward_quant/gpt_mmlu_pro_forward_quant_gpt_binary_matcher.csv",
        },
        "Unsure": {
            "Qwen3": "answer-matching/scores/mmlu/unsure_quant/gpt_mmlu_pro_unsure_quant_qwen3_binary_matcher.csv",
            "Qwen2.5": "answer-matching/scores/mmlu/unsure_quant/gpt_mmlu_pro_unsure_quant_qwen2.5_binary_matcher.csv",
            "Gemma": "answer-matching/scores/mmlu/unsure_quant/gpt_mmlu_pro_unsure_quant_gemma_binary_matcher.csv",
            "GPT":   "answer-matching/scores/mmlu/unsure_quant/gpt_mmlu_pro_unsure_quant_gpt_binary_matcher.csv",
        },
        "Verbose": {
            "Qwen3": "answer-matching/scores/mmlu/verbose_quant/gpt_mmlu_pro_verbose_quant_qwen3_binary_matcher.csv",
            "Qwen2.5": "answer-matching/scores/mmlu/verbose_quant/gpt_mmlu_pro_verbose_quant_qwen2.5_binary_matcher.csv",
            "Gemma": "answer-matching/scores/mmlu/verbose_quant/gpt_mmlu_pro_verbose_quant_gemma_binary_matcher.csv",
            "GPT":   "answer-matching/scores/mmlu/verbose_quant/gpt_mmlu_pro_verbose_quant_gpt_binary_matcher.csv",
        }
    }

    data = []
    for category, model_files in file_paths.items():
        for model, path in model_files.items():
            df = pd.read_csv(path)
            acc, _ = mean_accuracy(df)
            data.append((category, model, acc))

    # Put results into a dataframe
    plot_df = pd.DataFrame(data, columns=["Category", "Model", "Accuracy"])

    # --- Grouped bar chart ---
    categories = list(file_paths.keys())
    models = list(next(iter(file_paths.values())).keys())  # ["Qwen", "Gemma", "GPT"]

    group_width = 0.8   # width of each group (smaller = more space between groups)
    bar_width = group_width / len(models)
    x = np.arange(len(categories)) * (1 + 0.4)  # add 40% spacing between groups
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for i, model in enumerate(models):
        subset = plot_df[plot_df["Model"] == model]
        # shift bars within each group
        ax.bar(
            x + (i - (len(models) - 1) / 2) * bar_width,
            subset["Accuracy"],
            bar_width,
            label=model
        )
        for xi, acc in zip(x + (i - (len(models) - 1) / 2) * bar_width, subset["Accuracy"]):
            ax.text(xi, acc + 0.01, f"{acc:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Average Score")
    ax.set_title("MMLU Pro Quant Binary Judging")
    ax.legend()

    plt.tight_layout()
    plt.savefig("mmlu_pro_quant_binary_judge.png")
    print("Saved plot")


if __name__ == "__main__":
    main()