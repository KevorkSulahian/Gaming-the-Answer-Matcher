import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import re

def mean_accuracy(df):
    def extract_number(value):
        """Extract first numeric value (int or float) from a string or numeric cell."""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return value
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(value))
        return float(match.group()) if match else None

    # Apply extraction to the 'score' column
    numeric_scores = df["score"].apply(extract_number)

    # Drop NaNs
    numeric_scores = numeric_scores.dropna()

    # Keep only reasonable values (e.g., 0 ≤ score ≤ 1)
    numeric_scores = numeric_scores[(numeric_scores >= 0) & (numeric_scores <= 1)]

    # Compute mean
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
    # Combine qualitative + quantitative file paths
    file_groups = {
        "gpt-4.1-mini qualitative": {
            "baseline": {
                "Qwen3": "wworkspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_qual/qwen3-4b/gpt_mmlu_pro_baseline_qual_cont_test_Qwen3-4B_judge.csv",
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_qual/qwen2.5-7b/gpt_mmlu_pro_baseline_qual_qwen2.5_cont_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_qual/gemma2b/gpt_mmlu_pro_baseline_qual_gemma_cont_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_qual/gpt-4.1 mini/gpt_mmlu_pro_baseline_qual_gpt_cont_matcher.csv",
            },
            "multiple-forward": {
                "Qwen3": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_qual/qwen3-4b/gpt_mmlu_pro_forward_qual_binary_Qwen3-4B_judge.csv",
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_qual/qwen2.5-7b/gpt_qual_forward_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_qual/gemma2b/gpt_mmlu_pro_forward_qual_gemma_binary_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_qual/gpt-4.1 mini/gpt_qual_forward_scores_gpt-4.1_binary_matcher.csv",
            },
            "strategic": {
                "Qwen3": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_qual/qwen3-4b/gpt_mmlu_pro_unsure_qual_qwen3_binary_matcher.csv",
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_qual/qwen2.5-7b/gpt_qual_strategic_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_qual/gemma2b/gpt_mmlu_pro_unsure_qual_gemma_binary_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_qual/gpt-4.1 mini/gpt_qual_strategic_scores_gpt_binary_matcher.csv",
            },
            "verbose": {
                "Qwen3": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_qual/qwen3-4b/gpt_mmlu_pro_verbose_qual_qwen3_binary_matcher.csv",
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_qual/qwen2.5-7b/gpt_qual_verbose_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_qual/gemma2b/gpt_mmlu_pro_verbose_qual_gemma_binary_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_qual/gpt-4.1 mini/gpt_qual_verbose_scores_gpt-4.1_binary_matcher.csv",
            }
        },
        "gpt-4.1-mini quantitative": {
            "baseline": {
                "Qwen3": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_quant/qwen3-4b/gpt_mmlu_pro_baseline_quant_qwen3_binary_matcher.csv",
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_quant/qwen2.5-7b/gpt_mmlu_pro_baseline_quant_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_quant/gemma2b/gpt_mmlu_pro_baseline_quant_gemma_binary_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_quant/gpt-4.1 mini/gpt_quant_baseline_scores_gpt-4.1_binary_matcher.csv",
            },
            "multiple-forward": {
                "Qwen3": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_quant/qwen3-4b/gpt_mmlu_pro_forward_quant_binary_Qwen3-4B_judge.csv",
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_quant/qwen2.5-7b/gpt_mmlu_pro_forward_quant_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_quant/gemma2b/gpt_mmlu_pro_forward_quant_gemma_binary_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_quant/gpt-4.1 mini/gpt_mmlu_pro_forward_quant_gpt-4.1_binary_matcher.csv",
            },
            "strategic": {
                "Qwen3": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_quant/qwen3-4b/gpt_mmlu_pro_unsure_quant_qwen3_binary_matcher.csv",
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_quant/qwen2.5-7b/gpt_mmlu_pro_unsure_quant_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_quant/gemma2b/gpt_mmlu_pro_unsure_quant_gemma_binary_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_quant/gpt-4.1 mini/gpt_mmlu_pro_unsure_quant_gpt-4.1_binary_matcher.csv",
            },
            "verbose": {
                "Qwen3": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_quant/qwen3-4b/gpt_mmlu_pro_verbose_quant_qwen3_binary_matcher.csv",
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_quant/qwen2.5-7b/gpt_mmlu_pro_verbose_quant_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_quant/gemma2b/gpt_mmlu_pro_verbose_quant_gemma_binary_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_quant/gpt-4.1/gpt_mmlu_pro_verbose_quant_gpt-4.1_binary_matcher.csv",
            }
        },
        "qwen2.5 qualitative": {
            "baseline": {
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_qual/qwen2.5-7b/qwen_qual_baseline_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_qual/gemma2b/qwen_mmlu_pro_baseline_qual_gemma_binary_judge.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_qual/gpt-4.1 mini/qwen_qual_baseline_scores_gpt_binary_matcher.csv",
            },
            "multiple-forward": {
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_qual/qwen2.5-7b/qwen_qual_forward_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_qual/gemma2b/qwen_mmlu_pro_forward_qual_binary_gemma_judge.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_qual/gpt-4.1 mini/qwen_qual_forward_scores_gpt-4.1_binary_matcher.csv",
            },
            "strategic": {
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_qual/qwen2.5-7b/qwen_qual_strategic_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_qual/gemma2b/qwen_mmlu_pro_unsure_qual_binary_gemma_judge.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_qual/gpt-4.1 mini/qwen_qual_strategic_scores_gpt_binary_matcher.csv",
            },
            "verbose": {
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_qual/qwen2.5-7b/qwen_qual_verbose_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_qual/gemma2b/qwen_mmlu_pro_verbose_qual_binary_gemma_judge.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_qual/gpt-4.1 mini/qwen_qual_verbose_scores_gpt-4.1_binary_matcher.csv",
            }
        },
        "qwen2.5 quantitative": {
            "baseline": {
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_quant/qwen2.5-7b/qwen_quant_baseline_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_quant/gemma2b/qwen_mmlu_pro_baseline_quant_binary_gemma_judge.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/baseline_quant/gpt-4.1 mini/qwen_quant_baseline_scores_gpt-4.1_binary_matcher.csv",
            },
            "multiple-forward": {
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_quant/qwen2.5-7b/qwen_quant_forward_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_quant/gemma2b/qwen_mmlu_pro_forward_quant_binary_gemma_judge.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/forward_quant/gpt-4.1 mini/gpt_mmlu_pro_forward_quant_gpt-4.1_binary_matcher.csv",
            },
            "strategic": {
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_quant/qwen2.5-7b/qwen_quant_strategic_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_quant/gemma2b/gpt_mmlu_pro_unsure_quant_gemma_binary_matcher.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/unsure_quant/gpt-4.1 mini/qwen_quant_strategic_scores_gpt-4.1_binary_matcher.csv",
            },
            "verbose": {
                "Qwen2.5": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_quant/qwen2.5-7b/qwen_quant_verbose_scores_qwen2.5_binary_matcher.csv",
                "Gemma": "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_quant/gemma2b/qwen_mmlu_pro_verbose_quant_cont_gemma_judge.csv",
                "GPT":   "workspace/Gaming-the-Answer-Matcher/answer-matching/scores/mmlu/verbose_quant/gpt-4.1/qwen_quant_verbose_scores_gpt-4.1_binary_matcher.csv",
            }
        }
    }

    # Collect data
    # data = []
    # for qtype, exps in file_groups.items():
    #     for experiment, models in exps.items():
    #         for judge, path in models.items():  # 'judge' = Gemma, GPT, Qwen2.5, Qwen3
    #             df = pd.read_csv(path)
    #             acc, _ = mean_accuracy(df)
    #             model_label = f"gpt-4.1-mini qualitative" if qtype == "qualitative" else "gpt-4.1-mini quantitative"
    #             data.append({
    #                 "Experiment": experiment,
    #                 "Judge": judge,
    #                 "Accuracy": acc,
    #                 "Testee": model_label
    #             })
        
    # plot_df = pd.DataFrame(data)
    
    # # --- Plot using Seaborn ---
    # import seaborn as sns
    # sns.set_style("darkgrid")
    # sns.set_palette("muted")
    
    # # Map judge names to match screenshot format
    # judge_map = {"Gemma": "gemma2b", "GPT": "gpt4.1mini", "Qwen2.5": "qwen2.5_7b", "Qwen3": "qwen3_4b"}
    # plot_df["Judge"] = plot_df["Judge"].map(judge_map)
    
    # judges = ["gemma2b", "gpt4.1mini", "qwen2.5_7b", "qwen3_4b"]
    # experiments = ["baseline", "multiple-forward", "strategic", "verbose"]
    # testees = ["gpt-4.1-mini qualitative", "gpt-4.1-mini quantitative"]
    
    # fig, axes = plt.subplots(1, len(judges), figsize=(16, 5), sharey=True)
    
    # for ax, judge in zip(axes, judges):
    #     subset = plot_df[plot_df["Judge"] == judge]
        
    #     # Use seaborn barplot
    #     sns.barplot(data=subset, x="Experiment", y="Accuracy", hue="Testee", 
    #                order=experiments, ax=ax, errorbar=None)
        
    #     ax.set_title(judge, fontsize=12)
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    #     ax.set_ylim(0, 1)
    #     ax.set_xlabel("Experiment", fontsize=10)
    #     ax.legend_.remove() if ax.legend_ else None
    
    # axes[0].set_ylabel("Accuracy", fontsize=11)
    
    # # Add legend to the top right
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, title="Testee", loc='upper right', 
    #            bbox_to_anchor=(0.98, 0.98), fontsize=10, title_fontsize=11)
    
    # plt.tight_layout(rect=[0, 0, 0.95, 1])
    # plt.savefig("binary_matcher_accuracy_2.png", dpi=300, bbox_inches="tight")
    # print("✅ Saved: binary_matcher_accuracy.png")

    data = []
    for qtype, exps in file_groups.items():
        for experiment, models in exps.items():
            for judge, path in models.items():
                df = pd.read_csv(path)
                acc, _ = mean_accuracy(df)
                # extract just the type (qualitative / quantitative)
                qtype_label = "qualitative" if "qual" in qtype else "quantitative"
                data.append({
                    "Experiment": experiment,
                    "Judge": judge,
                    "Accuracy": acc,
                    "Type": qtype_label
                })

    plot_df = pd.DataFrame(data)

    # --- Normalize judge names ---
    judge_map = {
        "Gemma": "gemma2b",
        "GPT": "gpt4.1mini",
        "Qwen2.5": "qwen2.5_7b",
        "Qwen3": "qwen3_4b",
    }
    plot_df["Judge"] = plot_df["Judge"].map(judge_map)

    # --- Aggregate across models (average Accuracy per Judge, Experiment, and Type) ---
    agg_df = (
        plot_df
        .groupby(["Experiment", "Judge", "Type"], as_index=False)["Accuracy"]
        .mean()
    )

    # --- Plot setup ---
    sns.set_style("darkgrid")
    sns.set_palette("muted")

    judges = ["gemma2b", "gpt4.1mini", "qwen2.5_7b", "qwen3_4b"]
    experiments = ["baseline", "multiple-forward", "strategic", "verbose"]
    types = ["qualitative", "quantitative"]

    fig, axes = plt.subplots(1, len(judges), figsize=(16, 5), sharey=True)

    for ax, judge in zip(axes, judges):
        subset = agg_df[agg_df["Judge"] == judge]
        sns.barplot(
            data=subset,
            x="Experiment",
            y="Accuracy",
            hue="Type",
            order=experiments,
            hue_order=types,
            ax=ax,
            errorbar=None
        )
        ax.set_title(judge, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Experiment", fontsize=10)
        ax.legend_.remove() if ax.legend_ else None

    axes[0].set_ylabel("Accuracy", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Question Type",
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        fontsize=10,
        title_fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig("cont_matcher_accuracy_by_type.png", dpi=300, bbox_inches="tight")
    print("✅ Saved: cont_matcher_accuracy_by_type.png")

if __name__ == "__main__":
    main()
