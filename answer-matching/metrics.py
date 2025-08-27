import pandas as pd
import math
import numpy as np
from statsmodels.stats.proportion import proportions_ztest


def mean_accuracy(df):
    valid_scores = df["score"].astype(str) 
    # valid = valid_scores 
    valid = valid_scores[valid_scores.str.len() < 2]
    valid = valid.astype(int)
    accuracy = valid.mean()

    # verbose_mask = valid_scores.str.len() > 2
    # verbose = df.loc[verbose_mask, "question"] 
    #returns questions that the model didnt reliably score

    # return accuracy, valid, verbose
    return accuracy, len(valid)

def calc_asr(path):
    #baseline with wrong answers
    wrong_base_scores = pd.read_csv(path) 
    #if 1, given that the answers were all wrong:
    asr = (wrong_base_scores["score"].astype(int).mean())
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
    filtered['score_base'] = filtered['score_base'].astype(int)
    filtered['score_gamed'] = filtered['score_gamed'].astype(int)


    counter = ((filtered["score_base"] == 0) & (filtered["score_gamed"] == 1)).sum()
    asr = (counter/len(filtered))
    return asr, counter, len(filtered) #asr, successes, total

def significance(x1, n1, x2, n2): #successes, total_count
    stat, pval = proportions_ztest([x1, x2], [n1, n2])
    print(f"Baseline ASR: {x1/n1:.3f}")
    print(f"Attack   ASR: {x2/n2:.3f}")
    print(f"Z-statistic: {stat:.3f}")
    print(f"P-value: {pval:.4f}")

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

    print(f"""mean_baseline: {mean_base},
        mean_attack: {mean_attack},
        cohen's d: {cohen_d},
        cohen's h: {cohen_h}""")
    
    return {
        "mean_baseline": mean_base,
        "mean_attack": mean_attack,
        "cohen_d": cohen_d,
        "cohen_h": cohen_h
    }



def main():
    #mean accuracy
    df1 = pd.read_csv("gpqa_scores/gpqa_diamond_qual_qwen_matches_baseline.csv")
    acc1, num_samples = mean_accuracy(df1)
    print(f"Accuracy Qwen Qual: {acc1}, Total Questions: {num_samples}")  
    df2 = pd.read_csv("gpqa_scores/gpqa_diamond_quant_qwen_matches_baseline.csv")
    acc2, num_samples = mean_accuracy(df2)
    print(f"Accuracy Qwen Quant: {acc2}, Total Questions: {num_samples}")  
    df3 = pd.read_csv("gpqa_scores/gpqa_diamond_qual_gpt_matches_baseline.csv")
    acc3, num_samples = mean_accuracy(df3)
    print(f"Accuracy GPT Qual: {acc3}, Total Questions: {num_samples}")  
    df4 = pd.read_csv("gpqa_scores/gpqa_diamond_quant_gpt_matches_baseline.csv")
    acc4, num_samples = mean_accuracy(df4)
    print(f"Accuracy GPT Quant: {acc4}, Total Questions: {num_samples}")  

    print(f"Total QWEN: {mean_accuracy(pd.concat([df1, df2], ignore_index=True))[0]}")  
    print(f"Total GPT: {mean_accuracy(pd.concat([df3, df4], ignore_index=True))[0]}")  


    #ASR
    #baseline/control
    #GPQA
    base_qual = ""
    base_quant = ""
    base_asr_qual, total_base_qual = calc_asr(base_qual)
    print(f"Baseline ASR Qual: {base_asr_qual}, Total Questions: {total_base_qual}")
    base_asr_quant, total_base_quant = calc_asr(base_quant)
    print(f"Baseline ASR Quant: {base_asr_quant}, Total Questions: {total_base_quant}")

    #give matcher output csv paths for a gaming experiment
    gamed_qual = "" 
    gamed_quant = ""
    gamed_asr_qual, suc_qual, tot_g_qual = decision_flip(base_qual, gamed_qual)
    print(f"Gaming ASR Qual: {gamed_asr_qual}, Total Questions: {tot_g_qual}, Successes: {suc_qual} ")
    gamed_asr_quant, suc_quant, tot_g_quant = decision_flip(base_quant, gamed_quant)
    print(f"Gaming ASR Quant: {gamed_asr_quant}, Total Questions: {tot_g_quant}, Successes: {suc_quant} ")

    #Statistical Significance
    #Qual - Qwen
    significance(base_asr_qual*total_base_qual, total_base_qual, suc_qual, tot_g_qual)
    #Quant - Qwen
    significance(base_asr_quant*total_base_quant, total_base_quant, suc_quant, tot_g_quant)

    #Magnitude
    normal_base_qual = "scores/gpqa/baseline/gpqa_diamond_qual_qwen_matches_baseline_merged.csv"
    cohens(normal_base_qual, gamed_qual)
    normal_base_quant = "scores/gpqa/baseline/gpqa_diamond_quant_qwen_matches_baseline_merged.csv"
    cohens(normal_base_quant, gamed_quant)

if __name__ == "__main__":
    main()