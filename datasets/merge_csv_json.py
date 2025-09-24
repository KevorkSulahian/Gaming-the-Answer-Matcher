#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import json
import argparse

def merge_csv_json(csv_path, json_path, output_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Load JSON
    with open(json_path, "r") as f:
        scores_data = json.load(f)

    # Extract question → score mapping
    scores_mapping = {}
    for k, v in scores_data.items():
        try:
            score_val = float(v["score"])
            question_text = k.split("::")[-1]  # keep only the part after "::"
            scores_mapping[question_text] = score_val
        except Exception:
            continue  # skip malformed entries

    # Try to find a column in the CSV that contains the questions
    question_col = None
    for col in df.columns:
        if df[col].astype(str).isin(scores_mapping.keys()).any():
            question_col = col
            break

    # Add scores column
    if question_col:
        df["score"] = df[question_col].map(scores_mapping)
    else:
        print("⚠️ Could not find a matching question column in the CSV. Scores will be empty.")
        df["score"] = None

    # Save merged CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Merged file saved at: {output_path}")


def main():
    # parser = argparse.ArgumentParser(description="Merge CSV table with JSON scores")
    # parser.add_argument("csv_file", help="Path to the input CSV file")
    # parser.add_argument("json_file", help="Path to the input JSON file with scores")
    # parser.add_argument("output_file", help="Path to save the merged CSV")

    # args = parser.parse_args()

    merge_csv_json("answer-matching/scores/mmlu/verbose_quant/gpt_mmlu_pro_verbose_quant_binary_Gemma-2-2B-IT_judge.csv", "answer-matching/scores/mmlu/verbose_quant/Qwen3-4B_verbose_quant_binary_answer_matches.json", "answer-matching/scores/mmlu/verbose_quant/gpt_mmlu_pro_verbose_quant_binary_Qwen3-4B_judge.csv")

    merge_csv_json("answer-matching/scores/mmlu/verbose_quant/gpt_mmlu_pro_verbose_quant_binary_Gemma-2-2B-IT_judge.csv", "answer-matching/scores/mmlu/verbose_quant/Qwen3-4B_verbose_quant_cont_answer_matches.json", "answer-matching/scores/mmlu/verbose_quant/gpt_mmlu_pro_verbose_quant_cont_Qwen3-4B_judge.csv")

if __name__ == "__main__":
    main()