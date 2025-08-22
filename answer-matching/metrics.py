import pandas as pd


def mean_accuracy(df):
    valid_scores = df["score"].astype(str)  
    valid = valid_scores[valid_scores.str.len() < 2]
    valid = valid.astype(int)
    accuracy = valid.mean()

    verbose_mask = valid_scores.str.len() > 2
    verbose = df.loc[verbose_mask, "question"] 
    #returns questions that the model didnt reliably score

    return accuracy, valid, verbose


# def main():
#     df1 = pd.read_csv("gpqa_scores/gpqa_diamond_qual_qwen_matches_baseline.csv")
#     acc1, _, verbose1 = mean_accuracy(df1)
#     print(f"Accuracy Qwen Qual: {acc1}")  

#     df2 = pd.read_csv("gpqa_scores/gpqa_diamond_quant_qwen_matches_baseline.csv")
#     acc2, _, verbose2 = mean_accuracy(df2)
#     print(f"Accuracy Qwen Quant: {acc2}")  

#     df3 = pd.read_csv("gpqa_scores/gpqa_diamond_qual_gpt_matches_baseline.csv")
#     acc3, _, verbose3 = mean_accuracy(df3)
#     print(f"Accuracy GPT Qual: {acc3}")  

#     df4 = pd.read_csv("gpqa_scores/gpqa_diamond_quant_gpt_matches_baseline.csv")
#     acc4, _, verbose4 = mean_accuracy(df4)
#     print(f"Accuracy GPT Quant: {acc4}")  

#     print(f"Total QWEN: {mean_accuracy(pd.concat([df1, df2], ignore_index=True))[0]}")  

#     print(f"Total GPT: {mean_accuracy(pd.concat([df3, df4], ignore_index=True))[0]}")  

if __name__ == "__main__":
    main()