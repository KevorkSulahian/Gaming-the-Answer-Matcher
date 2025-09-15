import pandas as pd

def main():
    mmlu_pro_qual = pd.read_csv("datasets/mmlu/mmlu_pro_qualitative_test.csv")
    mmlu_pro_quant = pd.read_csv("datasets/mmlu/mmlu_pro_quantitative_test.csv")

    sampled_qual = mmlu_pro_qual.sample(n=150, random_state=40)
    sampled_quant = mmlu_pro_quant.sample(n=150, random_state=40)

    sampled_qual.to_csv("datasets/mmlu/mmlu_pro_qualitative_test_sample.csv", index=False)
    sampled_quant.to_csv("datasets/mmlu/mmlu_pro_quantitative_test_sample.csv", index=False)

if __name__ == "__main__":
    main()
