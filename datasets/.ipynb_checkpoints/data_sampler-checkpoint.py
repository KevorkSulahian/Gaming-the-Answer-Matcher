import pandas as pd

def main():
    mmlu_pro_qual = pd.read_csv("datasets/mmlu/mmlu_pro_qualitative_test.csv")
    mmlu_pro_quant = pd.read_csv("datasets/mmlu/mmlu_pro_quantitative_test.csv")

    mmlu_pro_qual_answers = pd.read_csv("answer-generation/mmlu/mmlu-multiple/gpt_mmlu_pro_forward_qual_test_answers.csv")
    mmlu_pro_quant_answers = pd.read_csv("answer-generation/mmlu/mmlu-multiple/gpt_mmlu_pro_forward_quant_test_answers.csv")

    sampled_qual = mmlu_pro_qual.sample(n=2000, random_state=40)
    sampled_quant = mmlu_pro_quant.sample(n=2000, random_state=40)

    sample_qual_questions = sampled_qual["question"].tolist()
    sample_quant_questions = sampled_quant["question"].tolist()

    answers_sample_qual = mmlu_pro_qual_answers[mmlu_pro_qual_answers['question'].isin(sample_qual_questions)]
    answers_sample_quant = mmlu_pro_quant_answers[mmlu_pro_quant_answers['question'].isin(sample_quant_questions)]

    sampled_qual.to_csv("datasets/mmlu/mmlu_pro_qualitative_test_sample_large.csv", index=False)
    sampled_quant.to_csv("datasets/mmlu/mmlu_pro_quantitative_test_sample_large.csv", index=False)

    answers_sample_qual.to_csv("answer-generation/mmlu/mmlu-baseline/gpt_mmlu_pro_forward_qual_test_answers_sample_large.csv", index=False)
    answers_sample_quant.to_csv("answer-generation/mmlu/mmlu-baseline/gpt_mmlu_pro_forward_quant_test_answers_sample_large.csv", index=False)

if __name__ == "__main__":
    main()
