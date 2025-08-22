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
