import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def load_results(path='./results/results.csv'):
    df = pd.read_csv(path, nrows=10)
    index_col = list(df.columns[:-4])
    df = pd.read_csv(path, index_col=index_col)
    return df
