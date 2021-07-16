import pandas as pd
import numpy as np

def cimputer(inFile, outFile, kind):
    dataset = pd.read_csv(inFile)
    df = dataset.copy(deep=True)
    for feature in df.columns:
        if df[feature].dtype == "float":
            if kind == "mean":
                df[feature] = df[feature].fillna(df[feature].mean())
            elif kind == "median":
                df[feature] = df[feature].fillna(df[feature].median())
            elif kind == "mode":
                df[feature] = df[feature].fillna(df[feature].mode()[0])
        elif df[feature].dtype == "object":
            df[feature] = df[feature].fillna(df[feature].mode()[0])
    df.to_csv(outFile, encoding='utf-8', index=False)
