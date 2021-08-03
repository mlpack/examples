import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def Imputer(data, kind = "mean"):
    df = data.copy()
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
    return df
    
def cimputer(fname: str,
             kind: str = "mean",
             dateCol: str = None,
             dataDir: str = "data") -> None:
    
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
    if dateCol != "":
        df = pd.read_csv(fname, parse_dates=[dateCol])
    else:
        df = pd.read_csv(fname)
    dfImp = Imputer(df, kind)
    if fname.find(f"{dataDir}/") != -1:
        dfImp.to_csv(f"./{fname[:-4]}_{kind}_imputed.csv", index=False)
    else:
        dfImp.to_csv(f"./{dataDir}/{fname[:-4]}_{kind}_imputed.csv", index=False)
    
    
def Resample(data, replace, n_samples):
    
    indices = data.index
    random_sampled_indices = np.random.choice(indices,
                                              size=n_samples,
                                              replace=replace)
    return data.loc[random_sampled_indices]


def cresample(fname: str,
              target: str,
              neg_value: str,
              pos_value: str,
              kind: str,
              dateCol: str = None,
              random_state = 123,
              dataDir: str = "data") -> None:
    
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
    if kind == "smote":
        df = pd.read_csv(fname, header=None)
    else:
        if dateCol != "":
            df = pd.read_csv(fname, parse_dates=[dateCol])
        else:
            df = pd.read_csv(fname)
        negClass = df[df[target] == neg_value]
        posClass = df[df[target] == pos_value]
        df = df.drop("Date", axis=1)
    if kind == "oversample":
        posOverSampled = Resample(data=posClass, replace=True, n_samples=len(negClass))
        overSampled = pd.concat([negClass, posOverSampled])
        if fname.find(f"{dataDir}/") != -1:
            overSampled.to_csv(f"./{fname[:-4]}_oversampled.csv", index=False)
        else:
            overSampled.to_csv(f"./{dataDir}/{fname[:-4]}_oversampled.csv", index=False)
    if kind == "undersample":
        negUnderSampled = Resample(data=negClass, replace=False, n_samples=len(posClass))
        underSampled = pd.concat([negUnderSampled, posClass])
        if fname.find(f"{dataDir}/") != -1:
            underSampled.to_csv(f"./{fname[:-4]}_undersampled.csv", index=False)
        else:
            underSampled.to_csv(f"./{dataDir}/{fname[:-4]}_undersampled.csv", index=False)
    if kind == "smote":
        so = SMOTE()
        features, targets = so.fit_resample(df.iloc[:, :-1], df.iloc[:,-1])
        smoteSampled = pd.concat([pd.DataFrame(features), pd.DataFrame(targets)], axis=1)
        if fname.find(f"{dataDir}/") != -1:
            smoteSampled.to_csv(f"./{fname[:-4]}_smotesampled.csv", index=False)
        else:
            smoteSampled.to_csv(f"./{dataDir}/{fname[:-4]}_smotesampled.csv", index=False)

def cresamplenum(fname: str,
                 target: str,
                 neg_value: int,
                 pos_value: int,
                 kind: str = "oversample",
                 dateCol: str = None,
                 random_state = 123,
                 dataDir: str = "data") -> None:

    if kind == "smote":
        df = pd.read_csv(fname, skiprows=1)
    else:
        if dateCol:
            df = pd.read_csv(fname, parse_dates=[dateCol])
            df = df.drop(dateCol, axis=1)
        else:
            df = pd.read_csv(fname)
        negClass = df[df[target] == neg_value]
        posClass = df[df[target] == pos_value]

    if kind == "oversample":
        posOverSampled = Resample(data=posClass, replace=True, n_samples=len(negClass))
        overSampled = pd.concat([negClass, posOverSampled])
        if fname.find(f"{dataDir}/") != -1:
            overSampled.to_csv(f"./{fname[:-4]}_oversampled.csv", index=False)
        else:
            overSampled.to_csv(f"./{dataDir}/{fname[:-4]}_oversampled.csv", index=False)
    if kind == "undersample":
        negUnderSampled = Resample(data=negClass, replace=False, n_samples=len(posClass))
        underSampled = pd.concat([negUnderSampled, posClass])
        if fname.find(f"{dataDir}/") != -1:
            underSampled.to_csv(f"./{fname[:-4]}_undersampled.csv", index=False)
        else:
            underSampled.to_csv(f"./{dataDir}/{fname[:-4]}_undersampled.csv", index=False)
    if kind == "smote":
        so = SMOTE()
        features, targets = so.fit_resample(df.iloc[:, :-1], df.iloc[:,-1])
        smoteSampled = pd.concat([pd.DataFrame(features), pd.DataFrame(targets)], axis=1)
        if fname.find(f"{dataDir}/") != -1:
            smoteSampled.to_csv(f"./{fname[:-4]}_smotesampled.csv", index=False)
        else:
            smoteSampled.to_csv(f"./{dataDir}/{fname[:-4]}_smotesampled.csv", index=False)

