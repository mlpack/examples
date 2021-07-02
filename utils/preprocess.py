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
            dateCol: str = None) -> None:
    
    if dateCol != "":
        df = pd.read_csv(fname, parse_dates=[dateCol])
    else:
        df = pd.read_csv(fname)
        
    dfImp = Imputer(df, kind)
        
    dfImp.to_csv(f"{fname[:-4]}_{kind}_imputed.csv", index=False)
    
    
def Resample(data, replace, n_samples):
#     np.random.seed(random_state)
    indices = data.index
    random_sampled_indices = np.random.choice(indices,
                                              size=n_samples,
                                              replace=replace)
    return data.loc[random_sampled_indices]


# def dump(data, replace, n_samples, random_State = 123):
#     with open("test2.txt", "w") as f:
#         f.write("Test Pass")


def cresample(fname: str,
              target: str,
              neg_value: str,
              pos_value: str,
              kind: str = "oversample",
              dateCol: str = None,
              random_state = 123) -> None:
    
    if dateCol != "":
        df = pd.read_csv(fname, parse_dates=[dateCol])
    else:
        df = pd.read_csv(fname)
        
    negClass = df[df[target] == neg_value]
    posClass = df[df[target] == pos_value]
    
#     dump(posClass, True, len(negClass), randomState)
    
        
    if kind == "oversample":
        posOverSampled = Resample(data=posClass, replace=True, n_samples=len(negClass))
        overSampled = pd.concat([negClass, posOverSampled])
        overSampled.to_csv(f"{fname[:-4]}_oversampled.csv", index=False)
#     elif kind == "undersample":
#         negUnderSampled = Resample(data=negClass, replace=False, n_samples=len(posClass), random_state=random_state)
#         underSampled = pd.concat([negUnderSampled, posClass])
#         underSampled.to_csv(f"{fname[:-4]}_undersampled.csv", index=False)
#     elif kind == "smote":
#         os = SMOTE()
#         features, targets = os.fit_resample(df.iloc[:, :-1], df.iloc[:,-1])
#         smoteSampled = pd.concat([pd.DataFrame(features), pd.DataFrame(targets)], axis=1)
#         smoteSampled.to_csv(f"{fname[:-4]}_smotesampled.csv", index=False)
    
        
        
