import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def cscatter(filename: str, type_: str, figWidth: int, figHeight: int) -> None:
    sns.set(color_codes=True)
    df = pd.read_csv(filename, parse_dates=["Date"])
    fig = plt.figure(figsize=(figWidth, figHeight))
    mask = df["type"] == type_
    plt.scatter(df[mask].Date, df[mask].AveragePrice, cmap="plasma", c=df[mask].AveragePrice)
    plt.xlabel("Date")
    plt.ylabel("Average Price (USD)")
    plt.title(f"Average Price of {type_} Avocados Over Time")
    plt.savefig(f"cscatter_{type_}.png")
    plt.close()

def cbarplot(filename: str, x: str, y: str, figTitle: str, figWidth: int, figHeight: int) -> None:
    sns.set(color_codes=True)
    df = pd.read_csv(filename, parse_dates=["Date"])
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.barplot(x=x, y=y, data=df)
    plt.title(figTitle)
    plt.savefig(f"cbarplot_{figTitle}.png")
    plt.close()
    
def cheatmap(filename: str, cmap: str, annotate: bool, figTitle: str, figWidth: int, figHeight: int) -> None:
    sns.set(color_codes=True)
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.drop("Unnamed: 0", axis=1)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.heatmap(df.corr(), cmap=cmap, annot=annotate)
    plt.title(figTitle)
    plt.savefig(f"cheatmap_{figTitle}.png")
    plt.close()
    
def clmplot(filename: str, figWidth: int, figHeight: int) -> None:
    sns.set(color_codes=True)
    df = pd.read_csv(filename)
    df.columns = ["Y_Test", "Y_Preds"]
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.lmplot(x="Y_Test", y="Y_Preds", data=df)
    plt.savefig(f"clmplot_{filename}.png")
    plt.close()
    
def chistplot(filename: str, figTitle: str, figWidth: int, figHeight: int) -> None:
    sns.set(color_codes=True)
    df = pd.read_csv(filename)
    df.columns = ["Y_Test", "Y_Preds"]
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.histplot(df.Y_Test - df.Y_Preds)
    plt.title(f"{figTitle}")
    plt.savefig(f"chistplot_{figTitle}.png")
    plt.close()
    