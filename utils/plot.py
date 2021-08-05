import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

def cscatter(filename: str,
             xCol: str,
             yCol: str,
             dateCol:str = None,
             maskCol: str = None,
             type_: str = None,
             color: str = None,
             xLabel: str = None, 
             yLabel: str = None, 
             figTitle: str = None, 
             figWidth: int = 26, 
             figHeight: int = 7,
             plotDir: str = "plots") -> None:
    """
    Creates a scatter plot of size figWidth & figHeight, named figTitle and saves it.

        Parameters:
            filename (str): Name of the dataset to load.
            xCol (str): Name of the feature in dataset to plot against X axis.
            yCol (str): Name of the feature in dataset to plot against Y axis.
            dateCol (str): Name of the feature containing dates to parse; default to None.
            maskCol (str): Name of the feature in dataset to mask; defaults to None.
            type_ (str): Name of the feature in dataset to use for masking; defaults to None.
            color (str): Name of the feature in dataset to be used for color value in plotting;
                         defaults to None.
            xlabel (str): Label for X axis; defaults to None.
            ylabel (str): Label for Y axis; defaults to None.
            figTitle (str): Title for the figure to be save; defaults to None.
            figWidth (int): Width of the figure; defaults to 26.
            figHeight (int): Height of the figure; defaults to 7.
            plotDir (str): Name of the directory to save the generated plot; defaults to plots.

        Returns:
             (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    if dateCol:
        df = pd.read_csv(filename, parse_dates=[dateCol])
    else:
        df = pd.read_csv(filename)
    fig = plt.figure(figsize=(figWidth, figHeight))
    if maskCol:
        mask = df[maskCol] == type_
        if color:
            plt.scatter(df[mask][xCol], df[mask][yCol], cmap="plasma", c=df[mask][color])
        else:
            plt.scatter(df[mask][xCol], df[mask][yCol], cmap="plasma")
    else:
        if color:
            plt.scatter(df[xCol], df[yCol], cmap="plasma", c=df[color])
        else:
            plt.scatter(df[xCol], df[yCol], cmap="plasma")
    plt.xlabel(f"{xLabel}")
    plt.ylabel(f"{yLabel}")
    plt.title(f"{figTitle}")
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()

def cbarplot(filename: str, 
             x: str, 
             y: str, 
             dateCol: str = None, 
             figTitle: str = None, 
             figWidth: int = 5, 
             figHeight: int = 7,
             plotDir: str = "plots") -> None:
    """
    Creates a bar plot of size figWidth & figHeight, named figTitle between x & y.

        Parameters:
                filename (str): Name of the dataset to load.
                x (str): Name of the feature in dataset to plot against X axis.
                y (str): Name of the feature in dataset to plot against Y axis.
                dateCol (str): name of the feature containing dates to parse; default to None.
                maskCol (str): name of the feature in dataset to mask; defaults to None.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 5.
                figHeight (int): Height of the figure; defaults to 7.
                plotDir (str): Name of the directory to save the generated plot; defaults to plots.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    if dateCol:
        df = pd.read_csv(filename, parse_dates=[dateCol])
    else:
        df = pd.read_csv(filename)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.barplot(x=x, y=y, data=df)
    plt.title(figTitle)
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()

def cheatmap(filename: str, 
             cmap: str, 
             annotate: bool, 
             figTitle: str, 
             figWidth: int = 15, 
             figHeight: int = 15,
             plotDir: str = "plots") -> None:
    """
    Creates a heatmap (correlation map) of the dataset and saves it.

        Parameters:
                filename (str): Name of the dataset to load.
                cmap (str): Name of the color map to be used for plotting.
                annotate (bool): Indicates whether plot should be annotated with correlation values.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 12.
                figHeight (int): Height of the figure; defaults to 6.
                plotDir (str): Name of the directory to save the generated plot; defaults to plots.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    df = pd.read_csv(filename)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.heatmap(df.corr(), cmap=cmap, annot=annotate, square=True, fmt=".2f")
    plt.title(figTitle)
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()

def clmplot(filename: str,
            xCol: str,
            yCol: str,
            figTitle: str = None, 
            figWidth: int = 6, 
            figHeight: int = 7,
            plotDir: str = "plots") -> None:
    """
    Generates a regression plot on the given dataset and saves it.

        Parameters:
                filename (str): Name of the dataset to load.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 6.
                figHeight (int): Height of the figure; defaults to 7.
                plotDir (str): Name of the directory to save the generated plot; defaults to plots.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    df = pd.read_csv(filename)
    # df.columns = ["Y_Test", "Y_Preds"]
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.lmplot(x=xCol, y=yCol, data=df)
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()

def chistplot(filename: str,
              xCol: str,
              figTitle: str = None, 
              figWidth: int = 6, 
              figHeight: int = 4,
              plotDir: str = "plots") -> None:
    """
    Generated a histogram on the given dataset and saves it.

        Parameters:
                filename (str): Name of the dataset to load.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 6.
                figHeight (int): Height of the figure; defaults to 4.
                plotDir (str): Name of the directory to save the generated plot; defaults to plots.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    df = pd.read_csv(filename)
    # df.columns = ["Y_Test", "Y_Preds"]
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.histplot(x=xCol, data=df)
    plt.title(f"{figTitle}")
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()

def cmissing(filename: str, 
             cmap: str, 
             figTitle: str, 
             figWidth: int = 6, 
             figHeight: int = 4,
             plotDir: str = "plots") -> None:
    """
    Creates a heatmap of missing values in each feature of the dataset and saves it.
    
        Parameters:
                filename (str): Name of the dataset to load.
                cmap (str): Name of the color map to be used for plotting.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 12.
                figHeight (int): Height of the figure; defaults to 6.
                plotDir (str): Name of the directory to save the generated plot; defaults to plots.
                

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    df = pd.read_csv(filename)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.heatmap(df.isnull(), cmap=cmap, cbar=False)
    plt.title(figTitle)
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close() 
    
def ccountplot(filename: str,
               xCol: str,
               figTitle: str = None,
               hue: str = None,
               figWidth: int = 6, 
               figHeight: int = 4,
               plotDir: str = "plots") -> None:
    """
    Creates a countplot of feature (xCol) in the dataset and saves it.
    
        Parameters:
                filename (str): Name of the dataset to load.
                xCol (str): Name of the feature to count 
                hue (str):
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 12.
                figHeight (int): Height of the figure; defaults to 6.
                plotDir (str): Name of the directory to save the generated plot; defaults to plots.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    df = pd.read_csv(filename)
    fig = plt.figure(figsize=(figWidth, figHeight))
    if hue != "":
        ax = sns.countplot(x=xCol, hue=hue, data=df)
    else:
        ax = sns.countplot(x=xCol, data=df)
    plt.title(f"{figTitle}")
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()
    
def cplotRocAUC(yTest: str,
                probs: str,
                figTitle: str = None,
                plotDir: str = "plots") -> None:  
    """
    Generates a ROC AUC curve from the give targets & probabilities.
    
        Parameters:
                yTest (str): Name of the dataset to load containing the targets.
                probs (str): Name of the dataset to load containing the probabilities.
                figTitle (str): Title for the figure to be save; defaults to None.
                plotDir (str): Name of the directory to save the generated plot; defaults to plots.

            Returns:
                (None): Function does not return anything.
    """
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    yTest = pd.read_csv(yTest)
    prob = pd.read_csv(probs)
    pbs = prob.iloc[:,1]
    fper, tper, thresh = roc_curve(yTest, pbs)
    plt.plot(fper, tper, color="orange", label="ROC")
    plt.plot([0,1], [0,1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{figTitle}")
    plt.legend()
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()
    
def clineplot(fname: str,
              xCol: str,
              yCol: str,
              figTitle: str = None,
              figWidth: int = 16,
              figHeight: int = 6,
              plotDir: str = "plots") -> None:
    
    sns.set(color_codes=True)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    df = pd.read_csv(fname)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.lineplot(x=xCol, y=yCol, data=df)
    plt.title(f"{figTitle}")
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()

def cplotCatData(fName: str,
                targetCol: int,
                xLabel: str,
                yLabel: str,
                figTitle: str = None,
                figWidth: int = 8, 
                figHeight: int = 6,
                plotDir: str = "plots") -> None:

    """
    Generates a categorical plot.
    
        Parameters:
                fName (str): Name of the dataset to load.
                targetCol (int): Numeric value representing the target column.
                xlabel (str): Label for X axis; defaults to None.
                ylabel (str): Label for Y axis; defaults to None.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 8.
                figHeight (int): Height of the figure; defaults to 6.
                plotDir (str): Name of the directory to save the generated plot; defaults to plots.

            Returns:
                (None): Function does not return anything.
    """
    microChipData = pd.read_csv(fName)
    X = microChipData.iloc[:, :targetCol].values
    y = microChipData.iloc[:, targetCol].values
    pos = np.argwhere(y == 1)
    neg = np.argwhere(y == 0)
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = fig.add_subplot()
    ax.scatter(X[neg, 0], X[neg, 1], c="yellow", marker='o', edgecolor="black", linewidth=0.5)
    ax.scatter(X[pos, 0], X[pos, 1], c="black", marker='+')
    ax.set_xlabel(f"{xLabel}")
    ax.set_ylabel(f"{yLabel}")
    plt.legend(["y = 0", "y = 1"])
    plt.title(f"{figTitle}")
    plt.savefig(f"./{plotDir}/{figTitle}.png")
    plt.close()
