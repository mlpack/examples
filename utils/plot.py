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
             figHeight: int = 7) -> None:
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

        Returns:
             (None): Function does not return anything.
    """
    sns.set(color_codes=True)
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
    plt.savefig(f"{figTitle}.png")
    plt.close()

def cbarplot(filename: str,
             x: str,
             y: str,
             dateCol: str = None,
             figTitle: str = None,
             figWidth: int = 5,
             figHeight: int = 7) -> None:
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

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    if dateCol:
        df = pd.read_csv(filename, parse_dates=[dateCol])
    else:
        df = pd.read_csv(filename)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.barplot(x=x, y=y, data=df)
    plt.title(figTitle)
    plt.savefig(f"{figTitle}.png")
    plt.close()

def cheatmap(filename: str,
             cmap: str,
             annotate: bool,
             figTitle: str,
             figWidth: int = 12,
             figHeight: int = 6) -> None:
    """
    Creates a heatmap (correlation map) of the dataset and saves it.

        Parameters:
                filename (str): Name of the dataset to load.
                cmap (str): Name of the color map to be used for plotting.
                annotate (bool): Indicates whether plot should be annotated with correlation values.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 12.
                figHeight (int): Height of the figure; defaults to 6.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    df = pd.read_csv(filename)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.heatmap(df.corr(), cmap=cmap, annot=annotate, square=True, fmt=".2f")
    plt.title(figTitle)
    plt.savefig(f"{figTitle}.png")
    plt.close()

def clmplot(filename: str,
            figTitle: str = None,
            figWidth: int = 6,
            figHeight: int = 7) -> None:
    """
    Generates a regression plot on the given dataset and saves it.

        Parameters:
                filename (str): Name of the dataset to load.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 6.
                figHeight (int): Height of the figure; defaults to 7.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    df = pd.read_csv(filename)
    df.columns = ["Y_Test", "Y_Preds"]
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.lmplot(x="Y_Test", y="Y_Preds", data=df)
    plt.savefig(f"{figTitle}.png")
    plt.close()

def chistplot(filename: str,
              figTitle: str = None,
              figWidth: int = 6,
              figHeight: int = 4) -> None:
    """
    Generated a histogram on the given dataset and saves it.

        Parameters:
                filename (str): Name of the dataset to load.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 6.
                figHeight (int): Height of the figure; defaults to 4.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    df = pd.read_csv(filename)
    df.columns = ["Y_Test", "Y_Preds"]
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.histplot(df.Y_Test - df.Y_Preds)
    plt.title(f"{figTitle}")
    plt.savefig(f"{figTitle}.png")
    plt.close()
 
def cmissing(filename: str, 
             cmap: str, 
             figTitle: str, 
             figWidth: int = 6, 
             figHeight: int = 4) -> None:
    """
    Creates a heatmap of missing values in each feature of the dataset and saves it.
    
        Parameters:
                filename (str): Name of the dataset to load.
                cmap (str): Name of the color map to be used for plotting.
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 12.
                figHeight (int): Height of the figure; defaults to 6.

            Returns:
                (None): Function does not return anything.
    """
    sns.set(color_codes=True)
    df = pd.read_csv(filename)
    fig = plt.figure(figsize=(figWidth, figHeight))
    ax = sns.heatmap(df.isnull(), cmap=cmap, cbar=False)
    plt.title(figTitle)
    plt.savefig(f"{figTitle}.png")
    plt.close() 
    
    
def ccountplot(filename: str,
               xCol: str,
               figTitle: str = None,
               hue: str = None,
               figWidth: int = 6, 
               figHeight: int = 4) -> None:
    
    """
    Creates a countplot of feature (xCol) in the dataset and saves it.
    
        Parameters:
                filename (str): Name of the dataset to load.
                xCol (str): Name of the feature to count 
                hhue (str):
                figTitle (str): Title for the figure to be save; defaults to None.
                figWidth (int): Width of the figure; defaults to 12.
                figHeight (int): Height of the figure; defaults to 6.

            Returns:
                (None): Function does not return anything.
    """
    
    sns.set(color_codes=True)
    df = pd.read_csv(filename)
    fig = plt.figure(figsize=(figWidth, figHeight))
    if hue != "":
        ax = sns.countplot(x=xCol, hue=hue, data=df)
    else:
        ax = sns.countplot(x=xCol, data=df)
    plt.title(f"{figTitle}")
    plt.savefig(f"{figTitle}.png")
    plt.close()
    

def cplotRocAUC(yTest: str,
                probs: str,
                outfile: str = "roc_auc") -> None:
    
    yTest = pd.read_csv(yTest)
    prob = pd.read_csv(probs)
    pbs = prob.iloc[:,1]
    fper, tper, thresh = roc_curve(yTest, pbs)
    plt.plot(fper, tper, color="orange", label="ROC")
    plt.plot([0,1], [0,1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{outfile}.png")
    plt.close()