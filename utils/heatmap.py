import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cheatmap(inFile, width=15, height=10, outFile='heatmap.png'):
    dataset = pd.read_csv(inFile)
    sns.heatmap(dataset.corr(), annot=True)
    plt.savefig(outFile)

