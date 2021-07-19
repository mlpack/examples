import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cheatmap(inFile, outFile='heatmap.png', width=15, height=10):
    plt.figure(figsize=(width,height))
    dataset = pd.read_csv(inFile)
    sns.heatmap(dataset.corr(), annot=True)
    plt.savefig(outFile)