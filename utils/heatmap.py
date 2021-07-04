import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import figure

def cheatmap(inFile, width=15, height=10, outFile='heatmap.png'):
    figure(figsize=(width,height))
    dataset = pd.read_csv(inFile)
    sns.heatmap(dataset.corr(), annot=True)
    plt.show()
    plt.savefig(outFile)