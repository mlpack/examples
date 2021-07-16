import pandas as pd
import matplotlib.pyplot as plt

def cpandashist(inFile, bins, width=20,height=15, outFile = 'histogram.png'):
    dataset = pd.read_csv(inFile)
    dataset.hist(bins = 50, figsize=(20,15))
    plt.savefig(outFile)
