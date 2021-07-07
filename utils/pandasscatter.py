from matplotlib import figure
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def cpandasscatter(inFile, x, y, outFile='output.png', height=10, width=10):
    dataset = pd.read_csv(inFile)
    fig = dataset.plot(kind="scatter", x=x, y=y, alpha=0.1, figsize=(width, height))
    fig.figure.savefig(outFile)

def cpandasscattercolor(inFile, x, y, label, c, outFile='output1.png', height=10, width=10):
    dataset = pd.read_csv(inFile)
    fig = dataset.plot(kind="scatter", x=x, y=y, alpha=0.4,s=dataset["population"]/100,
                       label=label, c=c, cmap=plt.get_cmap("jet"), colorbar=True,
                       sharex = False)
    fig.figure.savefig(outFile)

def cpandasscattermap(inFile, imgFile, x, y, label, c, outFile="output2.png", height=10, width=7):
    figure(figsize=(10,7))
    im = plt.imread(imgFile)
    dataset = pd.read_csv(inFile)
    implot = plt.imshow(im, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
    plt.scatter(x=dataset[x], y=dataset[y], s=dataset["population"]/100, label=label, c=dataset[c], cmap=plt.get_cmap("jet"), alpha= 0.5)
    plt.colorbar()
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)
    plt.savefig(outFile)