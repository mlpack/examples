import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import mpimg

def cpandasscatter(inFile, x, y, outFile= 'output.png', height=10, width=10):
    dataset = pd.read_csv(inFile)
    fig = dataset.plot(kind="scatter", x=x, y=y, alpha=0.1, figsize=(width, height))
    fig.figure.savefig(outFile)

def cpandasscattercolor(inFile, x, y, label, c, outFile= 'output1.png', height= 10, width = 10):
    dataset = pd.read_csv(inFile)
    fig = dataset.plot(kind="scatter", x=x, y=y, alpha=0.4,s=dataset["population"]/100,
                        label=label, c=c, cmap=plt.get_cmap("jet"), colorbar=True, 
                        sharex = False)
    fig.figure.savefig(outFile)

def cpandasscattermap(inFile, imgFile, x, y, label, c, outFile="output2.png", height=10, width = 7):
    img_file = mpimg.imread(imgFile);
    dataset = pd.read_csv(inFile);
    ax = dataset.plot(kind = "scatter", x=x, y=y, alpha= 0.4, s=dataset["population"]/100, label= label, c=c, cmap=plt.get_cmap("jet"), colorbar=True, sharex= False)
    plt.imshow(img_file, extent=[-124.55,-113.80, ])