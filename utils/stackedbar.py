import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cstackedbar(values, colors, filename='output.png', width=3, height=10):
 plt.rcParams["figure.figsize"] = (3, 10)
 plt.rcParams["savefig.pad_inches"] = 0

 values = values.split(';')
 colors = colors.split(';')

 if len(values[-1]) == 0:
  values = values[0:-1]

 if len(colors[-1]) == 0:
  colors = colors[0:-1]

 v = 0
 i = 0
 for value in values:
  b = plt.bar(0, float(value), 10, bottom=v, color=( \
    int(colors[i]) / 255, \
    int(colors[i + 1]) / 255,\
    int(colors[i + 2]) / 255, 1))
  v += float(value)
  i += 3

 plt.axis('off')
 plt.autoscale(tight=True)
 plt.savefig(filename)

 im = Image.open(filename)
 im2 = im.rotate(270, expand=True)
 im2.save(filename)
