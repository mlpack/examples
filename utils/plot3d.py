import matplotlib as mpl
mpl.use('Agg')

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

def cplot3d(x, y, z, l, xl, yl, zl, mode = 0, filename='output.png', height=10, width=10):
  x = x.strip()
  y = y.strip()
  z = z.strip()
  l = l.strip()

  x = x.split('\n')
  y = y.split('\n')
  z = z.split('\n')
  l = l.split('\n')

  plt.rcParams["figure.figsize"] = (width, height)
  plt.rcParams["savefig.pad_inches"] = 0
  fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
  for i in range(len(x)):
    xSub = x[i].split(';')
    ySub = y[i].split(';')
    zSub = z[i].split(';')

    if len(xSub[-1]) == 0:
      xSub = xSub[0:-1]

    if len(ySub[-1]) == 0:
      ySub = ySub[0:-1]

    if len(zSub[-1]) == 0:
      zSub = zSub[0:-1]

    xSub = [float(i) for i in xSub]
    ySub = [float(i) for i in ySub]
    zSub = [float(i) / 1000 for i in zSub]

    if mode == 0:
      ax.plot(xSub, ySub, zSub, linewidth=1, label=l[i])
    elif mode == 1:
      ax.scatter(xSub, ySub, zSub, label=l[i])
    else:
      ax.plot(xSub, ySub, zSub, linewidth=1, label=l[i])
      ax.scatter(xSub, ySub, zSub)

  ax.set_xlabel(xl)
  ax.set_ylabel(yl)
  ax.set_zlabel(zl)
  ax.legend()

  fig.savefig(filename)
