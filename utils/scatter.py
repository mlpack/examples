import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import imageio
import numpy as np
import os

def cscatter(x, y, a, c, size, filename='output.gif', height=2000, width=4000):
  x = x.split(';')
  y = y.split(';')
  a = a.split(';')
  c = c.split(';')

  if len(x[-1]) == 0:
   x = x[0:-1]

  if len(y[-1]) == 0:
    y = y[0:-1]

  if len(a[-1]) == 0:
    a = a[0:-1]

  if len(c[-1]) == 0:
    c = c[0:-1]

  x = [float(i) for i in x]
  y = [float(i) for i in y]
  a = [int(i) for i in a]
  c = [float(i) for i in c]

  colors = cm.rainbow(np.linspace(0, 1, 13))
  n = 0
  cn = 0
  im = 0
  images = []
  with imageio.get_writer(filename, mode='I', fps=1) as writer:
    for i in range(0, size):
      fig, ax = plt.subplots()

      cx = []
      cy = []
      for j in range(0, int(len(c) / 2 / size)):
        cx.append(c[cn])
        cy.append(c[cn + 1])
        cn += 2

      color = []
      for j in range(0, len(x)):
        color.append(colors[a[n]])
        n += 1
      ax.scatter(x, y, 5, color=color)

      ax.scatter(cx, cy, 20, color='black', marker='x')
      ax.text(0.05, 0.95, "Iteration - " + str(i), transform=ax.transAxes, fontsize=10, verticalalignment='top')

      plt.savefig('c-' + str(im) + '.png')
      plt.close()

      image = imageio.imread('c-' + str(im) + '.png')
      writer.append_data(image)
      os.remove('c-' + str(im) + '.png')

      im += 1
