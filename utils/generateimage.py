"""
@file generate_images.py
@author Roshan Swain
Generates jpg files from csv.
mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def cgenerateimage(inFile, outFile = 'output.png'):
    dataset = np.genfromtxt(inFile, delimiter = ',', dtype = np.uint8)
    im = Image.fromarray(dataset)
    im.save(outFile)

