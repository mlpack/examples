"""
@file generate_images.py
@author Atharva Khandait

Generates jpg files from csv.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""

from PIL import Image
import numpy as np

def ImagesFromCSV(filename,
                  imgShape = (56, 560),
                  destination = 'samples',
                  saveIndividual = False):

    # Import the data into a numpy matrix.
    # sample.shape => (28 * 2, 28 * 5)
    samples = np.genfromtxt(filename, delimiter = ',', dtype = np.uint8)
    tempImage = Image.fromarray(np.reshape(samples, imgShape), 'L')
    print("printing temp images")
    tempImage.save(destination + '/sample0.jpg')

# Save posterior samples.
ImagesFromCSV('gan/samples_csv_files.csv', destination =
'samples_posterior')