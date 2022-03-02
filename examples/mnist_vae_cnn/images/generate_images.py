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
                  imgShape = (28, 28),
                  destination = 'samples',
                  saveIndividual = False):

  # Import the data into a numpy matrix.
  samples = np.genfromtxt(filename, delimiter = ',', dtype = np.uint8)

  # Reshape and save it as an image in the destination.
  tempImage = Image.fromarray(np.reshape(samples[:, 0], imgShape), 'L')
  if saveIndividual:
    tempImage.save(destination + '/sample0.jpg')

  # All the images will be concatenated to this for a combined image.
  allSamples = tempImage

  for i in range(1, samples.shape[1]):
    tempImage = np.reshape(samples[:, i], imgShape)

    allSamples = np.concatenate((allSamples, tempImage), axis = 1)

    tempImage = Image.fromarray(tempImage, 'L')
    if saveIndividual:
      tempImage.save(destination + '/sample' + str(i) + '.jpg')

  tempImage = allSamples
  allSamples = Image.fromarray(allSamples, 'L')
  allSamples.save(destination + '/allSamples' + '.jpg')

  print ('Samples saved in ' + destination + '/.')

  return tempImage

# Save posterior samples.
ImagesFromCSV('./samples_csv_files/samples_posterior.csv', destination =
    'samples_posterior')

# Save prior samples with individual latent varying.
latentSize = 10
allLatent = ImagesFromCSV('./samples_csv_files/samples_prior_latent0.csv',
    destination = 'samples_prior')

for i in range(1, latentSize):
  allLatent = np.concatenate((allLatent,
      (ImagesFromCSV('./samples_csv_files/samples_prior_latent' + str(i) + '.csv',
      destination = 'samples_prior'))), axis = 0)

saved = Image.fromarray(allLatent, 'L')
saved.save('./samples_prior/allLatent.jpg')

# Save prior samples with 2d latent varying.
nofSamples = 20
allLatent = ImagesFromCSV('./samples_csv_files/samples_prior_latent_2d0.csv',
    destination = 'latent')

for i in range(1, nofSamples):
  allLatent = np.concatenate((allLatent,
      (ImagesFromCSV('./samples_csv_files/samples_prior_latent_2d' + str(i) +
      '.csv', destination = 'samples_prior'))), axis = 0)

saved = Image.fromarray(allLatent, 'L')
saved.save('./samples_prior/2dLatent.jpg')
