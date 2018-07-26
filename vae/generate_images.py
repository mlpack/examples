from PIL import Image
import numpy as np

def ImagesFromCSV(filename, imgShape = (28, 28), destination = 'samples'):

  # Import the data into a numpy matrix.
  samples = np.genfromtxt(filename, delimiter=',', dtype=np.uint8)

  # Reshape and save it as an image in the destination.
  temp_image = Image.fromarray(np.reshape(samples[:, 0], imgShape), 'L')
  temp_image.save(destination + '/image0.jpg')

  # All the images will be concatenated to this for a combined image.
  image_combined = temp_image

  for i in range(1, samples.shape[1]):

    temp_image = np.reshape(samples[:, i], imgShape)

    image_combined = np.concatenate((image_combined, temp_image), axis=1)

    temp_image = Image.fromarray(temp_image, 'L')
    temp_image.save(destination + '/image' + str(i) + '.jpg')

  image_combined = Image.fromarray(image_combined, 'L')
  image_combined.save(destination + '/image_combined' + '.jpg')

  print (str(samples.shape[1]) + ' samples saved in ' + destination + '/.')


ImagesFromCSV('samples_prior.csv', destination = 'samples_vae_prior')
ImagesFromCSV('samples_posterior.csv', destination = 'samples_vae_posterior')
