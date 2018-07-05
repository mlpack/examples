from PIL import Image
import numpy as np

outputSamples = np.genfromtxt('outputSamples.csv', delimiter=',', dtype=np.uint8)

temp_image = Image.fromarray(np.reshape(outputSamples[:, 0], (28, 28)), 'L')
temp_image.save('generated_samples_2/image0' + '.jpg')

latent_varying = temp_image

for i in range(1, outputSamples.shape[1]):
  temp_image = np.reshape(outputSamples[:, i], (28, 28))

  latent_varying = np.concatenate((latent_varying, temp_image), axis=1)

  temp_image = Image.fromarray(temp_image, 'L')
  temp_image.save('generated_samples_2/image' + str(i) + '.jpg')

latent_varying = Image.fromarray(latent_varying, 'L')
latent_varying.save('generated_samples_2/image_combined' + '.jpg')

print (str(outputSamples.shape[1]) + ' samples saved in generated_samples_2/.')