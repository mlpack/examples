# MNIST

### DESCRIPTION OF THE DATASET

MNIST dataset contains images of handwritten digits. It is one of the most common datasets used for image classifcation. It has 60,000 28x28 grayscale images under the training set and 10,000 28x28 grayscale images under the test set. Each pixel has a value between 0 and 255.

![alt text](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png "Sample Images From The MNIST Database")
*Sample Images From The MNIST Database*

![alt text](https://miro.medium.com/max/490/1*nlfLUgHUEj5vW7WVJpxY-g.png "A visualization of the sample image at index 7777")
*A visualization of the sample image at index 7777*

### Processing The Dataset

##### 1. Loading The Dataset From The File To A Useable Format
The Load Class is used to load a matrix from the file.The type of file is guessed automatically from the extension. For a list of supported file types please refer to [Load Class](https://github.com/mlpack/mlpack/blob/master/src/mlpack/core/data/load.hpp). Please note that since mlpack requires matrices to be in column major format and usually data is stored in a row major format, the matrix by default is transposed. In case the data is originally in column major format, consider setting the parameter to False. Refer to documentation of Load Class for further information.
After loading the dataset is stored in the *tempDataset* Matrix.

##### 2. Splitting The Dataset
The Split Class is used to split the Dataset into Train and Validation Datasets. The train dataset is used in the training procedure, the weights are updated and adjusted to fit this dataset. The validation dataset is used to constantly evaludate the model, which is used to adjust the model hyperparamters like the learning rate. The data is split randomly and RATIO defines the amount of data in Validation Set.
After splitting the train dataset is stored in the *train* matrix and validation in the *valid* matrix.

### Results Obtained on Different Models

##### 1. DigitRecognizer
For More Details related to the Model, check out the **DigitRecognizer** Directory.
After running the model for 50 cycles, I was able to obtain the best validation set accuracy of **97.2143%**.

##### 2. DigitRecognizerBatchNorm
For More Details related to the Model, check out the **DigitRecognizerBatchNorm** Directory.
After running the model for 50 cycles, I was able to obtain the best validation set accuracy of **91.5952%**.

##### 3. DigitRecognizerCNN
For More Details related to the Model, check out the **DigitRecognizerCNN** Directory.
After running the model for 20 cycles, I was able to obtain the best validation set accuracy of **92.8095%**.