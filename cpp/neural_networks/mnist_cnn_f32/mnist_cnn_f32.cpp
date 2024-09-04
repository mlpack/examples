/**
 * An example of using Convolutional Neural Network (CNN) for
 * solving Digit Recognizer problem from Kaggle website.
 *
 * The full description of a problem as well as datasets for training
 * and testing are available here: https://www.kaggle.com/c/digit-recognizer.
 *
 * This example is similar to the mnist_cnn. The main difference is that,
 * this one loads the dataset as a float32 and creates a float32 model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @author Daivik Nema
 */
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

#if ((ENS_VERSION_MAJOR < 2) || ((ENS_VERSION_MAJOR == 2) && (ENS_VERSION_MINOR < 13)))
  #error "need ensmallen version 2.13.0 or later"
#endif

using namespace arma;
using namespace mlpack;
using namespace std;

CEREAL_REGISTER_MLPACK_LAYERS(arma::fmat);

Row<size_t> getLabels(const arma::fmat& predOut)
{
  Row<size_t> predLabels(predOut.n_cols);
  for (uword i = 0; i < predOut.n_cols; ++i)
  {
    predLabels(i) = predOut.col(i).index_max();
  }
  return predLabels;
}

int main()
{
  // Dataset is randomly split into validation
  // and training parts with following ratio.
  constexpr double RATIO = 0.1;

  // Allow 60 passes over the training data, unless we are stopped early by
  // EarlyStopAtMinLoss.
  const int EPOCHS = 60;

  // Number of data points in each iteration of SGD.
  const int BATCH_SIZE = 50;

  // Step size of the optimizer.
  const double STEP_SIZE = 1.2e-3;

  cout << "Reading data ..." << endl;

  // Labeled dataset that contains data for training is loaded from CSV file.
  // Rows represent features, columns represent data points.
  arma::fmat dataset;

  // The original file can be downloaded from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("../../../data/mnist_train.csv", dataset, true);

  // Split the dataset into training and validation sets.
  arma::fmat train, valid;
  data::Split(dataset, train, valid, RATIO);

  // The train and valid datasets contain both - the features as well as the
  // class labels. Split these into separate mats.
  const arma::fmat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1) /
      256.0;
  const arma::fmat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1) /
      256.0;

  // Labels should specify the class of a data point and be in the interval [0,
  // numClasses).

  // Create labels for training and validatiion datasets.
  const arma::fmat trainY = train.row(0);
  const arma::fmat validY = valid.row(0);

  // Specify the NN model. NegativeLogLikelihood is the output layer that
  // is used for classification problem. RandomInitialization means that
  // initial weights are generated randomly in the interval from -1 to 1.
  FFN<NegativeLogLikelihoodType<arma::fmat>, RandomInitialization, arma::fmat> model;

  // Specify the model architecture.
  // In this example, the CNN architecture is chosen similar to LeNet-5.
  // The architecture follows a Conv-ReLU-Pool-Conv-ReLU-Pool-Dense schema. We
  // have used leaky ReLU activation instead of vanilla ReLU. Standard
  // max-pooling has been used for pooling. The first convolution uses 6 filters
  // of size 5x5 (and a stride of 1). The second convolution uses 16 filters of
  // size 5x5 (stride = 1). The final dense layer is connected to a softmax to
  // ensure that we get a valid probability distribution over the output classes

  // Layers schema.
  // 28x28x1 --- conv (6 filters of size 5x5. stride = 1) ---> 24x24x6
  // 24x24x6 --------------- Leaky ReLU ---------------------> 24x24x6
  // 24x24x6 --- max pooling (over 2x2 fields. stride = 2) --> 12x12x6
  // 12x12x6 --- conv (16 filters of size 5x5. stride = 1) --> 8x8x16
  // 8x8x16  --------------- Leaky ReLU ---------------------> 8x8x16
  // 8x8x16  --- max pooling (over 2x2 fields. stride = 2) --> 4x4x16
  // 4x4x16  ------------------- Dense ----------------------> 10

  // Add the first convolution layer.
  model.Add<ConvolutionType<
             NaiveConvolution<ValidConvolution>,
             NaiveConvolution<FullConvolution>,
             NaiveConvolution<ValidConvolution>,
             arma::fmat>>(6,  // Number of output activation maps.
                         5,  // Filter width.
                         5,  // Filter height.
                         1,  // Stride along width.
                         1,  // Stride along height.
                         0,  // Padding width.
                         0   // Padding height.
  );

  // Add first ReLU.
  model.Add<LeakyReLUType<arma::fmat>>();

  // Add first pooling layer. Pools over 2x2 fields in the input.
  model.Add<MaxPoolingType<arma::fmat>>(2, // Width of field.
                        2, // Height of field.
                        2, // Stride along width.
                        2, // Stride along height.
                        true);

  // Add the second convolution layer.
  model.Add<ConvolutionType<
            NaiveConvolution<ValidConvolution>,
            NaiveConvolution<FullConvolution>,
            NaiveConvolution<ValidConvolution>,
            arma::fmat>>(16, // Number of output activation maps.
                         5,  // Filter width.
                         5,  // Filter height.
                         1,  // Stride along width.
                         1,  // Stride along height.
                         0,  // Padding width.
                         0   // Padding height.
  );

  // Add the second ReLU.
  model.Add<LeakyReLUType<arma::fmat>>();

  // Add the second pooling layer.
  model.Add<MaxPoolingType<arma::fmat>>(2, 2, 2, 2, true);

  // Add the final dense layer.
  model.Add<LinearType<arma::fmat>>(10);
  model.Add<LogSoftMaxType<arma::fmat>>();

  model.InputDimensions() = vector<size_t>({ 28, 28 });

  cout << "Start training ..." << endl;

  // Set parameters for the Adam optimizer.
  ens::Adam optimizer(
      STEP_SIZE,  // Step size of the optimizer.
      BATCH_SIZE, // Batch size. Number of data points that are used in each
                  // iteration.
      0.9,        // Exponential decay rate for the first moment estimates.
      0.999, // Exponential decay rate for the weighted infinity norm estimates.
      1e-8,  // Value used to initialise the mean squared gradient parameter.
      EPOCHS * trainX.n_cols, // Max number of iterations.
      1e-8,           // Tolerance.
      true);

  // Train the CNN model. If this is the first iteration, weights are
  // randomly initialized between -1 and 1. Otherwise, the values of weights
  // from the previous iteration are used.
  model.Train(trainX,
              trainY,
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              // Stop the training using Early Stop at min loss.
              ens::EarlyStopAtMinLossType<arma::fmat>(
                  [&](const arma::fmat& /* param */)
                  {
                    double validationLoss = model.Evaluate(validX, validY);
                    cout << "Validation loss: " << validationLoss << "."
                        << endl;
                    return validationLoss;
                  }));

  // Matrix to store the predictions on train and validation datasets.
  arma::fmat predOut;
  // Get predictions on training data points.
  model.Predict(trainX, predOut);
  // Calculate accuracy on training data points.
  Row<size_t> predLabels = getLabels(predOut);
  double trainAccuracy =
      accu(predLabels == trainY) / (double) trainY.n_elem * 100;

  // Get predictions on validation data points.
  model.Predict(validX, predOut);
  predLabels = getLabels(predOut);
  // Calculate accuracy on validation data points.
  double validAccuracy =
      accu(predLabels == validY) / (double) validY.n_elem * 100;

  cout << "Accuracy: train = " << trainAccuracy << "%,"
            << "\t valid = " << validAccuracy << "%" << endl;

  data::Save("model.bin", "model", model, false);

  cout << "Predicting on test set..." << endl;

  // Get predictions on test data points.
  // The original file could be download from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("../../../data/mnist_test.csv", dataset, true);
  const arma::fmat testX = dataset.submat(1, 0, dataset.n_rows - 1, dataset.n_cols - 1)
      / 256.0;
  const arma::fmat testY = dataset.row(0);
  model.Predict(testX, predOut);
  // Calculate accuracy on test data points.
  predLabels = getLabels(predOut);
  double testAccuracy =
      accu(predLabels == testY) / (double) testY.n_elem * 100;

  cout << "Accuracy: test = " << testAccuracy << "%" << endl;

  cout << "Saving predicted labels to \"results.csv.\"..." << endl;
  // Saving results into Kaggle compatible CSV file.
  predLabels.save("results.csv", arma::csv_ascii);

  cout << "Neural network model is saved to \"model.bin\"" << endl;
  cout << "Finished" << endl;
}
