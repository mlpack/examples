/**
 * An example of using Feed Forward Neural Network (FFN) for
 * solving Digit Recognizer problem from Kaggle website.
 *
 * The full description of a problem as well as datasets for training
 * and testing are available here https://www.kaggle.com/c/digit-recognizer
 * using BatchNorm
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @author Eugene Freyman
 * @author Manthan-R-Sheth
 */

// NOTE: this example does not currently work!  The BatchNorm and PReLU layers
// need to be adapted to the new mlpack 4 style for layers.  (See
// https://github.com/mlpack/mlpack/pull/2777.)

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

#if ((ENS_VERSION_MAJOR < 2) || ((ENS_VERSION_MAJOR == 2) && (ENS_VERSION_MINOR < 13)))
  #error "need ensmallen version 2.13.0 or later"
#endif

using namespace mlpack;
using namespace arma;
using namespace std;

Row<size_t> getLabels(mat& predOut)
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
  // Dataset is randomly split into training
  // and validation parts with following ratio.
  double RATIO = 0.1;

  // The number of neurons in the first layer.
  constexpr int H1 = 100;
  // The number of neurons in the second layer.
  constexpr int H2 = 100;

  // The solution is done in several approaches (CYCLES), so each approach
  // uses previous results as starting point and have a different optimizer
  // options (here the step size is different).

  // Allow infinite number of iterations until we stopped by EarlyStopAtMinLoss
  constexpr int MAX_ITERATIONS = 0;

  // Step size of an optimizer.
  constexpr double STEP_SIZE = 5e-4;

  // Number of data points in each iteration of SGD
  // Power of 2 is better for data parallelism
  size_t BATCH_SIZE = 64;

  // Labeled dataset that contains data for training is loaded from CSV file,
  // rows represent features, columns represent data points.
  mat dataset;
  // The original file could be download from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("../../../data/mnist_train.csv", dataset, true);

  // Splitting the dataset on training and validation parts.
  mat train, valid;
  data::Split(dataset, train, valid, RATIO);

  // Getting training and validating dataset with features only.
  const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
  const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);

  // Labels should specify the class of a data point and be in the interval [0,
  // numClasses).

  // Creating labels for training and validating dataset.
  const mat trainY = train.row(0);
  const mat validY = valid.row(0);

  // Specifying the NN model. NegativeLogLikelihood is the output layer that
  // is used for classification problem. RandomInitialization means that
  // initial weights in neurons are generated randomly in the interval
  // from -1 to 1.
  FFN<NegativeLogLikelihood, RandomInitialization> model;
  // This is intermediate layer that is needed for connection between input
  // data and PRelU layer. Parameters specify the number of input features
  // and number of neurons in the next layer.
  model.Add<Linear>(H1);
  // The first PReLU activation layer. parameter can be set as constructor
  // argument.
  model.Add<PReLU>();
  // BatchNorm layer applied after PReLU activation as it gives
  // better results practically.
  model.Add<BatchNorm>();
  // Intermediate layer between PReLU activation layers.
  model.Add<Linear>(H2);
  // The second PReLU layer.
  model.Add<PReLU>();
  // Second BatchNorm layer
  model.Add<BatchNorm>();
  // Intermediate layer.
  model.Add<Linear>(10);
  // LogSoftMax layer is used together with NegativeLogLikelihood for mapping
  // output values to log of probabilities of being a specific class.
  model.Add<LogSoftMax>();

  cout << "Training ..." << endl;

  // Set parameters for the Adam optimizer.
  ens::Adam optimizer(
      STEP_SIZE,  // Step size of the optimizer.
      BATCH_SIZE, // Batch size. Number of data points that are used in each
                  // iteration.
      0.9,        // Exponential decay rate for the first moment estimates.
      0.999, // Exponential decay rate for the weighted infinity norm estimates.
      1e-8,  // Value used to initialise the mean squared gradient parameter.
      MAX_ITERATIONS, // Max number of iterations.
      1e-8,           // Tolerance.
      true);

  // Train neural network. If this is the first iteration, weights are
  // random, using current values as starting point otherwise.
  model.Train(trainX,
              trainY,
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              // Stop the training using Early Stop at min loss.
              ens::EarlyStopAtMinLoss(
                  [&](const arma::mat& /* param */)
                  {
                    double validationLoss = model.Evaluate(validX, validY);
                    cout << "Validation loss: " << validationLoss
                        << "." << endl;
                    return validationLoss;
                  }));
  mat predOut;
  // Getting predictions on training data points.
  model.Predict(trainX, predOut);
  // Calculating accuracy on training data points.
  Row<size_t> predLabels = getLabels(predOut);
  double trainAccuracy =
      accu(predLabels == trainY) / (double) trainY.n_elem * 100;
  // Getting predictions on validating data points.
  model.Predict(validX, predOut);
  // Calculating accuracy on validating data points.
  predLabels = getLabels(predOut);
  double validAccuracy =
      accu(predLabels == validY) / (double) validY.n_elem * 100;

  cout << "Accuracy: train = " << trainAccuracy << "%,"
            << " valid = " << validAccuracy << "%" << endl;

  data::Save("model.bin", "model", model, false);
  cout << "Predicting ..." << endl;

  // Loading test dataset (the one whose predicted labels
  // should be sent to Kaggle website).
  // As before, it's necessary to get rid of header.

  // The original file could be download from
  // https://www.kaggle.com/c/digit-recognizer/data

  data::Load("../../../data/mnist_test.csv", dataset, true);
  mat testY = dataset.row(0);
  dataset.shed_row(0); // Remove labels.

  mat testPredOut;
  // Getting predictions on test data points .
  model.Predict(dataset, testPredOut);
  // Generating labels for the test dataset.
  Row<size_t> testPred = getLabels(testPredOut);

  double testAccuracy = arma::accu(testPred == testY) /
      (double) testY.n_elem * 100;
  cout << "Accuracy: test = " << testAccuracy << "%" << endl;

  cout << "Saving predicted labels to \"results.csv\"" << endl;
  testPred.save("results.csv", arma::csv_ascii);

  cout << "Neural network model is saved to \"model.bin\"" << endl;
  cout << "Finished" << endl;
}
