#include <ensmallen.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <Kaggle/kaggle_utils.hpp>
#include <ensmallen_bits/callbacks/callbacks.hpp>
#include<models/alexnet.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;


int main()
{
  // Dataset is randomly split into validation
  // and training parts with following ratio.
  constexpr double RATIO = 0.7;

  // Number of iteration per cycle.
  constexpr int EPOCHS = 5;

  // Number of cycles.
  constexpr int CYCLES = 40;

  // Step size of the optimizer.
  constexpr double STEP_SIZE = 1.2e-3;

  // Number of data points in each iteration of SGD.
  constexpr int BATCH_SIZE = 32;

  std::cout << "Reading data ..." << endl;

  // Labeled dataset that contains data for training is loaded from CSV file.
  // Rows represent features, columns represent data points.
  mat tempDataset;

  // The original file can be downloaded from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("./../data/mnist_train.csv", tempDataset, true);

  arma::mat paddings(49408, tempDataset.n_cols);
  paddings.zeros();

  tempDataset.insert_rows(tempDataset.n_rows - 1, paddings);
  // The original Kaggle dataset CSV file has headings for each column,
  // so it's necessary to get rid of the first row. In Armadillo representation,
  // this corresponds to the first column of our data matrix.
  mat dataset = tempDataset.submat(0, 1,
      tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  // Split the dataset into training and validation sets.
  mat train, valid;
  data::Split(dataset, train, valid, RATIO);

  // The train and valid datasets contain both - the features as well as the
  // class labels. Split these into separate mats.
  const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
  const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);

  // According to NegativeLogLikelihood output layer of NN, labels should
  // specify class of a data point and be in the interval from 1 to
  // number of classes (in this case from 1 to 10).

  // Create labels for training and validatiion datasets.
  const mat trainY = train.row(0) + 1;
  const mat validY = valid.row(0) + 1;

  // Specify the NN model. NegativeLogLikelihood is the output layer that
  // is used for classification problem. RandomInitialization means that
  // initial weights are generated randomly in the interval from -1 to 1.
  FFN<NegativeLogLikelihood<>, RandomInitialization> model;

  AlexNet alexnet(1, 224, 224, 10, true);
  Sequential<> *layer = alexnet.GetModel();

  model.Add<IdentityLayer<>>();
  model.Add(layer);

  std::cout << "Training ..." << endl;

  // Set parameters of Stochastic Gradient Descent (SGD) optimizer.
  SGD<AdamUpdate> optimizer(
    // Step size of the optimizer.
    STEP_SIZE,
    // Batch size. Number of data points that are used in each iteration.
    BATCH_SIZE,
    // Max number of iterations.
    EPOCHS * trainY.n_cols,
    // Tolerance, used as a stopping condition. Such a small value
    // means we almost never stop by this condition, and continue gradient
    // descent until the maximum number of iterations is reached.
    1e-8,
    // Shuffle. If optimizer should take random data points from the dataset at
    // each iteration.
    true,
    // Adam update policy.
    AdamUpdate(1e-8, 0.9, 0.999));
  model.Train(trainX,
              trainY,
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              ens::EarlyStopAtMinLoss(1),
              ens::StoreBestCoordinates<arma::mat>());

  // Don't reset optimizers parameters between cycles.
  optimizer.ResetPolicy() = false;

  mat predOut;
  // Getting predictions on training data points.
  model.Predict(trainX, predOut);
  // Calculating accuracy on training data points.
  Row<size_t> predLabels = getLabels(predOut);
  double trainAccuracy = accuracy(predLabels, trainY);
  // Getting predictions on validating data points.
  model.Predict(validX, predOut);
  // Calculating accuracy on validating data points.
  predLabels = getLabels(predOut);
  double validAccuracy = accuracy(predLabels, validY);
  std::cout << "Valid Accuracy: " << validAccuracy << std::endl;

  std::cout << "Predicting ..." << endl;

  // Loading test dataset (the one whose predicted labels
  // should be sent to Kaggle website).
  // As before, it's necessary to get rid of header.

  // The original file could be download from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("./../data/mnist_test.csv", tempDataset, true);
  mat testX = tempDataset.submat(0, 1,
    tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  mat testPredOut;
  // Getting predictions on test data points .
  model.Predict(testX, testPredOut);
  // Generating labels for the test dataset.
  Row<size_t> testPred = getLabels(testPredOut);
  std::cout << "Results Saved." << std::endl;

  return 0;
}