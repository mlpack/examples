/**
 * Simple Convolutional Neural Network for
 * object classification using CIFAR-10 dataset.
 *
 * @author David Port Louis
 */
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include "periodic_save.hpp"

using namespace mlpack;
using namespace arma;
using namespace std;
using namespace ens;

// Utility function to generate class labels from probabilities.
Row<size_t> getLabels(const mat& yPreds)
{
  Row<size_t> yLabels(yPreds.n_cols);
  for (uword i = 0; i < yPreds.n_cols; ++i)
  {
    yLabels(i) = yPreds.col(i).index_max();
  }
  return yLabels;
}

int main()
{
  // Hyperparameters for optimizer (Feel free to tweak these).

  // Dataset train & validation split ratio.
  const double RATIO = 0.1;

  // Maximum number of iterations to train.
  // MAX_ITERATIONS is set to 10 for reducing CI build time, consider setting it
  // to 200 or more for obtaining fruitful models.
  constexpr int MAX_ITERATIONS = 10;

  // Step size of the optimizer.
  constexpr double STEP_SIZE = 0.002;

  // Number of data points in each iteration of SGD.
  constexpr int BATCH_SIZE = 64;

  // Cifar 10 Dataset containing 3072 features (32 * 32) + labels is loaded from
  // CSV file.
  mat dataset;
  data::Load("../data/cifar-10_train.csv", dataset, true);

  // Header column is dropped.
  dataset.shed_col(0);

  // Split the dataset into training and validation sets.
  mat train, valid;
  data::Split(dataset, train, valid, RATIO);

  // Split the features and labels
  const mat trainX = train.submat(0, 0, train.n_rows - 2, train.n_cols - 1);
  const mat validX = valid.submat(0, 0, valid.n_rows - 2, valid.n_cols - 1);

  const mat trainY = train.row(train.n_rows - 1);
  const mat validY = valid.row(valid.n_rows - 1);

  // Number of iterations, should be equal to the No. of
  // Datapoints seen times the MAX_ITERATIONS
  size_t numIterations = trainX.n_cols * MAX_ITERATIONS;

  // Create the Feed Forward Neural Network with Random weight initalization and
  // NegativeLogLikelihood Loss (NLLLoss).
  FFN<NegativeLogLikelihood, RandomInitialization> model;

  // @Model architecture.
  // 32 x 32 x 3  -- Conv (6 maps with 5 x 5 kernel, stride 1) ---> 28 x 28 x 6
  // 28 x 28 x 6  ------------- Leaky ReLU -----------------------> 28 x 28 x 6
  // 28 x 28 x 6  ---- Max Pooling (2 x 2 kernel, stride 2) ------> 14 x 14 x 6
  // 14 x 14 x 6  -- Conv (16 maps with 5 x 5 kernel, stride 1) --> 10 x 10 x 16
  // 10 x 10 x 16 ------------- Leaky ReLU -----------------------> 10 x 10 x 16
  // 10 x 10 x 16 --- Max Pooling (2 x 2 kernel, stride 2) -------> 5 x 5 x 16
  // 5 x 5 x 16   --------------- Linear -------------------------> 120
  // 120          ------------- Leaky ReLU -----------------------> 120
  // 120          --------------- Linear -------------------------> 84
  // 84           ------------- Leaky ReLU -----------------------> 84
  // 84           --------------- Linear -------------------------> 10
  // 10           ------------- Log Softmax ----------------------> 10

  model.Add<Convolution>(6, 5, 5, 1, 1, 0, 0);
  model.Add<LeakyReLU>();
  model.Add<MaxPooling>(2, 2, 2, 2, true);
  model.Add<Convolution>(16, 5, 5, 1, 1, 0, 0);
  model.Add<LeakyReLU>();
  model.Add<MaxPooling>(2, 2, 2, 2, true);
  model.Add<Linear>(120);
  model.Add<LeakyReLU>();
  model.Add<Linear>(84);
  model.Add<LeakyReLU>();
  model.Add<Linear>(10);
  model.Add<LogSoftMax>();

  // Set the input dimensions to 32x32x3.
  model.InputDimensions() = std::vector<size_t>({ 32, 32, 3 });

  cout << "Start training ..." << endl;

  // Train the CNN model using Adam Optimizer and above hyperparameters.
  Adam optimizer(STEP_SIZE, BATCH_SIZE, 0.9, 0.999, 1e-8, numIterations, 1e-8,
      true);
  model.Train(trainX,
              trainY,
              optimizer,
              PrintLoss(),
              ProgressBar(),
              EarlyStopAtMinLoss(),
              PeriodicSave<decltype(model)>(model, "./new_models2/"));

  cout << "Starting evaluation on training set..." << endl;

  // Matrix to store prediction on train and validation datasets.
  mat predProbs;

  // Make predictions on training data points.
  model.Predict(trainX, predProbs);

  // Convert the predicted probalities into labels.
  Row<size_t> predLabels = getLabels(predProbs);

  // Calculate accuracy on training data points.
  double trainAccuracy = accu(predLabels == trainY) /
      (double) trainY.n_elem * 100;

  cout << "Starting evaluation on validation set..." << endl;

  // Make predictions on validation data points.
  model.Predict(validX, predProbs);

  // Convert the predicted probabilities into labels.
  predLabels = getLabels(predProbs);

  // Calculate accuracy on validation data points.
  double validAccuracy = accu(predLabels == validY) /
      (double) validY.n_elem * 100;

  cout << "Accuracy: train = " << trainAccuracy << "%,"
       << "\t valid = " << validAccuracy <<"%" << endl;

  // Save trained model.
  data::Save("cifarNet.xml", "model", model, false);
}
