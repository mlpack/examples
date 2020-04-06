/**
 * An example of using Convolutional Neural Network (CNN) for
 * solving Digit Recognizer problem from Kaggle website.
 *
 * The full description of a problem as well as datasets for training
 * and testing are available here https://www.kaggle.com/c/digit-recognizer
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @author Daivik Nema
 */

#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;

using namespace arma;
using namespace std;

using namespace ens;

arma::Row<size_t> getLabels(arma::mat predOut)
{
  arma::Row<size_t> predLabels(predOut.n_cols);
  for (arma::uword i = 0; i < predOut.n_cols; ++i)
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

  // Allow infinite number of iterations until we stopped by EarlyStopAtMinLoss
  constexpr int MAX_ITERATIONS = 0;

  // Step size of the optimizer.
  constexpr double STEP_SIZE = 1.2e-3;

  // Number of data points in each iteration of SGD.
  constexpr int BATCH_SIZE = 50;

  cout << "Reading data ..." << endl;

  // Labeled dataset that contains data for training is loaded from CSV file.
  // Rows represent features, columns represent data points.
  mat tempDataset;

  // The original file can be downloaded from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("../data/train.csv", tempDataset, true);

  // The original Kaggle dataset CSV file has headings for each column,
  // so it's necessary to get rid of the first row. In Armadillo representation,
  // this corresponds to the first column of our data matrix.
  mat dataset =
      tempDataset.submat(0, 1, tempDataset.n_rows - 1, tempDataset.n_cols - 1);

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
  model.Add<Convolution<>>(1,  // Number of input activation maps.
                           6,  // Number of output activation maps.
                           5,  // Filter width.
                           5,  // Filter height.
                           1,  // Stride along width.
                           1,  // Stride along height.
                           0,  // Padding width.
                           0,  // Padding height.
                           28, // Input width.
                           28  // Input height.
  );

  // Add first ReLU.
  model.Add<LeakyReLU<>>();

  // Add first pooling layer. Pools over 2x2 fields in the input.
  model.Add<MaxPooling<>>(2, // Width of field.
                          2, // Height of field.
                          2, // Stride along width.
                          2, // Stride along height.
                          true);

  // Add the second convolution layer.
  model.Add<Convolution<>>(6,  // Number of input activation maps.
                           16, // Number of output activation maps.
                           5,  // Filter width.
                           5,  // Filter height.
                           1,  // Stride along width.
                           1,  // Stride along height.
                           0,  // Padding width.
                           0,  // Padding height.
                           12, // Input width.
                           12  // Input height.
  );

  // Add the second ReLU.
  model.Add<LeakyReLU<>>();

  // Add the second pooling layer.
  model.Add<MaxPooling<>>(2, 2, 2, 2, true);

  // Add the final dense layer.
  model.Add<Linear<>>(16 * 4 * 4, 10);
  model.Add<LogSoftMax<>>();

  cout << "Start training ..." << endl;

  // Set parameters of Stochastic Gradient Descent (SGD) optimizer.
  SGD<AdamUpdate> optimizer(
      // Step size of the optimizer.
      STEP_SIZE,
      // Batch size. Number of data points that are used in each iteration.
      BATCH_SIZE,
      // Max number of iterations.
      MAX_ITERATIONS,
      // Tolerance, used as a stopping condition. Such a small value
      // means we almost never stop by this condition, and continue gradient
      // descent until the maximum number of iterations is reached.
      -1,
      // Shuffle. If optimizer should take random data points from the dataset
      // at each iteration.
      true,
      // Adam update policy.
      AdamUpdate(1e-8, 0.9, 0.999));

  // Train the CNN model. If this is the first iteration, weights are
  // randomly initialized between -1 and 1. Otherwise, the values of weights
  // from the previous iteration are used.
  model.Train(trainX,
              trainY,
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              // Stop the training using Early Stop at min loss.
              ens::EarlyStopAtMinLoss());

  // Matrix to store the predictions on train and validation datasets.
  mat predOut;
  // Get predictions on training data points.
  model.Predict(trainX, predOut);
  // Calculate accuracy on training data points.
  arma::Row<size_t> predLabels = getLabels(predOut);
  double trainAccuracy = arma::accu(predLabels == trainY) / trainY.n_elem;
  // Get predictions on validating data points.
  model.Predict(validX, predOut);
  // Calculate accuracy on validating data points.
  predLabels = getLabels(predOut);
  double validAccuracy = arma::accu(predLabels == validY) / validY.n_elem;

  std::cout << "Accuracy: train = " << trainAccuracy << "%,"
            << "\t valid = " << validAccuracy << "%" << std::endl;

  mlpack::data::Save("model.bin", "model", model, false);

  std::cout << "Predicting ..." << std::endl;

  // Load test dataset
  // The original file could be download from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("../data/test.csv", tempDataset, true);

  // As before, it's necessary to get rid of column headings.
  mat testX =
      tempDataset.submat(0, 1, tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  // Matrix to store the predictions on test dataset.
  mat testPredOut;
  // Get predictions on test data points.
  model.Predict(testX, testPredOut);
  // Generate labels for the test dataset.
  Row<size_t> testPred = getLabels(testPredOut);
  std::cout << "Saving predicted labels to \"results.csv.\"..." << std::endl;

  // Saving results into Kaggle compatibe CSV file.
  testPred.save("results.csv", arma::csv_ascii);
  std::cout << "Neural network model is saved to \"model.bin\"" << std::endl;
  std::cout << "Finished" << std::endl;
}
