/**
 * @file lstm_dga_detection_train.cpp
 * @author Ryan Curtin
 *
 * This program trains a DGA (domain generation algorithm) detector using a
 * simple LSTM-based RNN.  The model is trained and then saved to disk.  The
 * lstm_dga_detection_predict.cpp program can be used for computing predictions.
 *
 * As input, provide a set of DGA domains in a file, in the following format:
 *
 * ```
 * label,domain
 * malicious,baddomain.net
 * benign,mlpack.org
 * ...
 * ```
 */
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

using namespace mlpack;
using namespace std;

int main(int argc, char** argv)
{
  // Configuration options are collected here.  Change these if you want to
  // change the behavior of the example.
  const size_t lstmUnits = 50;
  const size_t linearUnits = 100;
  const size_t numEpochs = 10;
  ens::StandardSGD opt(0.00001 /* step size */,
                       64 /* batch size */,
                       0 /* number of iterations: this is set later! */,
                       1e-5 /* termination tolerance */);

  // Create the dataset and allocate the space for it.
  size_t numPoints = 10000;
  arma::cube dataset(2, numPoints, 5, arma::fill::zeros);
  arma::cube responses(1, numPoints, 1, arma::fill::zeros);

  for (size_t i = 0; i < numPoints; ++i)
  {
    size_t result = (Random() > 0.5) ? 1 : 0;

    for (size_t t = 0; t < 5; ++t)
    {
      dataset(result, i, t) = 1.0;
    }

    responses(0, i, 0) = result;
  }

  // The mlpack RNN class expects a cube representing the input at each time
  // step, and representing the output at each time step.  Therefore, we will
  // split our dataset so that the output is the input plus one time step.
  //
  // We'll also hold out 10% of the data as a test set.
  size_t numTrainPoints = 0.9 * dataset.n_cols;
  arma::cube trainData = dataset.cols(0, numTrainPoints - 1);
  arma::cube trainResponses = responses.cols(0, numTrainPoints - 1);
  arma::cube testData = dataset.cols(numTrainPoints, dataset.n_cols - 1);
  arma::cube testResponses = responses.cols(numTrainPoints,
      responses.n_cols - 1);

  // Now we have loaded the dataset.  The next step is to build the network.
  // The network is configured in single-response mode, so the entire sequence
  // is read in and then a prediction is made.
  RNN<NegativeLogLikelihood, RandomInitialization> network(dataset.n_slices, true);
  network.Add<LSTM>(lstmUnits);
  //network.Add<Linear>(linearUnits);
  network.Add<ReLU>();
  network.Add<Linear>(linearUnits);
  network.Add<ReLU>();
  network.Add<Linear>(2);
  network.Add<LogSoftMax>();

  opt.MaxIterations() = numEpochs * trainData.n_cols;
  network.Train(trainData,
                trainResponses,
                opt,
                ens::ProgressBar(),
                ens::EarlyStopAtMinLoss());

  // Compute performance metrics on the training set and on the test set.
  arma::cube trainPredictions, testPredictions;
  network.Predict(trainData, trainPredictions);
  network.Predict(testData, testPredictions);

  size_t trainCorrect = 0;
  for (size_t i = 0; i < trainData.n_cols; ++i)
  {
    std::cout << "prediction: " << trainPredictions(0, i, 0) << " vs " << trainPredictions(1, i, 0) << ", response " << trainResponses(0, i, 0) << "\n";
    const bool pred1 = trainPredictions(1, i, 0) >= trainPredictions(0, i, 0);
    const bool is1 = (trainResponses(0, i, 0) == 1.0);
    if (pred1 == is1)
      ++trainCorrect;
  }

  size_t testCorrect = 0;
  for (size_t i = 0; i < testData.n_cols; ++i)
  {
    const bool pred1 = testPredictions(1, i, 0) >= testPredictions(0, i, 0);
    const bool is1 = (testResponses(0, i, 0) == 1.0);
    if (pred1 == is1)
      ++testCorrect;
  }

  cout << "Model performance:" << endl;
  cout << "  Training accuracy: " << trainCorrect << " of " << numTrainPoints
      << " correct ("
      << (100.0 * double(trainCorrect) / double(numTrainPoints)) << "%)."
      << endl;
  cout << "  Test accuracy:     " << testCorrect << " of " << testData.n_cols
      << " correct ("
      << (100.0 * double(testCorrect) / double(testData.n_cols)) << "%)."
      << endl;

  // Save the trained model.
  data::Save("lstm_dga_detector.bin",
             "lstm_model",
             network,
             true /* fatal on failure */);
}
