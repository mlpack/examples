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
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " domains.csv" << endl;
    exit(1);
  }

  // Configuration options are collected here.  Change these if you want to
  // change the behavior of the example.
  const size_t lstmUnits = 250;
  const size_t linearUnits = 100;
  const size_t numEpochs = 30;
  ens::Adam opt(0.005 /* step size */,
                32 /* batch size */,
                0.9, 0.999, 1e-8,
                0 /* number of iterations: this is set later! */,
                1e-5 /* termination tolerance */);

  // The first step is to load the file and parse each domain.
  //
  // We will one-hot encode the letters of each domain, and this will be our
  // input to the network.  This means that our input matrix will be an
  // arma::cube of size:
  //
  //   rows = 41 (26 lowercase letters, 10 digits, period, hyphen, end,
  //              2 labels)
  //   columns = number of domain names
  //   slices = maximum length of domain name + 1
  //
  // For a given domain name (column), each slice will represent a one-hot
  // encoded character in the domain name.  In the slice following the last
  // character of the domain name, we will append a one-hot encoded label
  // (either benign or malicious).
  //
  // We can't use mlpack's loaders directly here, so we will load manually.
  // The first step is to find the maximum domain name length and the number of
  // domain names in the file.
  fstream f(argv[1]);
  if (!f.is_open())
  {
    cerr << "Could not open dataset '" << argv[1] << "'!" << endl;
    exit(1);
  }

  size_t numDomains = 0;
  size_t maxDomainLength = 0;
  string line;
  while (f.good())
  {
    getline(f, line);

    // Skip empty lines.
    if (line.size() == 0)
      continue;

    // Skip header line.
    if (numDomains == 0 && line.substr(0, 5) == "label")
      continue;

    // Find the length of the domain.
    size_t commaPos = line.find(',');
    if (commaPos == string::npos)
    {
      cerr << "Invalid domain on line " << numDomains + 2 << ": " << line
          << endl;
      cerr << "Expected format:" << endl;
      cerr << "  malware,domain.com" << endl;
      cerr << "  benign,domain.org" << endl;
      exit(1);
    }
    else
    {
      maxDomainLength = max(maxDomainLength, line.size() - commaPos - 1);
    }

    ++numDomains;
  }

  cout << "File '" << argv[1] << "' has " << numDomains << " domains with a "
      << "maximum length of " << maxDomainLength << " characters." << endl;

  // Create the dataset and allocate the space for it.
  arma::cube dataset(39, numDomains, maxDomainLength + 2, arma::fill::zeros);
  arma::cube responses(2, numDomains, maxDomainLength + 2, arma::fill::zeros);
  arma::uvec domainLens(numDomains);

  dataset.shed_slices(33, dataset.n_slices - 1);

  // Now take a second pass over the data file to actually do the loading.
  f.close();
  f.open(argv[1]);
  if (!f.is_open())
  {
    cerr << "Could not open dataset '" << argv[1] << "'!" << endl;
  }

  size_t i = 0;
  while (f.good())
  {
    getline(f, line);

    // Skip empty lines.
    if (line.size() == 0)
      continue;

    // Skip header line.
    if (i == 0 && line.substr(0, 5) == "label")
      continue;

    // Split into label and domain.  No error checking needed, because if we
    // passed on the first pass then the second pass should be fine.
    size_t isMalicious = 0;
    size_t commaPos = line.find(',');
    if (line.substr(0, commaPos) == "malicious")
      isMalicious = 1;
    else if (line.substr(0, commaPos) != "benign")
    {
      cerr << "Invalid label on line " << numDomains + 2 << ": "
          << line.substr(0, commaPos) << endl;
      exit(1);
    }

    // One-hot encode all of the characters.
    const size_t domainLen = std::min(line.size() - commaPos - 1, (size_t) dataset.n_slices - 1);
    for (size_t t = 0; t < domainLen; ++t)
    {
      char c = tolower(line[commaPos + 1 + t]);
      size_t dim = 0;
      if (c >= 'a' && c <= 'z')
        dim = size_t(c - 'a');
      else if (c >= '0' && c <= '9')
        dim = 26 + size_t(c - '0');
      else if (c == '-')
        dim = 36;
      else if (c == '.')
        dim = 37;
      else
      {
        cerr << "Domain has invalid character on line " << i + 2 << ": "
            << line.substr(commaPos + 1) << endl;
        exit(1);
      }

      dataset(dim, i, t) = 1.0;
    }

    // Mark all subsequent time steps as end-of-input.
    for (size_t t = domainLen; t < dataset.n_slices; ++t)
      dataset(38, i, t) = 1.0; // Denote end of input.

    // Now encode the label.
    if (isMalicious)
      responses.tube(1, i).fill(1.0);
    else
      responses.tube(0, i).fill(1.0);

    ++i; // Move to next row.
  }

  // The mlpack RNN class expects a cube representing the input at each time
  // step, and representing the output at each time step.  Therefore, we will
  // split our dataset so that the output is the input plus one time step.
  //
  // We'll also hold out 10% of the data as a test set.
  size_t numTrainDomains = 0.9 * dataset.n_cols;
  arma::cube trainData = dataset.cols(0, numTrainDomains - 1);
  arma::cube trainResponses = responses.cols(0, numTrainDomains - 1);
  arma::cube testData = dataset.cols(numTrainDomains, dataset.n_cols - 1);
  arma::cube testResponses = responses.cols(numTrainDomains,
      responses.n_cols - 1);

  // Now we have loaded the dataset.  The next step is to build the network.
  // The network is configured in single-response mode, so the entire sequence
  // is read in and then a prediction is made.
  RNN<MeanSquaredError, RandomInitialization> network(8,
      false, MeanSquaredError(), RandomInitialization(-0.3, 0.3));
  network.Add<LSTM>(lstmUnits);
  network.Add<Sigmoid>();
  network.Add<Linear>(2);
  network.Add<Sigmoid>();

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
    const bool predMalicious = trainPredictions(1, i, 0) >= trainPredictions(0, i, 0);
    const bool isMalicious = (trainResponses(1, i, 0) == 1.0);
    if (predMalicious == isMalicious)
      ++trainCorrect;
  }

  size_t testCorrect = 0;
  for (size_t i = 0; i < testData.n_cols; ++i)
  {
    const bool predMalicious = testPredictions(1, i, 0) >= testPredictions(0, i, 0);
    const bool isMalicious = (testResponses(1, i, 0) == 1.0);
    //std::cout << "prediction: " << testPredictions(0, i, 0) << ", " << testPredictions(1, i, 0) << " vs. response "
    //    << testResponses(0, i, 0) << endl;
    if (predMalicious == isMalicious)
      ++testCorrect;
  }

  cout << "Model performance:" << endl;
  cout << "  Training accuracy: " << trainCorrect << " of " << numTrainDomains
      << " correct ("
      << (100.0 * double(trainCorrect) / double(numTrainDomains)) << "%)."
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
