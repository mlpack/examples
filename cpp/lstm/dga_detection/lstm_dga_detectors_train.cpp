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
  const size_t numEpochs = 5;
  ens::Adam opt(0.001 /* step size */,
                1 /* batch size */,
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

  string line;
  vector<string> benignDomains, maliciousDomains;
  size_t maxBenignDomainLen = 0, maxMaliciousDomainLen = 0, lineNum = 0;
  while (f.good())
  {
    getline(f, line);

    // Skip empty lines.
    if (line.size() == 0)
      continue;

    // Skip header line.
    if (lineNum == 0 && line.substr(0, 5) == "label")
      continue;

    // Find the length of the domain.
    size_t commaPos = line.find(',');
    if (commaPos == string::npos)
    {
      cerr << "Invalid domain on line " << lineNum << ": " << line << endl;
      cerr << "Expected format:" << endl;
      cerr << "  malware,domain.com" << endl;
      cerr << "  benign,domain.org" << endl;
      exit(1);
    }

    const size_t domainLen = line.size() - commaPos - 1;
    if (line.substr(0, commaPos) == "benign")
    {
      benignDomains.push_back(line.substr(commaPos + 1));
      maxBenignDomainLen = std::max(maxBenignDomainLen, domainLen);
    }
    else if (line.substr(0, commaPos) == "malicious")
    {
      maliciousDomains.push_back(line.substr(commaPos + 1));
      maxMaliciousDomainLen = std::max(maxMaliciousDomainLen, domainLen);
    }
    else
    {
      cerr << "Invalid label on line " << lineNum << ": "
          << line.substr(0, commaPos) << endl;
      exit(1);
    }

    ++lineNum;
  }

  cout << "File '" << argv[1] << "' has " << benignDomains.size() << " benign "
      << "domains with a maximum length of " << maxBenignDomainLen << ", and "
      << maliciousDomains.size() << " malicious domains with a maximum length "
      << "of " << maxMaliciousDomainLen << "." << endl;

  // Create the datasets and build them.
  arma::cube benignDataset(38, benignDomains.size(), maxBenignDomainLen,
      arma::fill::zeros);
  arma::cube benignResponses(1, benignDomains.size(), maxBenignDomainLen,
      arma::fill::none);
  arma::uvec benignLengths(benignDomains.size());

  arma::cube maliciousDataset(38, maliciousDomains.size(),
      maxMaliciousDomainLen, arma::fill::zeros);
  arma::cube maliciousResponses(1, maliciousDomains.size(),
      maxMaliciousDomainLen, arma::fill::none);
  arma::uvec maliciousLengths(maliciousDomains.size());

  // Build the benign dataset.
  for (size_t i = 0; i < benignDomains.size(); ++i)
  {
    const std::string& domain = benignDomains[i];

    // One-hot encode each character.
    for (size_t t = 0; t < domain.size(); ++t)
    {
      char c = tolower(domain[t]);
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
        cerr << "Domain '" << domain << "' has invalid character '" << domain[t]
            << "'!" << endl;
        exit(1);
      }

      if (t < domain.size() - 1)
        benignDataset(dim, i, t) = 1.0;
      if (t > 0)
        benignResponses(0, i, t - 1) = dim;
    }

    benignLengths[i] = domain.size();
  }

  // Build the malicious dataset.
  for (size_t i = 0; i < maliciousDomains.size(); ++i)
  {
    const std::string& domain = maliciousDomains[i];

    // One-hot encode each character.
    for (size_t t = 0; t < domain.size(); ++t)
    {
      char c = tolower(domain[t]);
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
        cerr << "Domain '" << domain << "' has invalid character '" << domain[t]
            << "'!" << endl;
        exit(1);
      }

      if (t < domain.size() - 1)
        maliciousDataset(dim, i, t) = 1.0;
      if (t > 0)
        maliciousResponses(0, i, t - 1) = dim;
    }

    maliciousLengths[i] = domain.size();
  }

  // The mlpack RNN class expects a cube representing the input at each time
  // step, and representing the output at each time step.  Therefore, we will
  // split our dataset so that the output is the input plus one time step.

  // We'll also hold out 10% of the data as a test set.
  size_t k = 0.9 * benignDataset.n_cols;
  size_t n = benignDataset.n_cols;
  arma::cube trainBenignInput = benignDataset.cols(0, k - 1);
  arma::cube trainBenignResponses = benignResponses.cols(0, k - 1);
  arma::uvec trainBenignLengths = benignLengths.subvec(0, k - 1);
  arma::cube testBenignInput = benignDataset.cols(k, n - 1);
  arma::cube testBenignResponses = benignResponses.cols(k, n - 1);
  arma::uvec testBenignLengths = benignLengths.subvec(k, n - 1);

  k = 0.9 * maliciousDataset.n_cols;
  n = maliciousDataset.n_cols;
  arma::cube trainMaliciousInput = maliciousDataset.cols(0, k - 1);
  arma::cube trainMaliciousResponses = maliciousResponses.cols(0, k - 1);
  arma::uvec trainMaliciousLengths = maliciousLengths.subvec(0, k - 1);
  arma::cube testMaliciousInput = maliciousDataset.cols(k, n - 1);
  arma::cube testMaliciousResponses = maliciousResponses.cols(k, n - 1);
  arma::uvec testMaliciousLengths = maliciousLengths.subvec(k, n - 1);

  // Now we have loaded the dataset.  The next step is to build the network.
  // The network is configured in single-response mode, so the entire sequence
  // is read in and then a prediction is made.
  RNN<NegativeLogLikelihood, RandomInitialization> benignModel(
      benignDataset.n_slices);
  benignModel.Add<LSTM>(lstmUnits);
  benignModel.Add<Linear>(39);
  benignModel.Add<LogSoftMax>();

  RNN<NegativeLogLikelihood, RandomInitialization> maliciousModel(
      maliciousDataset.n_slices);
  maliciousModel.Add<LSTM>(lstmUnits);
  maliciousModel.Add<Linear>(39);
  maliciousModel.Add<LogSoftMax>();

  opt.MaxIterations() = numEpochs * trainBenignInput.n_cols;
  benignModel.Train(trainBenignInput, trainBenignResponses, trainBenignLengths,
      opt, ens::ProgressBar());
  opt.MaxIterations() = numEpochs * trainMaliciousInput.n_cols;
  maliciousModel.Train(trainMaliciousInput, trainMaliciousResponses,
      trainMaliciousLengths, opt, ens::ProgressBar());

  // Compute performance metrics on the training set and on the test set.
  arma::cube trainBenignPredictions, testBenignPredictions,
      trainMaliciousPredictions, testMaliciousPredictions;
  benignModel.Predict(trainBenignInput, trainBenignPredictions,
      trainBenignLengths, 1);
  maliciousModel.Predict(trainBenignInput, trainMaliciousPredictions,
      trainBenignLengths, 1);
  benignModel.Predict(testBenignInput, testBenignPredictions,
      testBenignLengths, 1);
  maliciousModel.Predict(testBenignInput, testMaliciousPredictions,
      testMaliciousLengths, 1);

  size_t trainCorrect = 0;
  for (size_t i = 0; i < trainBenignInput.n_cols; ++i)
  {
    // Compute the likelihood of the sequence being generated by each RNN.
    double benLikelihood = 0.0;
    double malLikelihood = 0.0;
    size_t steps = trainBenignLengths[i];
    for (size_t t = 0; t < steps; ++t)
    {
      const size_t trueDim = trainBenignResponses(0, i, t);
      benLikelihood += trainBenignPredictions(trueDim, i, t);
      malLikelihood += trainMaliciousPredictions(trueDim, i, t);
    }

    const bool predMalicious = (malLikelihood > benLikelihood);
    if (!predMalicious)
      ++trainCorrect;
  }

  size_t testCorrect = 0;
  for (size_t i = 0; i < testBenignInput.n_cols; ++i)
  {
    // Compute the likelihood of the sequence being generated by each RNN.
    double benLikelihood = 0.0;
    double malLikelihood = 0.0;
    const size_t steps = testMaliciousLengths[i];
    for (size_t t = 0; t < steps; ++t)
    {
      const size_t trueDim = testBenignResponses(0, i, t);
      benLikelihood += testBenignPredictions(trueDim, i, t);
      malLikelihood += testMaliciousPredictions(trueDim, i, t);
    }

    const bool predMalicious = (malLikelihood > benLikelihood);
    if (!predMalicious)
      ++testCorrect;
  }

  benignModel.Predict(trainMaliciousInput, trainBenignPredictions,
      trainMaliciousLengths, 1);
  maliciousModel.Predict(trainMaliciousInput, trainMaliciousPredictions,
      trainMaliciousLengths, 1);
  benignModel.Predict(testMaliciousInput, testBenignPredictions,
      testMaliciousLengths, 1);
  maliciousModel.Predict(testMaliciousInput, testMaliciousPredictions,
      testMaliciousLengths, 1);

  for (size_t i = 0; i < trainMaliciousInput.n_cols; ++i)
  {
    // Compute the likelihood of the sequence being generated by each RNN.
    double benLikelihood = 0.0;
    double malLikelihood = 0.0;
    size_t steps = trainMaliciousLengths[i];
    for (size_t t = 0; t < steps; ++t)
    {
      const size_t trueDim = trainBenignResponses(0, i, t);
      benLikelihood += trainBenignPredictions(trueDim, i, t);
      malLikelihood += trainMaliciousPredictions(trueDim, i, t);
    }

    const bool predMalicious = (malLikelihood > benLikelihood);
    if (predMalicious)
      ++trainCorrect;
  }

  for (size_t i = 0; i < testMaliciousInput.n_cols; ++i)
  {
    // Compute the likelihood of the sequence being generated by each RNN.
    double benLikelihood = 0.0;
    double malLikelihood = 0.0;
    const size_t steps = testMaliciousLengths[i];
    for (size_t t = 0; t < steps; ++t)
    {
      const size_t trueDim = testBenignResponses(0, i, t);
      benLikelihood += testBenignPredictions(trueDim, i, t);
      malLikelihood += testMaliciousPredictions(trueDim, i, t);
    }

    const bool predMalicious = (malLikelihood > benLikelihood);
    if (predMalicious)
      ++testCorrect;
  }

  const size_t numTrainDomains = trainBenignInput.n_cols +
      trainMaliciousInput.n_cols;
  const size_t numTestDomains = testBenignInput.n_cols +
      testMaliciousInput.n_cols;

  cout << "Model performance:" << endl;
  cout << "  Training accuracy: " << trainCorrect << " of " << numTrainDomains
      << " correct ("
      << (100.0 * double(trainCorrect) / double(numTrainDomains)) << "%)."
      << endl;
  cout << "  Test accuracy:     " << testCorrect << " of " << numTestDomains
      << " correct ("
      << (100.0 * double(testCorrect) / double(numTestDomains)) << "%)."
      << endl;

  // Save the trained model.
  data::Save("lstm_dga_detector_benign.bin",
             "lstm_model",
             benignModel,
             true /* fatal on failure */);
  data::Save("lstm_dga_detector_malicious.bin",
             "lstm_model",
             maliciousModel,
             true /* fatal on failure */);
}
