/**
 * @file lstm_dga_detection_predict.cpp
 * @author Ryan Curtin
 *
 * Given two trained DGA detection RNNs, make predictions.  Domains should be
 * input on stdin.
 *
 * Predictions are made by computing the likelihood of a domain coming from the
 * benign model and from the malicious model.  The predicted class is benign, if
 * the likelihood of the domain coming from the benign model is higher (and
 * malicious if vice versa).
 *
 * This is called the generalized likelihood ratio test (GLRT).
 *
 * To keep the model small and the code fast, we use `float` as a datatype
 * instead of the default `double`.
 */

// This must be defined to avoid RNN::serialize() throwing an error---we know
// what we are doing and have manually registered the layer types we care about.
#define MLPACK_ANN_IGNORE_SERIALIZATION_WARNING
#include <mlpack.hpp>

// To keep compilation time and program size down, we only register
// serialization for layers used in our RNNs.  Plus, given that we are using
// floats instead of doubles for our data, we need to register the layers for
// serialization either individually or all of them with
// CEREAL_REGISTER_MLPACK_LAYERS() (commented out below).
CEREAL_REGISTER_TYPE(mlpack::Layer<arma::fmat>);
CEREAL_REGISTER_TYPE(mlpack::MultiLayer<arma::fmat>);
CEREAL_REGISTER_TYPE(mlpack::RecurrentLayer<arma::fmat>);
CEREAL_REGISTER_TYPE(mlpack::LSTMType<arma::fmat>);
CEREAL_REGISTER_TYPE(mlpack::LinearType<arma::fmat>);
CEREAL_REGISTER_TYPE(mlpack::LogSoftMaxType<arma::fmat>);

// This will register all mlpack layers with the arma::fmat type.
// It is useful for playing around with the network architecture, but can make
// compilation time a lot longer.  Comment out the individual
// CEREAL_REGISTER_TYPE() calls above if you use the line below.
//
// CEREAL_REGISTER_MLPACK_LAYERS(arma::fmat);

using namespace mlpack;
using namespace std;

// Utility function: map a character to the one-hot encoded dimension that
// represents it.  Characters will be assigned to one of 38 dimensions.  If an
// incorrect character is given, size_t(-1) is returned. 
inline size_t CharToDim(const char inC)
{
  char c = tolower(inC);
  if (c >= 'a' && c <= 'z')
    return size_t(c - 'a'); 
  else if (c >= '0' && c <= '9')
    return 26 + size_t(c - '0');
  else if (c == '-')
    return 36;
  else if (c == '.')
    return 37;
  else
    return size_t(-1);
}

// Utility function: turn a domain string into an arma::cube.
inline void PrepareString(const string& domain,
                          arma::fcube& data,
                          arma::fcube& response)
{
  data.zeros(39, 1, domain.size());
  response.set_size(1, 1, domain.size());

  // One-hot encode each character.
  for (size_t t = 0; t < domain.size(); ++t)
  {
    const size_t dim = CharToDim(domain[t]);
    if (dim == size_t(-1))
    {
      cerr << "Domain '" << domain << "' has invalid character '" << domain[t]
          << "'!" << endl;
      exit(1);
    }

    data(dim, 0, t) = 1.0;

    if (t > 0)
      response(0, 0, t - 1) = dim;
  }

  // Set end-of-input response.
  response(0, 0, domain.size() - 1) = 38;
}

// Compute the likelihood that the string came from the model, given the
// predicted outputs of the model.
inline float ComputeLikelihood(const arma::fcube& predictions,
                               const arma::fcube& response)
{
  float likelihood = 0.0;
  for (size_t t = 0; t < response.n_slices; ++t)
    likelihood += predictions((size_t) response(0, 0, t), 0, t);

  return likelihood;
}

using namespace mlpack;
using namespace std;

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    cerr << "Usage: " << argv[0] << " benign_model.bin malicious_model.bin"
        << endl;
    cerr << " - Train a model with the lstm_dga_detection_train program."
        << endl;
  }

  // First load the model.
  RNN<NegativeLogLikelihoodType<arma::fmat>, RandomInitialization, arma::fmat>
      benignModel, maliciousModel;
  data::Load(argv[1], "lstm_model", benignModel, true /* fatal on failure */);
  data::Load(argv[2], "lstm_model", maliciousModel, true);

  // Now enter a loop where we read domains from stdin and then make
  // predictions.
  arma::fcube input, response, benignOutput, maliciousOutput;
  while (true)
  {
    string line;
    getline(cin, line);

    if (cin.eof())
    {
      // The user has terminated the program.
      return 0;
    }

    // Prepare the data for prediction and then make predictions with both
    // models.
    PrepareString(line, input, response);

    // Now compute prediction.
    benignModel.Predict(input, benignOutput);
    maliciousModel.Predict(input, maliciousOutput);

    const float benignLikelihood = ComputeLikelihood(benignOutput, response);
    const float maliciousLikelihood = ComputeLikelihood(maliciousOutput,
        response);
    const float score = benignLikelihood - maliciousLikelihood;

    if (benignLikelihood > maliciousLikelihood)
      cout << "benign (score " << score << ")" << std::endl;
    else
      cout << "malicious (score " << -score << ")" << std::endl;
  }
}
