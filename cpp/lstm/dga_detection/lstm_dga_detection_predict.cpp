/**
 * @file lstm_dga_detection_predict.cpp
 * @author Ryan Curtin
 *
 * Given a trained DGA detection model, make predictions.  Domains should be
 * input on stdin.
 */
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

using namespace mlpack;
using namespace std;

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " model.bin" << endl;
    cerr << " - Train a model with the lstm_dga_detection_train program."
        << endl;
  }

  // First load the model.
  RNN<MeanSquaredError, HeInitialization> network;
  data::Load(argv[1], "lstm_model", network, true /* fatal on failure */);

  // Now enter a loop where we read domains from stdin and then make
  // predictions.
  while (true)
  {
    string line;
    getline(cin, line);

    if (cin.eof())
    {
      // The user has terminated the program.
      return 0;
    }

    // We have to process the input into the correct one-hot format that the
    // model expects.  This will be a cube of shape:
    //
    //   rows = 41 (26 lowercase letters, 10 digits, period, hyphen, end,
    //              2 labels)
    //   columns = 1 (just one domain name)
    //   slices = up to maximum length of domain name supported by model
    const size_t len = std::max(network.BPTTSteps(), line.size() + 1);

    if (len != line.size() + 1)
    {
      cout << "Domain name is too long; will be truncated to '"
          << line.substr(0, len - 1) << "'." << endl;
    }

    arma::cube input(41, 1, len, arma::fill::zeros);
    for (size_t i = 0; i < len - 1; ++i)
    {
      char c = tolower(line[i]);
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
        cerr << "Cannot predict; domain has invalid character '" << c << "'!"
            << endl;
        continue;
      }

      input(dim, 0, i) = 1.0;
    }

    // Denote end of input.
    input(38, 0, len - 1) = 1.0;

    // Now compute prediction.
    arma::cube output;
    network.Predict(input, output);

    // If the activation for the malicious class is higher than the activation
    // for the benign class, then the domain is malicious.  We simply print the
    // prediction to stdout.
    if (output(40, 0, len - 1) >= output(39, 0, len - 1))
      cout << "malicious" << endl;
    else
      cout << "benign" << endl;
  }
}
