/**
 * @file lstm_baseline_main.cpp
 * @author Konstantin Sidorov
 *
 * Executable for LSTM baseline solution for ann::augmented tasks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/math/random.hpp>

#include <mlpack/core/data/binarize.hpp>

#include <mlpack/methods/ann/augmented/tasks/copy.hpp>
#include <mlpack/methods/ann/augmented/tasks/sort.hpp>
#include <mlpack/methods/ann/augmented/tasks/add.hpp>
#include <mlpack/methods/ann/augmented/tasks/score.hpp>

#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/rnn.hpp>

using namespace mlpack;

using namespace mlpack::ann::augmented::tasks;
using namespace mlpack::ann::augmented::scorers;

using namespace mlpack::ann;
using namespace mlpack::optimization;

using mlpack::data::Binarize;

using std::string;

PROGRAM_INFO("LSTM baseline for ann::augmented tasks",
    "This program runs and evaluates a simple LSTM recurrent neural network "
    "over several standard tasks from ann::augmented::tasks."
    "\n\n"
    "For example, the following will execute the LSTM network on sequence copy task instance "
    "with 6 as maximum sequence length, using 1000 samples for learning and running 1000 "
    "epochs over them:"
    "\n\n"
    "$ augmented_baseline --task=copy --epochs=1000 --samples=1000 --length=6 --repeats=1"
    "\n\n"
    "Supported tasks are listed below:"
    "\n\n"
    " * 'copy': sequence copy task\n"
    "\n"
    " * 'add': binary addition task\n"
    "\n"
    " * 'sort': sequence sort task (sequence elements are binary numbers)\n"
    "\n"
    "The parameters for each of the tasks should be specified with the "
    "options --length, --bit_length, or a "
    "combination of those options (as in case with --task=sort)."
    "\n\n"
    "Learning-specific parameters are also tunable (e.g., --epochs and --samples)");

PARAM_STRING_IN_REQ("task", "Task to execute LSTM network on.", "t");

PARAM_INT_IN("length", "Maximum sequence length (doesn't impact binary addition task).", "l", 2);
PARAM_INT_IN("repeats", "Number of repeats required to solve the task (specific for Copy task).", "r", 1);
PARAM_INT_IN("bit_length", "Maximum length of sequence elements in binary representation.", "b", 2);
PARAM_INT_IN("epochs", "Learning epochs.", "i", 25);
PARAM_INT_IN("samples", "Sample size used for fitting and evaluating the model.", "s", 64);

// Generate the LSTM model for benchmarking.
// NOTE: it's the same model for all tasks.
RNN<MeanSquaredError<> > GenerateModel(size_t inputSize,
                                       size_t outputSize,
                                       size_t maxRho) {
  RNN<MeanSquaredError<> > model(2);

  model.Add<IdentityLayer<> >();
  model.Add<Linear<> >(inputSize, 40);
  model.Add<LeakyReLU<> >();
  model.Add<LSTM<> >(40, 30, maxRho);
  model.Add<LeakyReLU<> >();
  model.Add<LSTM<> >(30, 30, maxRho);
  model.Add<LeakyReLU<> >();
  model.Add<LSTM<> >(30, 20, maxRho);
  model.Add<LeakyReLU<> >();
  model.Add<Linear<> >(20, outputSize);
  model.Add<SigmoidLayer<> >();

  return model;
}

//! Run CopyTask on the specified sequence length and repeat count.
void RunCopyTask(size_t maxLen,
                 size_t nRepeats,
                 size_t epochs,
                 size_t samples)
{
  Log::Info << "Running copy task benchmark for nRepeats = "
             << nRepeats << ", maxLen = " << maxLen << "\n";

  const size_t outputSize = 1;
  const size_t inputSize = 2;
  size_t maxRho = inputSize * (nRepeats + 1) * maxLen + 1;

  // Creating a baseline model.
  RNN<MeanSquaredError<> > model = GenerateModel(inputSize, outputSize, maxRho);

  Adam<decltype(model)> opt(model);

  // Initializing task instance.
  CopyTask task(maxLen, nRepeats, true);

  arma::mat trainPredictor, trainResponse;
  task.Generate(trainPredictor, trainResponse, samples);

  arma::field<arma::mat> testPredictor, testResponse;
  task.Generate(testPredictor, testResponse, samples, true);
  assert(testPredictor.n_elem == testResponse.n_elem &&
         testResponse.n_elem == samples);

  // Training loop
  Log::Info << "Running training loop for " << epochs << " epochs.\n";
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    Log::Debug << "Starting training epoch #"
               << epoch+1 << "\n";
    model.Rho() = trainPredictor.n_rows / inputSize;
    model.Train(trainPredictor, trainResponse, opt);
    Log::Debug << "Finished running training epoch #"
               << epoch+1 << "\n";
    std::cerr  << "Finished running training epoch #"
               << epoch+1 << "\n";
  }
  Log::Info << "Finished training loop.\n";

  // Evaluation loop
  Log::Info << "Running evaluation loop.\n";
  arma::field<arma::mat> modelOutput(samples);

  for (size_t example = 0; example < samples; ++example) {
    arma::mat predictor = testPredictor.at(example);
    arma::mat response = testResponse.at(example);

    Log::Debug << "Evaluating model on:\n";
    Log::Debug << "Input sequence:\n" << predictor.t();
    Log::Debug << "Ground truth sequence:\n" << response.t();

    model.Rho() = predictor.n_elem / inputSize;
    arma::mat softOutput;
    model.Predict(
      predictor,
      softOutput);

    Log::Debug << "Model predictions:\n";
    Log::Debug << softOutput.t();

    modelOutput.at(example) = softOutput.rows(
      predictor.n_elem / (inputSize * (nRepeats + 1)),
      softOutput.n_rows-1);
    testResponse.at(example) = testResponse.at(example).rows(
      predictor.n_elem / (inputSize * (nRepeats + 1)),
      softOutput.n_rows-1);
    Binarize<double>(modelOutput.at(example), modelOutput.at(example), 0.5);

    Log::Debug << "Model predictions after binarization:\n";
    Log::Debug << testResponse.at(example).t();
  }
  Log::Info << "Final score: "
            << SequencePrecision<arma::mat>(testResponse, modelOutput)
            << "\n";
  std::cout << "Final score: "
            << SequencePrecision<arma::mat>(testResponse, modelOutput)
            << "\n";
}


//! Run AddTask on the specified bit length.
void RunAddTask(size_t bitLen,
                size_t epochs,
                size_t samples)
{
  Log::Info << "Running add task benchmark for bitLen = " << bitLen << "\n";

  const size_t outputSize = 3;
  const size_t inputSize = 3;
  size_t maxRho = inputSize * (bitLen + 1) + 1;

  // Creating a baseline model.
  RNN<MeanSquaredError<> > model = GenerateModel(inputSize, outputSize, maxRho);

  Adam<decltype(model)> opt(model);

  // Initializing task instance.
  AddTask task(bitLen);

  arma::mat trainPredictor, trainResponse;
  task.Generate(trainPredictor, trainResponse, samples);

  arma::field<arma::mat> testPredictor, testResponse;
  task.Generate(testPredictor, testResponse, samples, true);
  assert(testPredictor.n_elem == testResponse.n_elem &&
         testResponse.n_elem == samples);

  // Training loop
  Log::Info << "Running training loop for " << epochs << " epochs.\n";
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    Log::Debug << "Starting training epoch #"
               << epoch+1 << "\n";
    model.Rho() = trainPredictor.n_rows / inputSize;
    model.Train(trainPredictor, trainResponse, opt);
    Log::Debug << "Finished running training epoch #"
               << epoch+1 << "\n";
    std::cerr  << "Finished running training epoch #"
               << epoch+1 << "\n";
  }
  Log::Info << "Finished training loop.\n";

  // Evaluation loop
  Log::Info << "Running evaluation loop.\n";
  arma::field<arma::mat> modelOutput(samples);
  for (size_t example = 0; example < samples; ++example) {
    arma::mat predictor = testPredictor.at(example);
    arma::mat response = testResponse.at(example);

    Log::Debug << "Evaluating model on:\n";
    Log::Debug << "Input sequence:\n" << predictor.t();
    Log::Debug << "Ground truth sequence:\n" << response.t();

    model.Rho() = predictor.n_elem / inputSize;
    arma::mat softOutput;
    model.Predict(
      predictor,
      softOutput);

    Log::Debug << "Model predictions:\n";
    Log::Debug << softOutput.t();

    modelOutput.at(example) = softOutput;
    Binarize<double>(modelOutput.at(example), modelOutput.at(example), 0.5);

    Log::Debug << "Model predictions after binarization:\n";
    Log::Debug << testResponse.at(example).t();
  }
  Log::Info << "Final score: "
            << SequencePrecision<arma::mat>(testResponse, modelOutput)
            << "\n";
  std::cout << "Final score: "
            << SequencePrecision<arma::mat>(testResponse, modelOutput)
            << "\n";
}

//! Run SortTask on the specified sequence length and bit length.
void RunSortTask(size_t maxLen,
                 size_t bitLen,
                 size_t epochs,
                 size_t samples) {
  Log::Info << "Running sort task benchmark for maxLen = "
            << maxLen << "\n, bitLen = " << bitLen << "\n";

  size_t outputSize = bitLen;
  size_t inputSize = bitLen;
  size_t maxRho = inputSize * (bitLen + 1) + 1;

  // Creating a baseline model.
  RNN<MeanSquaredError<> > model = GenerateModel(inputSize, outputSize, maxRho);

  Adam<decltype(model)> opt(model);

  // Initializing task instance.
  SortTask task(maxLen, bitLen);

  arma::mat trainPredictor, trainResponse;
  task.Generate(trainPredictor, trainResponse, samples);

  arma::field<arma::mat> testPredictor, testResponse;
  task.Generate(testPredictor, testResponse, samples, true);
  assert(testPredictor.n_elem == testResponse.n_elem &&
         testResponse.n_elem == samples);

  // Training loop
  Log::Info << "Running training loop for " << epochs << " epochs.\n";
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    Log::Debug << "Starting training epoch #"
               << epoch+1 << "\n";
    model.Rho() = trainPredictor.n_rows / inputSize;
    model.Train(trainPredictor, trainResponse, opt);
    Log::Debug << "Finished running training epoch #"
               << epoch+1 << "\n";
    std::cerr  << "Finished running training epoch #"
               << epoch+1 << "\n";
  }
  Log::Info << "Finished training loop.\n";

  // Evaluation loop
  Log::Info << "Running evaluation loop.\n";
  arma::field<arma::mat> modelOutput(samples);

  for (size_t example = 0; example < samples; ++example) {
    arma::mat predictor = testPredictor.at(example);
    arma::mat response = testResponse.at(example);

    Log::Debug << "Evaluating model on:\n";
    Log::Debug << "Input sequence:\n" << predictor.t();
    Log::Debug << "Ground truth sequence:\n" << response.t();

    model.Rho() = predictor.n_elem / inputSize;
    arma::mat softOutput;
    model.Predict(
      predictor,
      softOutput);

    Log::Debug << "Model predictions:\n";
    Log::Debug << softOutput.t();

    modelOutput.at(example) = softOutput;
    Binarize<double>(modelOutput.at(example), modelOutput.at(example), 0.5);

    Log::Debug << "Model predictions after binarization:\n";
    Log::Debug << testResponse.at(example).t();
  }
  Log::Info << "Final score: "
            << SequencePrecision<arma::mat>(testResponse, modelOutput)
            << "\n";
  std::cout << "Final score: "
            << SequencePrecision<arma::mat>(testResponse, modelOutput)
            << "\n";
}

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  string task = CLI::GetParam<string>("task");
  if (task == "copy") {
    int repeats = CLI::GetParam<int>("repeats");
    if (repeats <= 0) {
      Log::Fatal << "Invalid value for 'repeats': "
                 << "expecting a positive number, received "
                 << repeats << "\n";
    }
    int maxLen = CLI::GetParam<int>("length");
    if (maxLen <= 0) {
      Log::Fatal << "Invalid value for 'length': "
                 << "expecting a positive number, received "
                 << maxLen << "\n";
    }
    int epochs = CLI::GetParam<int>("epochs");
    if (repeats <= 0) {
      Log::Fatal << "Invalid value for 'epochs': "
                 << "expecting a positive number, received "
                 << repeats << "\n";
    }
    int samples = CLI::GetParam<int>("samples");
    if (samples <= 0) {
      Log::Fatal << "Invalid value for 'samples': "
                 << "expecting a positive number, received "
                 << samples << "\n";
    }
    RunCopyTask(maxLen, repeats, epochs, samples);
  }
  else if (task == "add") {
    int bitLen = CLI::GetParam<int>("bit_length");
    if (bitLen <= 0) {
      Log::Fatal << "Invalid value for 'bit_length': "
                 << "expecting a positive number, received "
                 << bitLen << "\n";
    }
    int epochs = CLI::GetParam<int>("epochs");
    if (epochs <= 0) {
      Log::Fatal << "Invalid value for 'epochs': "
                 << "expecting a positive number, received "
                 << epochs << "\n";
    }
    int samples = CLI::GetParam<int>("samples");
    if (samples <= 0) {
      Log::Fatal << "Invalid value for 'samples': "
                 << "expecting a positive number, received "
                 << samples << "\n";
    }
    RunAddTask(bitLen, epochs, samples);
  }
  else if (task == "sort") {
    int bitLen = CLI::GetParam<int>("bit_length");
    if (bitLen <= 0) {
      Log::Fatal << "Invalid value for 'bit_length': "
                 << "expecting a positive number, received "
                 << bitLen << "\n";
    }
    int maxLen = CLI::GetParam<int>("length");
    if (maxLen <= 0) {
      Log::Fatal << "Invalid value for 'length': "
                 << "expecting a positive number, received "
                 << bitLen << "\n";
    }
    int epochs = CLI::GetParam<int>("epochs");
    if (epochs <= 0) {
      Log::Fatal << "Invalid value for 'epochs': "
                 << "expecting a positive number, received "
                 << epochs << "\n";
    }
    int samples = CLI::GetParam<int>("samples");
    if (samples <= 0) {
      Log::Fatal << "Invalid value for 'samples': "
                 << "expecting a positive number, received "
                 << samples << "\n";
    }
    RunSortTask(maxLen, bitLen, epochs, samples);
  }
  else {
    Log::Fatal << "Can't recognize task type, aborting execution.\n";
    Log::Fatal << "Supported tasks: add, copy, sort.\n";
  }
}
