/**
 * An example of using Recurrent Neural Network (RNN) 
 * for sentiment analysis on encoded imdb dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @file LSTMSentimentAnalysis.cpp
 * @author Mehul Kumar Nirala
 */

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <ensmallen.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

/*
 * Function to calcute Accuracy for arma::cube
 */
double Accuracy(arma::cube& pred, arma::cube& Y, double tolerance = 0.5)
{
  double count = 0.0;
  arma::cube diff = pred-Y;
  for(size_t i = 0; i < diff.n_slices; i++)
  {
    mat temp = diff.slice(i);
    for (size_t j = 0; j < temp.n_cols; j++)
      if (abs(temp.at(0, j)) < tolerance)
        count ++;
  }
  // diff.print("dd");
  // cout<< count << " " <<diff.n_cols<<" "<<diff.n_slices<<std::endl;
  return (count / (diff.n_cols + 1e-50)) / (diff.n_slices + 1e-50);
}

int main()
{
  /* HYPERPARAMETERS */
  // Testing data is taken from the dataset in this ratio.
  const double RATIO = 0.2;

  // Number of cycles.
  const size_t EPOCH = 100;

  // Number of iteration per epoch.
  const size_t ITERATIONS_PER_EPOCH = 100;

  // Step size of an optimizer.
  const double STEP_SIZE = 1e-3;

  // Number of data points in each iteration of SGD.
  const size_t BATCH_SIZE = 16;

  // Save/Load model
  const bool saveModel = true;
  const bool loadModel = false;

  std::ifstream in("data_1000.txt");
  size_t vocabSize = 1000;
  std::vector<std::vector<int> > data;
  size_t seq_lengths = 0;
  if (in)
  {
    std::string line;
    while (std::getline(in, line))
    {
      data.push_back(std::vector<int>());
      // Break down the row into column values
      std::stringstream split(line);
      int value;
      while (split >> value)
          data.back().push_back(value);

      seq_lengths += data.back().size();
    }
  }

  // Average length of sentence.
  size_t mean_seq_length = seq_lengths/ data.size() - 1;

  // Creating dataset with mean sequence length x no. of data points.
  arma::mat dataset(mean_seq_length, data.size());
  arma::mat labels(1, data.size());
  for (size_t i = 0; i < data.size(); i++)
  { 
    labels.col(i) = data[i][0];
    for (size_t j = 0; j < std::min(mean_seq_length, data[i].size() - 1); j++)
      dataset.col(i).row(j) = data[i][j+1];

    // Pad zeros.
    for (size_t j = data[i].size(); j < mean_seq_length; j++)
      dataset.col(i).row(j) = 0;
  }

  arma::cube datasetX = arma::zeros<arma::cube>(vocabSize, dataset.n_cols, mean_seq_length);
  arma::cube datasetY(1, dataset.n_cols, mean_seq_length);

  for (size_t j = 0; j < dataset.n_cols; j++)
  {
    for (size_t i = 0; i < mean_seq_length; i++)
    {
      datasetX.at(dataset.at(i, j), j, i) = 1;
      datasetY.at(0, j, i) = labels.at(0, j);
    }
  }
  // Split the data into training and testing sets.
  std::cout << "Split data into training and testing sets." << std::endl;
  arma::cube trainX, trainY, testX, testY;
  size_t trainingSize = (1 - RATIO) * datasetX.n_cols;
  trainX = datasetX.subcube(span(), span(0, trainingSize-1), span());
  trainY = datasetY.subcube(span(), span(0, trainingSize-1), span());
  testX = datasetX.subcube(span(), span(trainingSize, datasetX.n_cols-1), span());
  testY = datasetY.subcube(span(), span(trainingSize, datasetY.n_cols-1), span());

  const size_t inputSize = vocabSize, outputSize = 1;

  // No of timesteps to look in RNN.
  const size_t rho = mean_seq_length;

  RNN<CrossEntropyError<>,HeInitialization> model(rho);

  // Model building/loading.
  std::cout << "Model building." << std::endl;
  if (loadModel)
  {
    std::cout << "Loading model ..." << std::endl;
    data::Load("saved_models/SentimentAnalysis.bin", "SentimentAnalysis", model);
  }
  else
  {
    model.Add<IdentityLayer<> >();
    model.Add<LSTM<> > (inputSize, 10, rho);
    model.Add<Dropout<> >(0.5);
    model.Add<LeakyReLU<> >();
    model.Add<Linear<> >(10, outputSize);
    model.Add<SigmoidLayer<> >();
  }

  // Setting parameters Stochastic Gradient Descent (SGD) optimizer.
  SGD<AdamUpdate> optimizer(
    STEP_SIZE, // Step size of the optimizer.
    BATCH_SIZE, // Batch size. Number of data points that are used in each iteration.
    ITERATIONS_PER_EPOCH, // Max number of iterations.
    1e-8,// Tolerance.
    true,// Shuffle.
    AdamUpdate(1e-8, 0.9, 0.999)// Adam update policy.
  );

  cout << "Training ..." << endl;
  // Cycles for monitoring the process of a solution.
  for (size_t i = 0; i < EPOCH; i++)
  {
    // Train neural network. If this is the first iteration, weights are
    // random, using current values as starting point otherwise.
    model.Train(trainX, trainY, optimizer);

    // Don't reset optimizer's parameters between cycles.
    optimizer.ResetPolicy() = false;

    arma::cube predOut;
    // Getting predictions on test data points.
    model.Predict(testX, predOut);

    // Calculating accuracy on test data points.
    double testAcc = Accuracy(predOut, testY);
    cout << i + 1 << " - Accuracy := "<< testAcc << endl;
  }

  cout << "Finished" << endl;
  cout << "Saving Model" << endl;
  if (saveModel)
  {
    data::Save("saved_models/SentimentAnalysis.bin", "SentimentAnalysis", model);
    std::cout << "Model saved in saved_models/." << std::endl;
  }
  return 0;
}
