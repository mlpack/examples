/**
 * An example of using Recurrent Neural Network (RNN) 
 * to make forcasts on a time series of number of kilowatt-hours used in a
 * residential home over a 3.5 month period, 25 November 2011 to 17 March 2012,
 * which we aim to solve using a simple LSTM neural network. Electricity usage
 * as recorded by the local utility company on an hour-by-hour basis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @file LSTMTimeSeriesUnivariate.cpp
 * @author Mehul Kumar Nirala
 */

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <ensmallen.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

// HYPERPARAMETERS
// Training data is randomly taken from the dataset in this ratio.
const double RATIO = 0.1;

// Number of cycles.
const size_t EPOCH = 100;

// Number of iteration per epoch.
const size_t ITERATIONS_PER_EPOCH = 100000;

// Step size of an optimizer.
const double STEP_SIZE = 5e-5;

// Number of data points in each iteration of SGD
const size_t BATCH_SIZE = 16;

// Data has only one dimensional
const size_t inputSize = 1;

// Predicting the next value hence, one dimensional
const size_t outputSize = 1;

// No of timesteps to look in RNN.
const size_t rho = 10;

// Max Rho for LSTM 
const size_t maxRho = rho;

// Save/Load model
const bool saveModel = true;
const bool loadModel = false;

/*
 * Function to calcute MSE for arma::cube
 */
double MSE(arma::cube& pred, arma::cube& Y)
{
  double err_sum = 0.0;
  arma::cube diff = pred-Y;
  for(size_t i = 0; i < diff.n_slices; i++)
  {
    mat temp = diff.slice(i);
    err_sum += accu(temp%temp);
  }
  return (err_sum) / (diff.n_elem + 1e-50);
}

/*
* Here, we have a univariate data set which records the number of airline passengers for each month.
* we modify the date to the time series in order to get some ideas about underlying trends,
*/
template<typename InputDataType = arma::mat,
   typename DataType = arma::cube,
   typename LabelType = arma::cube>
void CreateTimeSeriesData(InputDataType dataset, DataType& X, LabelType& y, size_t rho)
{
  for(size_t i = 0; i<dataset.n_cols - rho - 2; i++)
  {
    X.subcube(span(), span(i), span()) = dataset.submat(span(), span(i, i+rho-1));
    y.subcube(span(), span(i), span()) = dataset.submat(span(), span(i+1, i+rho));
  }
}

template<typename DataType = arma::mat>
DataType MinMaxScaler(DataType& dataset)
{
  arma::vec maxValues = arma::max(dataset, 1 /* for each dimension */);
  arma::vec minValues = arma::min(dataset, 1 /* for each dimension */);

  arma::vec rangeValues = maxValues - minValues;

  // Add a very small value if there are any zeros.
  rangeValues += 1e-25;

  dataset -= arma::repmat(minValues , 1, dataset.n_cols);
  dataset /= arma::repmat(rangeValues , 1, dataset.n_cols);
  return dataset;
}

int main()
{
  arma::mat dataset;

  // In Armadillo rows represent features, columns represent data points.
  cout << "Reading data ..." << endl;
  data::Load("LSTM/data/electricity-usage.csv", dataset, true);

  // The dataset CSV file has header, so it's necessary to
  // get rid of the this row, in Armadillo representation it's the first column
  // the first col in CSV is not required so removing the first row as well.
  dataset = dataset.submat(1, 1, 1, dataset.n_cols - 1);

  //Scale data for increased numerical stability.
  dataset = MinMaxScaler(dataset);

  arma::cube X, y;
  X.set_size(inputSize, dataset.n_cols - rho, rho);
  y.set_size(outputSize, dataset.n_cols - rho, rho);

  // Create testing and training sets for one-step-ahead regression.
  CreateTimeSeriesData(dataset, X, y, rho);

  // Split the data into training and testing sets.
  arma::cube trainX, trainY, testX, testY;
  size_t trainingSize = (1 - RATIO) * X.n_cols;
  trainX = X.subcube(span(), span(0, trainingSize-1), span());
  trainY = y.subcube(span(), span(0, trainingSize-1), span());
  testX = X.subcube(span(), span(trainingSize, X.n_cols-1), span());
  testY = y.subcube(span(), span(trainingSize, X.n_cols-1), span());
  
  // RNN Model
  RNN<MeanSquaredError<>,HeInitialization> model(rho);

  //MODEL BUILDING/LOADING
  if (loadModel)
  {
    std::cout << "Loading model ..." << std::endl;
    data::Load("saved_models/LSTMUni.bin", "LSTMUni", model);
  }
  else
  {
    model.Add<IdentityLayer<> >();
    model.Add<LSTM<> > (inputSize, 4, maxRho);
    model.Add<LeakyReLU<> >();
    model.Add<LSTM<> > (4, 4, maxRho);
    model.Add<LeakyReLU<> >();
    model.Add<Linear<> >(4, outputSize);
  }

  // Setting parameters Stochastic Gradient Descent (SGD) optimizer.
  SGD<AdamUpdate> optimizer(
    STEP_SIZE, // Step size of the optimizer.
    BATCH_SIZE, // Batch size. Number of data points that are used in each iteration.
    ITERATIONS_PER_EPOCH, // Max number of iterations
    1e-8,// Tolerance
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

    cube predOut;
    // Getting predictions on test data points.
    model.Predict(testX, predOut);
    // Calculating mse on test data points.
    double testMSE = MSE(predOut, testY);

    cout << i+1<< " - MeanSquaredError := "<< testMSE <<   endl;
  }

  cout << "Finished" << endl;
  cout << "Saving Model" << endl;
  if (saveModel)
  {
    data::Save("saved_models/LSTMUni.bin", "LSTMUni", model);
    std::cout << "Model saved in saved_models/." << std::endl;
  }
  return 0;
}