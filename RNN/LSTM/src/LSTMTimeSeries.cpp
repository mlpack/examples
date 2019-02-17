/**
 * An example of using Recurrent Neural Network (RNN) 
 * to make forcasts on a time series of international airline passengers,
 * which we aim to solve using the a simple LSTM neural network.
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @author Mehul Kumar Nirala
 */


#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/core/optimizers/adam/adam_update.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using Model = RNN<MeanSquaredError<>,HeInitialization>;

// HYPERPARAMETERS

// Training data is randomly taken from the dataset in this ratio.
const double RATIO = 0.1;

// Number of cycles.
const int EPOCH = 25;

// Number of iteration per epoch.
const int ITERATIONS_PER_EPOCH = 10;

// Step size of an optimizer.
const double STEP_SIZE = 5e-4;

// Number of data points in each iteration of SGD
const size_t BATCH_SIZE = 10;

// Data has only one dimensional
const size_t inputSize = 1;
// Predicting the next value hence, one dimensional
const size_t outputSize = 1;

// Using previous WINDOW_SIZE values to predict the next value in time series.
const size_t WINDOW_SIZE = 4;

// No of timesteps to look in RNN.
const size_t rho = WINDOW_SIZE;

// Max Rho for LSTM 
const size_t maxRho = rho;

// taking the first 100 Samples
const int NUM_SAMPLES = 100;
