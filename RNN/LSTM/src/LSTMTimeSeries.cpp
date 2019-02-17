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


template<typename InputDataType = arma::mat,typename DataType = arma::cube, typename LabelType = arma::cube>
void CreateData(InputDataType input, DataType& dataset, LabelType& label, int NUM_SAMPLES, int WINDOW_SIZE)
{
    for(int i = 0;i<NUM_SAMPLES;i++){
        dataset.subcube( span(0), span(i), span() ) =  input.rows(i, i+WINDOW_SIZE-1).t();
        for(int j = 0;j<WINDOW_SIZE;j++){
            label.at(0,i,j) =  input.at(i+j+1,0);
            // label.at(0,i,j) =  input.at(i+WINDOW_SIZE,0);
        }
    }
}



arma::cube trainX, trainY;
arma::cube testX, testY;

int main(int argc, char const *argv[]){

    arma::mat dataset;

    // In Armadillo rows represent features, columns represent data points.
    cout << "Reading data ..." << endl;
    data::Load("data/international-airline-passengers.csv", dataset, true);

    // The dataset CSV file has header, so it's necessary to
    // get rid of the this row, in Armadillo representation it's the first column
    // the first col in CSV is not required so removing the first row as well.
    dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

    //Scale data for increased numerical stability.
    dataset = StandardScaler(dataset);


    // Since the current implementation does not support arma::cube split 
    // So doing it manually
    int NUM_TSAMPLES = (int)(NUM_SAMPLES*(1-RATIO));
    int NUM_VSAMPLES = NUM_SAMPLES - NUM_TSAMPLES;

    arma::mat train, test;
    // Splitting the dataset on training and testing parts.
    data::Split(dataset,train, test, RATIO);

    //Converting to the conventional col as feature and row as data points format
    train = trans(train);
    test = trans(test);

    // Reshape the input data into appropriate form for RNNs.
    trainX.set_size(inputSize,NUM_TSAMPLES,rho);
    testX.set_size(inputSize,NUM_VSAMPLES,rho);
    trainY.set_size(outputSize,NUM_TSAMPLES,rho);
    testY.set_size(outputSize,NUM_VSAMPLES,rho);

    // Create testing and training sets for one-step-ahead regression.
    CreateData(train,trainX,trainY,NUM_TSAMPLES,WINDOW_SIZE);
    CreateData(test,testX,testY,NUM_VSAMPLES,WINDOW_SIZE);
    
    // RNN Model
    Model model = CreateModel(inputSize,outputSize,rho,maxRho);

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
    for (int i = 0; i < EPOCH; i++){
        // Train neural network. If this is the first iteration, weights are
        // random, using current values as starting point otherwise.
        model.Train(trainX, trainY, optimizer);

        // Don't reset optimizer's parameters between cycles.
        optimizer.ResetPolicy() = false;

        cube predOut;
        // Getting predictions on test data points.
        model.Predict(testX, predOut);
        // Calculating mse on test data points.
        double testMSE = calc_mse(predOut,testY);

        cout << i+1<< " - MeanSquaredError := "<< testMSE <<   endl;
    }

    cout << "Finished" << endl;
    return 0;
}