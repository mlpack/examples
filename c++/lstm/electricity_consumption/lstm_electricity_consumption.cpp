/**
 * An example of using Recurrent Neural Network (RNN) 
 * to make forecasts on a time series of number of kilowatt-hours used in a
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
 * @author Zoltan Somogyi
 */

/*
NOTE: the data need to be sorted by date in ascending order! The RNN learns from
oldest to newest!

DateTime,Consumption kWh,Off-peak,Mid-peak,On-peak
11/25/2011 01:00:00,0.39,1,0,0
11/25/2011 02:00:00,0.33,1,0,0
11/25/2011 03:00:00,0.27,1,0,0
11/25/2011 04:00:00,0.29,1,0,0
11/25/2011 05:00:00,0.29,1,0,0
11/25/2011 06:00:00,0.29,1,0,0
11/25/2011 07:00:00,0.28,1,0,0
11/25/2011 08:00:00,0.31,0,0,1
11/25/2011 09:00:00,0.33,0,0,1
11/25/2011 10:00:00,0.48,0,0,1
...
*/

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

using namespace std;
using namespace mlpack;
using namespace ens;

/*
 * Function to calculate MSE for arma::cube.
 */
double ComputeMSE(arma::cube& pred, arma::cube& Y)
{
  return SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

/**
 * The time series data for training the model contains the electricity
 * consumption in kWh for 'rho' hours in the past.  The target variable we want
 * to predict is the electricity consumption in kWh for the next hour.
 *
 * NOTE: We do not use the last input data point in the training because there
 * is no target (next hour) for that point.
 */
template<typename InputDataType = arma::mat,
         typename DataType = arma::cube,
         typename LabelType = arma::cube>
void CreateTimeSeriesData(InputDataType dataset,
                          DataType& X,
                          LabelType& y,
                          const size_t rho)
{
  for (size_t i = 0; i < dataset.n_cols - rho; i++)
  {
    X.subcube(arma::span(), arma::span(i), arma::span()) =
        dataset.submat(arma::span(), arma::span(i, i + rho - 1));
    y.subcube(arma::span(), arma::span(i), arma::span()) =
        dataset.submat(arma::span(), arma::span(i + 1, i + rho));
  }
}

/**
 * This function saves the input data for prediction and the prediction results
 * in CSV format. The prediction results are the electricity consumption in kWh
 * for the next hour and come from the last slice of the prediction.  The last
 * column is the prediction; the preceding column is the data used to generate
 * those predictions.
 */
void SaveResults(const string& filename,
                 const arma::cube& predictions,
                 data::MinMaxScaler& scale,
                 const arma::cube& testX)
{
  arma::mat flatDataAndPreds = testX.slice(testX.n_slices - 1);
  scale.InverseTransform(flatDataAndPreds, flatDataAndPreds);

  // The prediction result is the energy consumption for the next hour and 
  // comes from the last slice of the prediction.
  arma::mat temp = predictions.slice(predictions.n_slices - 1);
  scale.InverseTransform(temp, temp);

  // We shift the predictions such that the true values are synchronized with
  // the predictions, and we also add one more record to the input. Please note
  // that this means the last input record zero and the first prediction record
  // is also zero.
  temp.insert_cols(0, 1, true);
  flatDataAndPreds.insert_cols(flatDataAndPreds.n_cols, 1, true);

  // We add the predictions as the last column.
  flatDataAndPreds.insert_rows(flatDataAndPreds.n_rows,
      temp.rows(temp.n_rows - 1, temp.n_rows - 1));

  // Save the data to file. The last columns are the predictions; the preceding
  // column is the data used to generate those predictions.
  data::Save(filename, flatDataAndPreds);

  // Print the output to screen.
  cout << "The predicted energy consumption for the last hour is : " << endl;
  cout << " " << flatDataAndPreds(flatDataAndPreds.n_rows - 1,
                                  flatDataAndPreds.n_cols - 1) << endl;
}

int main()
{
  // Change the names of these files as necessary. They should be correct
  // already, if your program's working directory contains the data and/or
  // model.
  const string dataFile = "../../../data/electricity-usage.csv";
  // example: const string dataFile =
  //                  "C:/mlpack-model-app/electricity-usage.csv";
  // example: const string dataFile =
  //                  "/home/user/mlpack-model-app/electricity-usage.csv";

  const string modelFile = "lstm_univar.bin";
  // example: const string modelFile =
  //                  "C:/mlpack-model-app/lstm_univar.bin";
  // example: const string modelFile =
  //                  "/home/user/mlpack-model-app/lstm_univar.bin";

  const string predFile = "lstm_univar_predictions.csv";

  // If true, the model will be trained; if false, the saved model will be read
  // and used for prediction.
  // NOTE: training the model may take a long time; therefore once it is trained
  // you can set this to false and use the model for prediction.
  // NOTE: there is no error checking in this example to see if the trained
  // model exists.
  const bool bTrain = true;
  // You can load and further train a model.
  const bool bLoadAndTrain = false;

  // Training data is randomly taken from the dataset in this ratio.
  const double RATIO = 0.1;

  // Step size of an optimizer.
  const double STEP_SIZE = 5e-5;

  // Number of data points in each iteration of SGD.
  const size_t BATCH_SIZE = 10;

  // Data has only one dimension.
  const size_t inputSize = 1;

  // We are predicting the next value, hence, the output is one dimensional.
  const size_t outputSize = 1;

  // Number of timesteps to look backwards in RNN.
  const size_t rho = 10;

  // Number of cells in the LSTM (hidden layers in standard terms)
  // NOTE: you may play with this variable in order to further optimize the
  // model.  (as more cells are added, accuracy is likely to go up, but training
  // time may take longer)
  const int H1 = 10;

  arma::mat dataset;

  // In Armadillo rows represent features, columns represent data points.
  cout << "Reading data ..." << endl;
  data::Load(dataFile, dataset, true);

  // The CSV file has a header, so it is necessary to remove it. In Armadillo's
  // representation it is the first column.
  // The first column in the CSV is the date which is not required, therefore
  // removing it also (first row in in arma::mat).
  dataset = dataset.submat(1, 1, 1, dataset.n_cols - 1);

  // Split the dataset into training and validation sets.
  arma::mat trainData;
  arma::mat testData;
  data::Split(dataset, trainData, testData, RATIO, false);

  // Number of iterations per cycle.
  const int EPOCHS = 150;

  // Scale all data into the range (0, 1) for increased numerical stability.
  data::MinMaxScaler scale;
  // Fit scaler only on training data.
  scale.Fit(trainData);
  scale.Transform(trainData, trainData);
  scale.Transform(testData, testData);

  // We need to represent the input data for RNN in an arma::cube (3D matrix).
  // The 3rd dimension is the rho number of past data records the RNN uses for
  // learning.
  arma::cube trainX, trainY, testX, testY;
  trainX.set_size(inputSize, trainData.n_cols - rho, rho);
  trainY.set_size(outputSize, trainData.n_cols - rho, rho);
  testX.set_size(inputSize, testData.n_cols - rho, rho);
  testY.set_size(outputSize, testData.n_cols - rho, rho);

  // Create training sets for one-step-ahead regression.
  CreateTimeSeriesData(trainData, trainX, trainY, rho);
  // Create test sets for one-step-ahead regression.
  CreateTimeSeriesData(testData, testX, testY, rho);

  // Only train the model if required.
  if (bTrain || bLoadAndTrain)
  {
    // RNN regression model.
    RNN<MeanSquaredError, HeInitialization> model(rho);

    if (bLoadAndTrain)
    {
      // The model will be trained further.
      cout << "Loading and further training model..." << endl;
      data::Load(modelFile, "LSTMUnivar", model);
    }
    else
    {
      // Model building.
      model.Add<LSTM>(H1);
      model.Add<LeakyReLU>();
      model.Add<LSTM>(H1);
      model.Add<LeakyReLU>();
      model.Add<Linear>(outputSize);
    }

    // Set parameters for the Adam optimizer.
    Adam optimizer(
        STEP_SIZE,  // Step size of the optimizer.
        BATCH_SIZE, // Batch size. Number of data points that are used in each
                    // iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999,      // Exponential decay rate for the weighted infinity norm
                    // estimates.
        1e-8, // Value used to initialise the mean squared gradient parameter.
        trainData.n_cols * EPOCHS, // Max number of iterations.
        1e-8,                      // Tolerance.
        true);

    // Instead of terminating based on the tolerance of the objective function,
    // we'll depend on the maximum number of iterations, and terminate early
    // using the EarlyStopAtMinLoss callback.
    optimizer.Tolerance() = -1;

    cout << "Training ..." << endl;

    model.Train(trainX,
                trainY,
                optimizer,
                // PrintLoss Callback prints loss for each epoch.
                PrintLoss(),
                // Progressbar Callback prints progress bar for each epoch.
                ProgressBar(),
                // Stops the optimization process if the loss stops decreasing
                // or no improvement has been made. This will terminate the
                // optimization once we obtain a minima on training set.
                EarlyStopAtMinLoss());

    cout << "Finished training." << endl;
    cout << "Saving Model" << endl;
    data::Save(modelFile, "LSTMUnivar", model);
    cout << "Model saved in " << modelFile << endl;
  }

  // NOTE: the code below is added in order to show how in a real application
  // the model would be saved, loaded and then used for prediction. Please note
  // that we do not have the last data point in testX because we did not use it
  // for the training, therefore the prediction result will be for the hour
  // before.  In your own application you may of course load any dataset.

  // Load RNN model and use it for prediction.
  RNN<MeanSquaredError, HeInitialization> modelP(rho);
  cout << "Loading model ..." << endl;
  data::Load(modelFile, "LSTMUnivar", modelP);
  arma::cube predOutP;

  // Get predictions on the test data points.
  modelP.Predict(testX, predOutP);
  // Calculate the MSE on the predictions.
  double testMSEP = ComputeMSE(predOutP, testY);
  cout << "Mean Squared Error on Prediction data points: " << testMSEP << endl;

  // Save the output predictions and show the results.
  SaveResults(predFile, predOutP, scale, testX);

  // Use this on Windows in order to keep the console window open.
  // cout << "Ready!" << endl;
  // getchar();
}
