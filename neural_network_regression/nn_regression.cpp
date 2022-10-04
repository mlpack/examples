/**
 * @file nn_regression.cpp
 * @author Zoltan Somogyi
 *
 * MLPACK TUTORIAL: neural network regression
 *
 * Real world example which shows how to create a neural network mlpack/C++
 * model for regression, how to save and load the model and then use it for
 * prediction (inference).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

/**
 * ABOUT THE DATASET bodyfat.tsv (freely available dataset).
 *
 * The bodyfat dataset contains estimates of the percentage of body fat
 * determined by underwater weighing and various body circumference measurements
 * for 252 men. Accurate measurement of body fat is very expensive, but by using
 * machine learning it is possible to calculate a prediction with good accuracy
 * by just using some low cost measurements of the body. The columns in the
 * dataset are the following:
 *
 * Percent body fat (%) => this is the decision column (what we want to get from
 * the model).
 * Age (years)
 * Weight (lbs)
 * Height (inches)
 * Neck circumference (cm)
 * Chest circumference (cm)
 * Abdomen 2 circumference (cm)
 * Hip circumference (cm)
 * Thigh circumference (cm)
 * Knee circumference (cm)
 * Ankle circumference (cm)
 * Biceps (extended) circumference (cm)
 * Forearm circumference (cm)
 * Wrist circumference (cm)
 * Density determined from underwater weighing
 */
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

using namespace mlpack;
using namespace ens;
using namespace std;
using namespace arma;

/*
 * Function to calculate MSE for arma::cube.
 */
double ComputeMSE(mat& pred, mat& Y)
{
  return SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

int main()
{
  //! Path to the dataset used for training and testing.
  const string datasetPath = "./../data/BodyFat.tsv";
  // File for saving the model.
  const string modelFile = "nn_regressor.bin";

  // Testing data is taken from the dataset in this ratio.
  constexpr double RATIO = 0.1; // 10%

  //! - H1: The number of neurons in the 1st layer.
  constexpr int H1 = 64;
  //! - H2: The number of neurons in the 2nd layer.
  constexpr int H2 = 128;
  //! - H3: The number of neurons in the 3rd layer.
  constexpr int H3 = 64;

  // Number of epochs for training.
  const int EPOCHS = 100;
  //! - STEP_SIZE: Step size of the optimizer.
  constexpr double STEP_SIZE = 5e-2;
  //! - BATCH_SIZE: Number of data points in each iteration of SGD.
  constexpr int BATCH_SIZE = 32;
  //! - STOP_TOLERANCE: Stop tolerance;
  // A very small number implies that we do all iterations.
  constexpr double STOP_TOLERANCE = 1e-8;

  // If true, the model will be trained; if false, the saved model will be
  // read and used for prediction
  // NOTE: Training the model may take a long time, therefore once it is
  // trained you can set this to false and use the model for prediction.
  // NOTE: There is no error checking in this example to see if the trained
  // model exists!
  const bool bTrain = true;
  // You can load and further train a model by setting this to true.
  const bool bLoadAndTrain = false;

  arma::mat dataset;

  // In Armadillo rows represent features, columns represent data points.
  cout << "Reading data." << endl;
  bool loadedDataset = data::Load(datasetPath, dataset, true);
  // If dataset is not loaded correctly, exit.
  if (!loadedDataset)
    return -1;

  // Split the dataset into training and validation sets.
  arma::mat trainData, validData;
  data::Split(dataset, trainData, validData, RATIO);

  // The train and valid datasets contain both - the features as well as the
  // prediction. Split these into separate matrices.
  arma::mat trainX =
      trainData.submat(1, 0, trainData.n_rows - 1, trainData.n_cols - 1);
  arma::mat validX =
      validData.submat(1, 0, validData.n_rows - 1, validData.n_cols - 1);

  // Create prediction data for training and validatiion datasets.
  arma::mat trainY = trainData.row(0);
  arma::mat validY = validData.row(0);

  // Scale all data into the range (0, 1) for increased numerical stability.
  data::MinMaxScaler scaleX;
  // Scaler for predictions.
  data::MinMaxScaler scaleY;
  // Fit scaler only on training data.
  scaleX.Fit(trainX);
  scaleX.Transform(trainX, trainX);
  scaleX.Transform(validX, validX);

  // Scale training predictions.
  scaleY.Fit(trainY);
  scaleY.Transform(trainY, trainY);
  scaleY.Transform(validY, validY);

  // Only train the model if required.
  if (bTrain || bLoadAndTrain)
  {
    // Specifying the NN model.
    FFN<MeanSquaredError, HeInitialization> model;
    if (bLoadAndTrain)
    {
      // The model will be trained further.
      cout << "Loading and further training the model." << endl;
      data::Load(modelFile, "NNRegressor", model);
    }
    else
    {
      // This intermediate layer is needed for connection between input
      // data and the next LeakyReLU layer.
      // Parameters specify the number of neurons in the next layer.
      model.Add<Linear>(H1);
      // Activation layer:
      model.Add<LeakyReLU>();
      // Connection layer between two activation layers.
      model.Add<Linear>(H2);
      // Activation layer.
      model.Add<LeakyReLU>();
      // Connection layer.
      model.Add<Linear>(H3);
      // Activation layer.
      model.Add<LeakyReLU>();
      // Connection layer => output.
      // The output of one neuron is the regression output for one record.
      model.Add<Linear>(1);
    }

    // Set parameters for the Stochastic Gradient Descent (SGD) optimizer.
    ens::Adam optimizer(
        STEP_SIZE,  // Step size of the optimizer.
        BATCH_SIZE, // Batch size. Number of data points that are used in each
                    // iteration.
        0.9,        // Exponential decay rate for the first moment estimates.
        0.999,      // Exponential decay rate for the weighted infinity norm
                    // estimates.
        1e-8, // Value used to initialise the mean squared gradient parameter.
        trainData.n_cols * EPOCHS, // Max number of iterations.
        STOP_TOLERANCE,            // Tolerance.
        true);

    model.Train(trainX,
                trainY,
                optimizer,
                // PrintLoss Callback prints loss for each epoch.
                ens::PrintLoss(),
                // Progressbar Callback prints progress bar for each epoch.
                ens::ProgressBar(),
                // Stops the optimization process if the loss stops decreasing
                // or no improvement has been made. This will terminate the
                // optimization once we obtain a minima on training set.
                ens::EarlyStopAtMinLoss(20));

    cout << "Finished training. \nSaving Model" << endl;
    data::Save(modelFile, "NNRegressor", model);
    cout << "Model saved in " << modelFile << endl;
  }

  // NOTE: the code below is added in order to show how in a real application
  // the model would be saved, loaded and then used for prediction.
  // The following steps will be performed after normalizing the dataset.
  FFN<MeanSquaredError, HeInitialization> modelP;
  // Load weights into the model.
  data::Load(modelFile, "NNRegressor", modelP);

  // Create predictions on the dataset.
  arma::mat predOut;
  modelP.Predict(validX, predOut);

  // We will test the quality of our model by calculating Mean Squared Error on
  // validation dataset.
  double validMSE = ComputeMSE(validY, predOut);
  cout << "Mean Squared Error on Prediction data points: " << validMSE << endl;

  // To get meaningful predictions we need to undo the scaling operation on
  // predictions.
  scaleY.InverseTransform(predOut, predOut);
  // Save the prediction results.
  bool saved = data::Save("results.csv", predOut, true);

  if (!saved)
    cout << "Results have not been saved." << endl;
}
