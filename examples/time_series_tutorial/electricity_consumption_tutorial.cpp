#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <dataloaders/dataloader.hpp>
#include <utils/utils.hpp>
#include <models/lstm/simple_lstm.hpp>
#include <ensmallen.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

int main()
{
  const size_t BATCH_SIZE = 8;
  // Univariate.
  size_t inputSize = 1, outputSize = 1;
  // Number of timesteps to look backwards in RNN.
  const size_t rho = 10;
  // Number of cells in the LSTM (hidden layers in standard terms)
  // NOTE: you may play with this variable in order to further optimize the
  // model.  (as more cells are added, accuracy is likely to go up, but training
  // time may take longer)
  const int H1 = 10;
  // Max rho for LSTM.
  const size_t maxRho = rho;

  DataLoader<arma::cube, arma::cube> dataloader("electricity-consumption", 0.8, rho);

  // Scale the training and testing data.
  data::MinMaxScaler scaler;
  scaler.Fit(dataloader.TrainCSVData());
  scaler.Transform(dataloader.TestCSVData(), dataloader.TestCSVData());
  scaler.Transform(dataloader.TrainCSVData(), dataloader.TestCSVData());

  CreateTimeSeriesData(dataloader.TestCSVData(), dataloader.TrainX(),
      dataloader.TrainY(), rho, 0, dataloader.TrainCSVData().n_rows - 1, 1, 1);
  CreateTimeSeriesData(dataloader.TestCSVData(), dataloader.TestX(),
      dataloader.TestY(), rho, 0, dataloader.TestCSVData().n_rows - 1, 1, 1);

  RNN<MeanSquaredError<>, HeInitialization> model(rho);
  SimpleLSTM module1(inputSize, outputSize, H1);
  Sequential<>* layers = module1.GetModel();
  model.Add<IdentityLayer<>>();
  model.Add(layers);

  const double STEP_SIZE = 5e-5;
  const int EPOCHS = 150;

  SGD<AdamUpdate> optimizer(
      STEP_SIZE,                             // Step size of the optimizer.
      BATCH_SIZE,                            // Batch size.
      dataloader.TrainY().n_cols * EPOCHS,   // Max number of iterations.
      1e-8,                                  // Tolerance.
      true,                                  // Shuffle.
      AdamUpdate(1e-8, 0.9, 0.999));         // Adam update policy.
  cout << "Training." << endl;

  model.Train(dataloader.TrainX(),
              dataloader.TrainY(),
              optimizer,
              ens::PrintLoss(),   // PrintLoss Callback prints loss for each epoch.
              ens::ProgressBar(), // Progressbar Callback prints progress bar for each epoch.
              ens::EarlyStopAtMinLoss());

  cout << "Validation Results: \n MSE:" << std::endl;
  arma::cube preds;
  model.Predict(dataloader.TestX(), preds);
  cout << MSE(preds, dataloader.TestY()) << endl;
}