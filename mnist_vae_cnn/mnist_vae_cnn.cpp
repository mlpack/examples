/**
 * @file vae_cnn.cpp
 * @author Atharva Khandait
 *
 * A convolutional Variational autoencoder(VAE) model to generate MNIST.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// NOTE: this example does not currently build!  The Reparametrization and
// TransposedConvolution layers have not yet been adapted to the mlpack 4 layer
// style.  See https://github.com/mlpack/mlpack/pull/2777 for more information.

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

#include "vae_utils.hpp"

using namespace mlpack;
using namespace ens;

// Convenience typedefs
typedef FFN<ReconstructionLoss, HeInitialization> ReconModel;
typedef FFN<MeanSquaredError, HeInitialization> MeanSModel;

int main()
{
  // Training data is randomly taken from the dataset in this ratio.
  constexpr double trainRatio = 0.8;
  // The latent size of the VAE model.
  constexpr int latentSize = 20;
  // The batch size.
  constexpr int batchSize = 64;
  // The step size of the optimizer.
  constexpr double stepSize = 0.001;
  // Number of epochs/ cycle
  constexpr int epochs = 1;
  // Number of cycles
  constexpr int cycles = 10;
  // Whether to load a model to train.
  constexpr bool loadModel = false;
  // Whether to save the trained model.
  constexpr bool saveModel = true;
  // Whether to convert to binary MNIST.
  constexpr bool isBinary = false;

  std::cout << "Reading data ..." << std::endl;

  // Entire dataset(without labels) is loaded from a CSV file.
  // Each column represents a data point.
  arma::mat fullData;
  data::Load("../data/mnist_train.csv", fullData, true, true);

  // Originally on Kaggle dataset CSV file has header, so it's necessary to
  // get rid of this row, in Armadillo representation it's the first column.
  fullData = fullData.submat(0, 1, fullData.n_rows -1, fullData.n_cols - 1);
  fullData /= 255.0;

  // Get rid of the labels
  fullData = fullData.submat(1, 0, fullData.n_rows - 1, fullData.n_cols - 1);

  if (isBinary)
  {
    fullData = arma::conv_to<arma::mat>::from(
        arma::randu<arma::mat>(fullData.n_rows, fullData.n_cols) <= fullData);
  }
  else
  {
    fullData = (fullData - 0.5) * 2;
  }

  arma::mat train, validation;
  data::Split(fullData, validation, train, trainRatio);

  // Loss is calculated on train_test data after each cycle.
  arma::mat trainTest, dump;
  data::Split(train, dump, trainTest, 0.045);

  // No of iterations of the optimizer.
  int iterPerCycle = (epochs * train.n_cols);

  /**
   * Model architecture:
   *
   * Encoder:
   * 28x28x1 ---- conv (16 filters of size 5x5,
   *                  stride = 2, padding = 2) ----> 14x14x16
   * 14x14x16 ------------- Leaky ReLU ------------> 14x14x16
   * 14x14x16 --- conv (24 filters of size 5x5,
   *                   stride = 1, padding = 0) ---> 10x10x24
   * 10x10x24 ------------- Leaky ReLU ------------> 10x10x24
   * 10x10x24 ---------------- Dense --------------> 2 * latentSize
   *
   * Reparametrization layer:
   * 2 * latenSize --------------------------------> latenSize
   *
   * Decoder:
   * latentSize ------------- Dense ---------------> 10x10x24
   * 10x10x24 ------------- Leaky ReLU ------------> 10x10x24
   * 10x10x24 ---- transposed conv (16 filters of
   *         size 5x5, stride = 1, padding = 0) ---> 14x14x16
   * 14x14x16 ------------- Leaky ReLU ------------> 14x14x16
   * 14x14x16 ---- transposed conv (1 filter of
   *         size 15x15, stride = 0, padding = 1) -> 28x28x1
   */

  // Creating the VAE model.
  MeanSModel vaeModel;

  if (loadModel)
  {
    std::cout << "Loading model ..." << std::endl;
    data::Load("vae/saved_models/vaeCNN.bin", "vaeCNN", vaeModel);
  }
  else
  {
    /*
     * Encoder layers.
     */

    // Add the first convolution layer.
    vaeModel.Add<Convolution>(16, // Number of output activation maps.
                              5,  // Filter width.
                              5,  // Filter height.
                              2,  // Stride along width.
                              2,  // Stride along height.
                              2,  // Padding width.
                              2); // Padding height.

    // Add first ReLU.
    vaeModel.Add<LeakyReLU>();
    // Add the second convolution layer.
    vaeModel.Add<Convolution>(24, 5, 5, 1, 1, 0, 0);
    // Add the second ReLU.
    vaeModel.Add<LeakyReLU>();
    // Add the final dense layer.
    vaeModel.Add<Linear>(2 * latentSize);

    /*
     * Reparamtrization layer.
     */
    vaeModel.Add<Reparametrization>(latentSize);

    /*
     * Decoder layers.
     */
    vaeModel.Add<Linear>(10 * 10 * 24);
    vaeModel.Add<LeakyReLU>();

    // Add the first transposed convolution(deconvolution) layer.
    vaeModel.Add<TransposedConvolution>(
        16, // Number of output activation maps.
        5,  // Filter width.
        5,  // Filter height.
        1,  // Stride along width.
        1,  // Stride along height.
        0,  // Padding width.
        0); // Padding height.

    vaeModel.Add<LeakyReLU>();
    vaeModel.Add<TransposedConvolution>(1, 15, 15, 1, 1, 0, 0);
  }

  std::cout << "Training ..." << std::endl;

  // Set parameters for the Adam optimizer.
  Adam optimizer(
      stepSize,  // Step size of the optimizer.
      batchSize, // Batch size. Number of data points that are used in each
                 // iteration.
      0.9,       // Exponential decay rate for the first moment estimates.
      0.999, // Exponential decay rate for the weighted infinity norm estimates.
      1e-8,  // Value used to initialise the mean squared gradient parameter.
      iterPerCycle, // Max number of iterations.
      1e-8,         // Tolerance.
      true);

  const clock_t beginTime = clock();
  // Cycles for monitoring the progress.
  for (int i = 0; i < cycles; i++)
  {
    // Train neural network. If this is the first iteration, weights are
    // random, using current values as starting point otherwise.
    vaeModel.Train(train,
                   train,
                   optimizer,
                   PrintLoss(),
                   ProgressBar(),
                   Report());

    // Don't reset optimizer's parameters between cycles.
    optimizer.ResetPolicy() = false;

    std::cout << "Loss after cycle  " << i << " -> " <<
        MeanTestLoss<MeanSModel>(vaeModel, trainTest, batchSize) << std::endl;
  }

  std::cout << "Time taken to train -> " << float(clock() - beginTime) /
      CLOCKS_PER_SEC << " seconds" << std::endl;

  // Save the model if specified.
  if (saveModel)
  {
    data::Save("./saved_models/vaeCNN.bin", "vaeCNN", vaeModel);
    std::cout << "Model saved in vae/saved_models." << std::endl;
  }
}
