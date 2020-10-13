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
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>

#include "vae_utils.hpp"

#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;

using namespace ens;

// Convenience typedefs
typedef FFN<
    ReconstructionLoss<arma::mat, arma::mat, BernoulliDistribution<arma::mat>>,
    HeInitialization>
    ReconModel;

typedef FFN<MeanSquaredError<>, HeInitialization> MeanSModel;

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
  // The maximum number of possible iteration
  constexpr int maxIteration = 0;
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
  data::Load("./../data/mnist_full.csv", fullData, true, false);
  fullData /= 255.0;

  if (isBinary)
  {
    fullData = arma::conv_to<arma::mat>::from(
        arma::randu<arma::mat>(fullData.n_rows, fullData.n_cols) <= fullData);
  }
  else
    fullData = (fullData - 0.5) * 2;

  arma::mat train, validation;
  data::Split(fullData, validation, train, trainRatio);

  // Loss is calculated on train_test data after each cycle.
  arma::mat train_test, dump;
  data::Split(train, dump, train_test, 0.045);

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
   *         size 15x15, stride = 1, padding = 1) -> 28x28x1
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
    // To use a Sequential object as the first layer, we need to add an
    // identity layer before it.
    vaeModel.Add<IdentityLayer<>>();

    /*
     * Encoder.
     */
    Sequential<>* encoder = new Sequential<>();

    // Add the first convolution layer.
    encoder->Add<Convolution<>>(1,   // Number of input activation maps
                                16,  // Number of output activation maps.
                                5,   // Filter width.
                                5,   // Filter height.
                                2,   // Stride along width.
                                2,   // Stride along height.
                                2,   // Padding width.
                                2,   // Padding height.
                                28,  // Input width.
                                28); // Input height.

    // Add first ReLU.
    encoder->Add<LeakyReLU<>>();
    // Add the second convolution layer.
    encoder->Add<Convolution<>>(16, 24, 5, 5, 1, 1, 0, 0, 14, 14);
    // Add the second ReLU.
    encoder->Add<LeakyReLU<>>();
    // Add the final dense layer.
    encoder->Add<Linear<>>(10 * 10 * 24, 2 * latentSize);

    vaeModel.Add(encoder);

    /*
     * Reparamterization layer.
     */
    vaeModel.Add<Reparametrization<>>(latentSize);

    /*
     * Decoder.
     */
    Sequential<>* decoder = new Sequential<>();

    decoder->Add<Linear<>>(latentSize, 10 * 10 * 24);
    decoder->Add<LeakyReLU<>>();

    // Add the first transposed convolution(deconvolution) layer.
    decoder->Add<TransposedConvolution<>>(
        24,  // Number of input activation maps.
        16,  // Number of output activation maps.
        5,   // Filter width.
        5,   // Filter height.
        1,   // Stride along width.
        1,   // Stride along height.
        0,   // Padding width.
        0,   // Padding height.
        10,  // Input width.
        10); // Input height.

    decoder->Add<LeakyReLU<>>();
    decoder->Add<TransposedConvolution<>>(16, 1, 15, 15, 1, 1, 1, 1, 14, 14);

    vaeModel.Add(decoder);
  }

  std::cout << "Training ..." << std::endl;

  // Set parameters for the Adam optimizer.
  ens::Adam optimizer(
      stepSize,  // Step size of the optimizer.
      batchSize, // Batch size. Number of data points that are used in each
                 // iteration.
      0.9,       // Exponential decay rate for the first moment estimates.
      0.999, // Exponential decay rate for the weighted infinity norm estimates.
      1e-8,  // Value used to initialise the mean squared gradient parameter.
      maxIteration, // Max number of iterations.
      1e-8,         // Tolerance.
      true);

  std::cout << "Initial loss -> "
            << MeanTestLoss<MeanSModel>(vaeModel, train_test, 50) << std::endl;

  // Train neural network. If this is the first iteration, weights are
  // random, using current values as starting point otherwise.
  vaeModel.Train(train,
                 train,
                 optimizer,
                 ens::PrintLoss(),
                 ens::ProgressBar(),
                 // Stop the training using Early Stop at min loss.
                 ens::EarlyStopAtMinLoss());

  // Save the model if specified.
  if (saveModel)
  {
    data::Save("vae/saved_models/vaeCNN.bin", "vaeCNN", vaeModel);
    std::cout << "Model saved in vae/saved_models." << std::endl;
  }
}
