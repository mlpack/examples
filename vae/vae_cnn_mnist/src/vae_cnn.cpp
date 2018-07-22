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

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/adam/adam_update.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

int main()
{
  // Training data is randomly taken from the dataset in this ratio.
  constexpr double trainRatio = 0.8;
  // The latent size of the VAE model.
  constexpr int latentSize = 20;
  // The batch size.
  constexpr int batchSize = 50;
  // The step size of the optimizer.
  constexpr double stepSize = 0.0008;
  // The number of interations per cycle.
  constexpr int iterPerCycle = 56000;
  // Number of cycles.
  constexpr int cycles = 3;
  // Whether to load a model to train.
  constexpr bool loadModel = true;
  // Whether to save the trained model.
  constexpr bool saveModel = true;

  std::cout << "Reading data ..." << std::endl;

  // Entire dataset(without labels) is loaded from a CSV file.
  // Each column represents a data point.
  arma::mat fullData;
  data::Load("vae/mnist_full.csv", fullData, true, false);
  fullData /= 255.0;
  fullData = (fullData - 0.5) * 2;

  arma::mat train, validation;
  data::Split(fullData, validation, train, trainRatio);

  // Loss is calculated on train_test data after each cycle.
  arma::mat train_test, dump;
  data::Split(train, dump, train_test, 0.01);

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
  FFN<MeanSquaredError<>, HeInitialization> vaeModel;

  if (loadModel)
  {
    std::cout << "Loading model ..." << std::endl;
    data::Load("vae/saved_models/vaeModelCNN.xml", "vaeModelCNN", vaeModel);
  }
  else
  {
    // To use a Sequential object as the first layer, we need to add an
    // identity layer before it.
    vaeModel.Add<IdentityLayer<> >();

    /*
     * Encoder.
     */
    Sequential<>* encoder = new Sequential<>();

    // Add the first convolution layer.
    encoder->Add<Convolution<> >(
        1,  // Number of input activation maps
        16, // Number of output activation maps.
        5,  // Filter width.
        5,  // Filter height.
        2,  // Stride along width.
        2,  // Stride along height.
        2,  // Padding width.
        2,  // Padding height.
        28, // Input width.
        28);// Input height.

    // Add first ReLU.
    encoder->Add<LeakyReLU<> >();
    // Add the second convolution layer.
    encoder->Add<Convolution<> >(16, 24, 5, 5, 1, 1, 0, 0, 14, 14);
    // Add the second ReLU.
    encoder->Add<LeakyReLU<> >();
    // Add the final dense layer.
    encoder->Add<Linear<> >(10*10*24, 2 * latentSize);

    vaeModel.Add(encoder);

    /*
     * Reparamterization layer.
     */
    vaeModel.Add<Reparametrization<> >(latentSize);

    /*
     * Decoder.
     */
    Sequential<>* decoder = new Sequential<>();

    decoder->Add<Linear<> >(latentSize, 10*10*24);
    decoder->Add<LeakyReLU<> >();

    // Add the first transposed convolution(deconvolution) layer.
    decoder->Add<TransposedConvolution<> >(
        24, // Number of input activation maps.
        16, // Number of output activation maps.
        5,  // Filter width.
        5,  // Filter height.
        1,  // Stride along width.
        1,  // Stride along height.
        0,  // Padding width.
        0,  // Padding height.
        10, // Input width.
        10);// Input height.

    decoder->Add<LeakyReLU<> >();
    decoder->Add<TransposedConvolution<> >(16, 1, 15, 15, 1, 1, 1, 1, 14, 14);

    vaeModel.Add(decoder);
  }

  std::cout << "Training ..." << std::endl;

  // Setting parameters for the Stochastic Gradient Descent (SGD) optimizer.
  SGD<AdamUpdate> optimizer(
    // Step size of the optimizer.
    stepSize,
    // Number of data points that are used in each iteration.
    batchSize,
    // Max number of iterations.
    iterPerCycle,
    // Tolerance, used as a stopping condition. This small number means we never
    // stop by this condition and continue to optimize up to reaching maximum of
    // iterations.
    1e-8,
    // Shuffle, If optimizer should take random data points from the dataset at
    // each iteration.
    true,
    // Adam update policy.
    AdamUpdate());

  std::cout << "Initial loss -> " << vaeModel.Evaluate(train_test, train_test)
      << std::endl;

  const clock_t begin_time = clock();

  // Cycles for monitoring the progress.
  for (int i = 0; i < cycles; i++)
  {
    // Train neural network. If this is the first iteration, weights are
    // random, using current values as starting point otherwise.
    vaeModel.Train(train, train, optimizer);

    // Don't reset optimizer's parameters between cycles.
    optimizer.ResetPolicy() = false;

    std::cout << "Loss after cycle  " << i << " -> " <<
        vaeModel.Evaluate(train_test, train_test) << std::endl;
    std::cout << "Time taken for cycle -> " << float(clock() - begin_time) /
        CLOCKS_PER_SEC << " seconds" << std::endl;

    if (saveModel)
      data::Save("vae/saved_models/vaeModelCNN.xml", "vaeModelCNN", vaeModel);
  }

  std::cout << "Time taken to train -> " << float(clock() - begin_time) /
      CLOCKS_PER_SEC << " seconds" << std::endl;

  // Save the model if specified.
  if (saveModel)
  {
    std::cout << "Model saved in vae/saved_models." << std::endl;
  }
}
