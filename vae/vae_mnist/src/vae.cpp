/**
 * @file vae.cpp
 * @author Atharva Khandait
 *
 * A Variational autoencoder(VAE) model to generate MNIST.
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
#include <mlpack/core/optimizers/rmsprop/rmsprop_update.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>

#include <vae/vae_utils.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

// Convenience typedefs
typedef FFN<ReconstructionLoss<arma::mat,
                               arma::mat,
                               BernoulliDistribution<arma::mat> >,
            HeInitialization> ReconModel;

typedef FFN<MeanSquaredError<>, HeInitialization> MeanSModel;

int main()
{
  // Training data is randomly taken from the dataset in this ratio.
  constexpr double trainRatio = 0.8;
  // The number of neurons in the first layer.
  constexpr int h1 = 784;
  // The number of neurons in the second layer.
  constexpr int h2 = 550;
  // The number of neurons in the second layer.
  constexpr int h3 = 400;
  // The number of neurons in the second layer.
  constexpr int h4 = 200;
  // The latent size of the VAE model.
  constexpr int latentSize = 20;
  // The batch size.
  constexpr int batchSize = 100;
  // The step size of the optimizer.
  constexpr double stepSize = 0.001;
  // The number of interations per cycle.
  constexpr int iterPerCycle = 56000;
  // Number of cycles.
  constexpr int cycles = 100;
  // Whether to load a model to train.
  constexpr bool loadModel = false;
  // Whether to save the trained model.
  constexpr bool saveModel = true;
  // Whether to convert to binary MNIST.
  constexpr bool isBinary = true;
  // Beta parameter for disentangled networks.
  // constexpr double beta = 1.5;

  std::cout << "Reading data ..." << std::endl;

  // Entire dataset(without labels) is loaded from a CSV file.
  // Each column represents a data point.
  arma::mat fullData;
  data::Load("vae/mnist_full.csv", fullData, true, false);
  fullData /= 255.0;

  if (isBinary)
  {
    fullData = arma::conv_to<arma::mat>::from(arma::randu<arma::mat>
        (fullData.n_rows, fullData.n_cols) <= fullData);
  }
  else
    fullData = (fullData - 0.5) * 2;

  arma::mat train, validation;
  data::Split(fullData, validation, train, trainRatio);

  // Loss is calculated on train_test data after each cycle.
  arma::mat train_test, dump;
  data::Split(train, dump, train_test, 0.045);

  // Creating the VAE model.
  ReconModel vaeModel;

  if (loadModel)
  {
    std::cout << "Loading model ..." << std::endl;
    data::Load("vae/saved_models/vae.bin", "vae", vaeModel);
  }
  else
  {
    // To use a Sequential object as the first layer, we need to add an
    // identity layer before it.
    vaeModel.Add<IdentityLayer<> >();

    // Encoder.
    Sequential<>* encoder = new Sequential<>();

    encoder->Add<Linear<> >(train.n_rows, h1);
    encoder->Add<ReLULayer<> >();
    encoder->Add<Linear<> >(h1, h2);
    encoder->Add<ReLULayer<> >();
    encoder->Add<Linear<> >(h2, h3);
    encoder->Add<ReLULayer<> >();
    encoder->Add<Linear<> >(h3, h4);
    encoder->Add<ReLULayer<> >();
    encoder->Add<Linear<> >(h4, 2 * latentSize);

    vaeModel.Add(encoder);

    // Reparametrization layer.
    vaeModel.Add<Reparametrization<> >(latentSize);

    // Decoder.
    Sequential<>* decoder = new Sequential<>();

    decoder->Add<Linear<> >(latentSize, h4);
    decoder->Add<ReLULayer<> >();
    decoder->Add<Linear<> >(h4, h3);
    decoder->Add<ReLULayer<> >();
    decoder->Add<Linear<> >(h3, h2);
    decoder->Add<ReLULayer<> >();
    decoder->Add<Linear<> >(h2, h1);

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

  std::cout << "Initial loss -> " <<
      MeanTestLoss<ReconModel>(vaeModel, train_test, 50) << std::endl;

  const clock_t begin_time = clock();

  // Cycles for monitoring the progress.
  for (int i = 0; i < cycles; i++)
  {
    // Train neural network. If this is the first iteration, weights are
    // random, using current values as starting point otherwise.
    vaeModel.Train(train, train, optimizer);

    // Don't reset optimizer's parameters between cycles.
    optimizer.ResetPolicy() = false;

    std::cout << "Loss after cycle " << i << " -> " <<
        MeanTestLoss<ReconModel>(vaeModel, train_test, 50) << std::endl;
  }

  std::cout << "Time taken to train -> " << float(clock() - begin_time) /
      CLOCKS_PER_SEC << " seconds" << std::endl;

  if (saveModel)
  {
    data::Save("vae/saved_models/vae.bin", "vae", vaeModel);
    std::cout << "Model saved in vae/saved_models/." << std::endl;
  }
}
