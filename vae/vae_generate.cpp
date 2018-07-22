/**
 * @file vae_generate.cpp
 * @author Atharva Khandait
 *
 * Generate MNIST using trained VAE model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

using namespace mlpack;
using namespace mlpack::ann;

int main()
{
  // Whether to load training data.
  constexpr bool loadData = true;
  // The number of samples to generate.
  constexpr size_t n_samples = 20;

  arma::mat fullData, train, validation;

  if (loadData)
  {
    data::Load("../mnist_full.csv", fullData, true, false);
    fullData /= 255.0;
    fullData = (fullData - 0.5) * 2;

    data::Split(fullData, validation, train, 0.8);
  }

  // Load the trained model.
  FFN<MeanSquaredError<>, HeInitialization> vaeModel;
  data::Load("saved_models/vaeModelCNN.xml", "vaeModelCNN", vaeModel);

  arma::mat outputDists;

  // Generate randomly from the trained distribution.
  arma::mat samples = arma::randn<arma::mat>(20, n_samples);

  vaeModel.Forward(samples, outputDists, 3, 3);
  arma::mat outputSamples = outputDists;

  outputSamples = outputSamples / 2 + 0.5;
  outputSamples *= 255;
  outputSamples = arma::clamp(outputSamples, 0, 255);

  data::Save("outputSamples.csv", outputSamples, false, false);
}
