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
#include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>

#include "vae_utils.hpp"

using namespace mlpack;
using namespace mlpack::ann;

// Convenience typedef
typedef FFN<ReconstructionLoss<arma::mat,
                               arma::mat,
                               BernoulliDistribution<arma::mat> >,
            HeInitialization> ReconModel;

int main()
{
  // Whether to load training data.
  constexpr bool loadData = false;
  // The number of samples to generate.
  constexpr size_t nofSamples = 20;
  // Whether modelled on binary data.
  constexpr bool isBinary = false;
  // the latent size of the VAE model.
  constexpr size_t latentSize = 20;

  arma::mat fullData, train, validation;

  if (loadData)
  {
    data::Load("../build/vae/mnist_full.csv", fullData, true, false);
    fullData /= 255.0;
    fullData = (fullData - 0.5) * 2;

    data::Split(fullData, validation, train, 0.8);
  }

  arma::arma_rng::set_seed_random();

  FFN<MeanSquaredError<>, HeInitialization> vaeModel;
  // Load the trained model.
  if (isBinary)
  {
    data::Load("../build/vae/saved_models/vaeModelBinary.xml",
        "vaeModelBinary", vaeModel);
    vaeModel.Add<SigmoidLayer<> >();
  }
  else
  {
    data::Load("../build/vae/saved_models/vaeBeta.xml", "vaeBeta", vaeModel);
  }

  arma::mat outputDists, samples;

  /*
   * Sampling from the prior.
   */
  arma::mat gaussSamples = arma::randn<arma::mat>(latentSize, nofSamples);

  // Forward pass only through the decoder(and Sigmod layer in case of binary).
  vaeModel.Forward(gaussSamples,
                   outputDists,
                   3 /* Index of the decoder */,
                   3 + (size_t)isBinary /* Index of the last layer */);

  GetSample(outputDists, samples, isBinary);
  // Save the prior samples as csv.
  data::Save("samples_prior.csv", samples, false, false);

  /*
   * Sampling from the posterior.
   */
  if (loadData)
  {
  // Forward pass through the entire network given an input datapoint.
  vaeModel.Forward(validation.cols(0, 19),
                   outputDists,
                   1 /* Index of the encoder */,
                   3 + (size_t)isBinary /* Index of the last layer */);

  GetSample(outputDists, samples, isBinary);
  // Save the posterior samplesa s csv.
  data::Save("samples_posterior.csv", samples, false, false);
  }
}
