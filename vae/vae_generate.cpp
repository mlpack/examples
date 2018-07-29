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
  constexpr bool loadData = true;
  // The number of samples to generate.
  constexpr size_t nofSamples = 20;
  // Whether modelled on binary data.
  constexpr bool isBinary = false;
  // the latent size of the VAE model.
  constexpr size_t latentSize = 10;

  arma::mat fullData, train, validation;

  if (loadData)
  {
    data::Load("../build/vae/mnist_full.csv", fullData, true, false);
    fullData /= 255.0;

    if (isBinary)
    {
      fullData = arma::conv_to<arma::mat>::from(arma::randu<arma::mat>
          (fullData.n_rows, fullData.n_cols) <= fullData);
    }
    else
      fullData = (fullData - 0.5) * 2;

    data::Split(fullData, validation, train, 0.8);
  }

  arma::arma_rng::set_seed_random();

  // It doesn't matter what type of network we initialize, as we only need to
  // forward pass throught it and not initialize weights or take loss.
  FFN<> vaeModel;

  // Load the trained model.
  if (isBinary)
  {
    data::Load("../build/vae/saved_models/vaeBinaryMS.xml",
        "vaeBinaryMS", vaeModel);
    vaeModel.Add<SigmoidLayer<> >();
  }
  else
  {
    data::Load("../build/vae/saved_models/vaeMS.bin", "vaeMS", vaeModel);
  }

  arma::mat gaussianSamples, outputDists, samples;

  /*
   * Sampling from the prior.
   */
  gaussianSamples = arma::randn<arma::mat>(latentSize, nofSamples);

  // Forward pass only through the decoder(and Sigmod layer in case of binary).
  vaeModel.Forward(gaussianSamples,
                   outputDists,
                   3 /* Index of the decoder */,
                   3 + (size_t)isBinary /* Index of the last layer */);

  GetSample(outputDists, samples, isBinary);
  // Save the prior samples as csv.
  data::Save("samples_csv_files/samples_prior.csv", samples, false, false);

  /*
   * Sampling from the prior by varying all latent variables.
   */
  arma::mat gaussianVaried;

  for (int i = 0; i < latentSize; i++)
  {
    gaussianSamples = arma::randn<arma::mat>(latentSize, 1);
    gaussianVaried = arma::zeros(latentSize, nofSamples);
    gaussianVaried.each_col() = gaussianSamples;

    for (int j = 0; j < nofSamples; j++)
    {
      gaussianVaried.col(j)(i) = -1.5 + j * (3.0 / nofSamples);
    }

    // Forward pass only through the decoder(and Sigmod layer in case of binary).
    vaeModel.Forward(gaussianVaried,
                     outputDists,
                     3 /* Index of the decoder */,
                     3 + (size_t)isBinary /* Index of the last layer */);

    GetSample(outputDists, samples, isBinary);
    // Save the prior samples as csv.
    data::Save("samples_csv_files/samples_prior_latent" + std::to_string(i) + ".csv",
        samples, false, false);
  }

  /*
   * Sampling from the prior by varying two latent variables in 2d.
   */
  size_t latent1 = 3; // Latent variable to be varied vertically.
  size_t latent2 = 4; // Latent variable to be varied horizontally.

  for (int i = 0; i < nofSamples; i++)
  {
    gaussianVaried = arma::zeros(latentSize, nofSamples);

    for (int j = 0; j < nofSamples; j++)
    {
      // Set the vertical variable to a constant value for the outer loop.
      gaussianVaried.col(j)(latent1) = 1.5 - i * (3.0 / nofSamples);
      // Vary the horizontal variable from -1.5 to 1.5.
      gaussianVaried.col(j)(latent2) = -1.5 + j * (3.0 / nofSamples);
    }

    // Forward pass only through the decoder(and Sigmod layer in case of binary).
    vaeModel.Forward(gaussianVaried,
                     outputDists,
                     3 /* Index of the decoder */,
                     3 + (size_t)isBinary /* Index of the last layer */);

    GetSample(outputDists, samples, isBinary);
    // Save the prior samples as csv.
    data::Save("samples_csv_files/samples_prior_latent_2d" + std::to_string(i)
        + ".csv", samples, false, false);
  }

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
    // Save the posterior samples as csv.
    data::Save("samples_csv_files/samples_posterior.csv", samples, false, false);
  }
}
