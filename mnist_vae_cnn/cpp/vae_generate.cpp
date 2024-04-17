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
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

#include "vae_utils.hpp"

using namespace mlpack;

// Convenience typedef
typedef FFN<ReconstructionLoss, HeInitialization> ReconModel;

int main()
{
  // Whether to load training data.
  constexpr bool loadData = true;
  // The number of samples to generate.
  constexpr size_t nofSamples = 20;
  // Whether modelled on binary data.
  constexpr bool isBinary = false;
  // the latent size of the VAE model.
  constexpr size_t latentSize = 20;

  arma::mat fullData, train, validation;

  if (loadData)
  {
    data::Load("../data/mnist_train.csv", fullData, true, false);
    // Get rid of the header.
    fullData = fullData.submat(0, 1, fullData.n_rows - 1, fullData.n_cols - 1);
    fullData /= 255.0;
    // Get rid of the labels.
    fullData = fullData.submat(1, 0, fullData.n_rows - 1, fullData.n_cols - 1);

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
    data::Load("./saved_models/vaeBinaryMS.xml", "vaeBinaryMS", vaeModel);
    vaeModel.Add<Sigmoid>();
  }
  else
  {
    data::Load("./saved_models/vaeCNN.bin", "vaeMS", vaeModel);
  }

  arma::mat gaussianSamples, outputDists, samples;

  /*
   * Sampling from the prior.
   */
  gaussianSamples = arma::randn<arma::mat>(latentSize, nofSamples);

  // Forward pass only through the decoder (and Sigmoid layer in case of
  // binary).

  // NOTE: these layer indexes are wrong and will need to be adapted once this
  // example builds correctly.
  vaeModel.Forward(gaussianSamples,
                   outputDists,
                   3 /* Index of the decoder */,
                   3 + (size_t) isBinary /* Index of the last layer */);

  GetSample(outputDists, samples, isBinary);
  // Save the prior samples as csv.
  data::Save("./samples_csv_files/samples_prior.csv", samples, false, false);

  /*
   * Sampling from the prior by varying all latent variables.
   */
  arma::mat gaussianVaried;

  for (size_t i = 0; i < latentSize; i++)
  {
    gaussianSamples = arma::randn<arma::mat>(latentSize, 1);
    gaussianVaried = arma::zeros(latentSize, nofSamples);
    gaussianVaried.each_col() = gaussianSamples;

    for (size_t j = 0; j < nofSamples; j++)
    {
      gaussianVaried.col(j)(i) = -1.5 + j * (3.0 / nofSamples);
    }

    // Forward pass only through the decoder (and Sigmoid layer in case of
    // binary).
    vaeModel.Forward(gaussianVaried,
                     outputDists,
                     3 /* Index of the decoder */,
                     3 + (size_t) isBinary /* Index of the last layer */);

    GetSample(outputDists, samples, isBinary);
    // Save the prior samples as csv.
    data::Save(
        "./samples_csv_files/samples_prior_latent" + std::to_string(i) + ".csv",
        samples,
        false,
        false);
  }

  /*
   * Sampling from the prior by varying two latent variables in 2d.
   */
  size_t latent1 = 3; // Latent variable to be varied vertically.
  size_t latent2 = 4; // Latent variable to be varied horizontally.

  for (size_t i = 0; i < nofSamples; i++)
  {
    gaussianVaried = arma::zeros(latentSize, nofSamples);

    for (size_t j = 0; j < nofSamples; j++)
    {
      // Set the vertical variable to a constant value for the outer loop.
      gaussianVaried.col(j)(latent1) = 1.5 - i * (3.0 / nofSamples);
      // Vary the horizontal variable from -1.5 to 1.5.
      gaussianVaried.col(j)(latent2) = -1.5 + j * (3.0 / nofSamples);
    }

    // Forward pass only through the decoder
    // (and Sigmod layer in case of binary).
    vaeModel.Forward(gaussianVaried,
                     outputDists,
                     3 /* Index of the decoder */,
                     3 + (size_t)isBinary /* Index of the last layer */);

    GetSample(outputDists, samples, isBinary);
    // Save the prior samples as csv.
    data::Save("./samples_csv_files/samples_prior_latent_2d" + std::to_string(i)
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
    data::Save(
      "./samples_csv_files/samples_posterior.csv",
      samples,
      false,
      false);
  }
}
