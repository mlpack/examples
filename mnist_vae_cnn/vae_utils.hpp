/**
 * @file vae_utils.cpp
 * @author Atharva Khandait
 *
 * Utility function necessary for training and working with VAE models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_VAE_UTILS_HPP
#define MODELS_VAE_UTILS_HPP

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

using namespace mlpack;

// Calculates mean loss over batches.
template<typename NetworkType = FFN<MeanSquaredError, HeInitialization>,
         typename DataType = arma::mat>
double MeanTestLoss(NetworkType& model, DataType& testSet, size_t batchSize)
{
  double loss = 0;
  size_t nofPoints = testSet.n_cols;
  size_t i;

  for (i = 0; i < (size_t) nofPoints / batchSize; i++)
  {
    loss +=
        model.Evaluate(testSet.cols(batchSize * i, batchSize * (i + 1) - 1),
                       testSet.cols(batchSize * i, batchSize * (i + 1) - 1));
  }

  if (nofPoints % batchSize != 0)
  {
    loss += model.Evaluate(testSet.cols(batchSize * i, nofPoints - 1),
                           testSet.cols(batchSize * i, nofPoints - 1));
    loss /= (int) nofPoints / batchSize + 1;
  }
  else
  {
    loss /= nofPoints / batchSize;
  }

  return loss;
}

// Sample from the output distribution and post-process the outputs(because
// we pre-processed it before passing it to the model).
template<typename DataType = arma::mat>
void GetSample(DataType &input, DataType& samples, bool isBinary)
{
  if (isBinary)
  {
    samples = arma::conv_to<DataType>::from(
        arma::randu<DataType>(input.n_rows, input.n_cols) <= input);
    samples *= 255;
  }
  else
  {
    samples = input / 2 + 0.5;
    samples *= 255;
    samples = arma::clamp(samples, 0, 255);
  }
}

#endif
