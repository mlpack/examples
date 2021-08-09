/**
 * @file gan_utils.cpp
 * @author Roshan Swain
 * @author Atharva Khandait
 *
 * Utility function necessary for  working with GAN models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_GAN_UTILS_HPP
#define MODELS_GAN_UTILS_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;

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