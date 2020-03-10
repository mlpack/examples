/**
 * @file simple_lstm.hpp
 * @author Mehul Kumar Nirala
 * @author Zoltan Somogyi
 * @author Kartik Dutt
 * 
 * Definition of Simple LSTM generally used for time series preiction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_SIMPLE_LSTM_HPP
#define MODELS_SIMPLE_LSTM_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Definition of a SimpleLSTM.
 */
class SimpleLSTM
{
 public:
  //! Create the SimpleLSTM object.
  SimpleLSTM();

  /**
   * SimpleLSTM constructor intializes input shape, number of classes
   * and width multiplier.
   *
   * @param inputLength Input-Length for linear layer.
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param weights One of 'none', 'Google-Stock-Prices.csv' or path to weights.
   */
  SimpleLSTM(const size_t inputLength,
             const size_t outputSize = 1000,
             const size_t maxRho = 3,
             const size_t H1 = 100,
             const std::string &weights = "none");

  //! Get Layers of the model.
  Sequential<>* GetModel() { return model; };

  //! Load weights into the model.
  Sequential<>* LoadModel(const std::string &filePath);

  //! Save weights for the model.
  void SaveModel(const std::string& filePath);

 private:
  //! Locally stored SimpleLSTM Model.
  Sequential<>* model;

  //! Locally stored value of maxRho.
  size_t maxRho;

  //! Locally stored output channels for first linear layer.
  int H1;

  //! Locally stored length for input to linear layer.
  size_t inputLength;

  //! Locally stored length for input to linear layer.
  size_t outputSize;

  //! Locally stored type of pre-trained weights.
  std::string weights;
}; // class SimpleLSTM

} // namespace ann
} // namespace mlpack

#include "simple_lstm_impl.hpp" // Include implementation.

#endif
