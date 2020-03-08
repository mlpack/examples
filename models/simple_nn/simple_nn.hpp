/**
 * @file simple_nn.hpp
 * @author Eugene Freyman
 * @author Manthan-R-Sheth
 * @author Kartik Dutt
 * 
 * Definition of Simple Neural Network generally used for classification.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_SIMPLE_NN_HPP
#define MODELS_SIMPLE_NN_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Definition of a SimpleNN CNN.
 */
template<
  typename OutputLayerType = NegativeLogLikelihood<>,
  typename InitializationRuleType = RandomInitialization
>
class SimpleNN
{
 public:
  //! Create the SimpleNN object.
  SimpleNN();

  /**
   * SimpleNN constructor intializes input shape, number of classes
   * and width multiplier.
   *
   * @param inputLength Input-Length for linear layer.
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param weights One of 'none', 'mnist'(pre-training on mnist) or path to weights.
   */
  SimpleNN(const size_t inputLength,
           const size_t numClasses = 1000,
           const std::tuple<int, int> hiddenOutSize = std::tuple<int, int>(100, 100),
           const bool useBatchNorm = true,
           const std::string &weights = "none");

  //! Get Layers of the model.
  FFN<OutputLayerType,InitializationRuleType> GetModel() { return model; };

  //! Load weights into the model.
  FFN<OutputLayerType, InitializationRuleType> LoadModel(const std::string &filePath);

  //! Save weights for the model.
  void SaveModel(const std::string& filePath);

 private:
  //! Locally stored SimpleNN Model.
   FFN<OutputLayerType, InitializationRuleType> model;

   //! Locally stored boolean to determine whether or not to use batch norm layer.
   bool useBatchNorm;

   //! Locally stored output channels for first linear layer.
   int H1;

   //! Locally stored output channels for second linear layer.
   int H2;

   //! Locally stored length for input to linear layer.
   size_t inputLength;

   //! Locally stored number of output classes.
   size_t numClasses;

   //! Locally stored type of pre-trained weights.
   std::string weights;
}; // class SimpleNN

} // namespace ann
} // namespace mlpack

#include "simple_nn_impl.hpp" // Include implementation.

#endif
