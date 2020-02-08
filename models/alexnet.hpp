/**
 * @file alexnet.hpp
 * @author Kartik Dutt
 * 
 * Implementation of alex-net using mlpack.
 * 
 * For more information, see the following paper.
 * 
 * @code
 * @misc{
 *   author = {Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton},
 *   title = {ImageNet Classification with Deep Convolutional Neural Networks},
 *   year = {2012}
 * }
 * @endcode
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_ALEXNET_HPP
#define MODELS_ALEXNET_HPP

// Include all required libraries.
#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;

class AlexNet
{
 public:
  //! Create the AlexNet object.
  AlexNet();

  /**
   * AlexNet constructor intializes input shape, number of classes
   * and width multiplier.
   *  
   * @param inputWidth Width of the input image.
   * @param inputHeight Height of the input image.
   * @param inputChannels Number of input channels of the input image.
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param includeTop whether to include the fully-connected layer at 
   *        the top of the network.
   * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
   */
  AlexNet(const size_t inputWidth,
          const size_t inputHeight,
          const size_t inputChannel,
          const size_t numClasses = 1000,
          const bool includeTop = true,
          const std::string &weights = "None");

  /**
   * AlexNet constructor intializes input shape, number of classes
   * and width multiplier.
   *  
   * @param inputShape A three-valued tuple indicating input shape.
   *                   First value is number of Channels (Channels-First).
   *                   Second value is input height.
   *                   Third value is input width..
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param includeTop whether to include the fully-connected layer at 
   *        the top of the network.
   * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
   */
  AlexNet(const std::tuple<size_t, size_t, size_t> inputShape,
          const size_t numClasses = 1000,
          const bool includeTop = true,
          const std::string &weights = "None");

  // Custom Destructor.
  ~AlexNet();

  /** 
   * Defines Model Architecture.
   * 
   * @return Sequential Pointer to the sequential AlexNet model.
   */
  Sequential<>* CompileModel();

  /**
   * Load model from a path.
   * 
   *
   * @param filePath Path to load the model from.
   * @return Sequential Pointer to a sequential model.
   */
  Sequential<>* LoadModel(const std::string &filePath);

  /**
   * Save model to a location.
   *
   * @param filePath Path to save the model to.
   */
  void SaveModel(const std::string &filePath);

  /**
   * Return output shape of model.
   * @returns outputShape of size_t type.
   */
  size_t OutputShape() { return outputShape; };

  /**
   * Returns compiled version of model.
   * If called without compiling would result in empty Sequetial
   * Pointer.
   * 
   * @return Sequential Pointer to a sequential model.
   */
  Sequential<>* GetModel() { return alexNet; };

 private:
  //! Locally stored AlexNet Model.
  Sequential<>* alexNet;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored include the final dense layer.
   bool includeTop;

  //! Locally stored type of pre-trained weights.
  std::string weights;

  //! Locally stored output shape of the VGG19
  size_t outputShape;
};

#include "alexnet_impl.hpp"

#endif
