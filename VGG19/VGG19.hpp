/**
 * @file VGG19.hpp
 * @author Mehul Kumar Nirala
 *
 * An implementation of VGG19.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef VGGNET_HPP
#define VGGNET_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;

/**
 * VGG-19 is a convolutional neural network that is trained on more than a
 * million images from the ImageNet database. The network is 19 layers
 * deep and can classify images into 1000 object categories. Details about
 * the network architecture can be found in the following arXiv paper:
 *
 * Very Deep Convolutional Networks for Large-Scale Image Recognition
 * K. Simonyan, A. Zisserman
 * arXiv:1409.1556
 * https://arxiv.org/abs/1409.1556
 */

class VGG19
{
public:
  /**
  * VGG19 Constructor initializes the image input shape,
  * and numClasses.
  *
  * @param inputWidth Width of the input image.
  * @param inputHeight Height of the input image.
  * @param inputChannels Number of input channels of the input image.
  * @param numClasses optional number of classes to classify images into,
  *      only to be specified if include_top is  true.
  * @param includeTop Whether to include the 3 fully-connected layers at
  *      the top of the network.
  * @param pooling Optional pooling mode for feature extraction when
  *      include_top is false.
  * @param weights One of 'none', 'imagenet'(pre-training on ImageNet).
  */
  VGG19(const size_t inputWidth,
        const size_t inputHeight,
        const size_t inputChannel,
        const size_t numClasses,
        const bool includeTop = true,
        const std::string& pooling = "max",
        const std::string& weights = "imagenet");

  ~VGG19();

  Sequential<>* CompileModel();

  /**
  * Load model from a path.
  * 
  *
  * @param filePath Path to load the model from.
  * @return Sequential Pointer to a sequential model.
  */
  Sequential<>* LoadModel(const std::string& filePath);
  
  /**
  * Save model to a location.
  *
  * @param filePath Path to save the model to.
  */
  void SaveModel(const std::string& filePath);

  size_t GetOutputShape();

  Sequential<>* GetModel();

private:

  // VGGNet Model.
  Sequential<>* VGGNet;

  // Width of the image.
  size_t inputWidth;

  // Height of the image.
  size_t inputHeight;

  // Number of channels in the image.
  size_t inputChannel;

  // Number of output classes.
  size_t numClasses;

  // Include the final dense layer.
  bool includeTop;

  // Parameter for final pooling layer.
  std::string pooling;

  // Type of pre-trained weights.
  std::string weights;

  // Stores the output shape of the VGG19
  size_t outputShape;
  
};

#include "VGG19_impl.hpp"

#endif
