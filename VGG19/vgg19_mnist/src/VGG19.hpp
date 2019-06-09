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

class VGG19
{
public:

  VGG19(const size_t inputWidth,
        const size_t inputHeight,
        const size_t inputChannel,
        const size_t numClasses,
        const bool includeTop = true,
        const std::string pooling = "max",
        const std::string weights = "imagenet"
        );

  ~VGG19();

  Sequential<>* CompileModel();

  Sequential<>* LoadModel(const std::string filePath);
  
  void SaveModel(const std::string filePath);

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

  // Type of pre-trained weights.
  std::string weights;

  // Include the final dense layer.
  bool includeTop;

  // Parameter for final pooling layer.
  std::string pooling;

  // Stores the output shape of the VGG19
  size_t outputShape;
  
};

#include "VGG19_impl.hpp"

#endif