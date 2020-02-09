/**
 * @file alexnet_impl.hpp
 * @author Kartik Dutt
 * 
 * Implementation of alex-net using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_ALEXNET_HPP
#define MODELS_ALEXNET_HPP

#include "alexnet.hpp"

AlexNet::AlexNet(const size_t inputWidth,
                 const size_t inputHeight,
                 const size_t inputChannel,
                 const size_t numClasses,
                 const bool includeTop,
                 const std::string &weights):
                 inputWidth(inputWidth),
                 inputHeight(inputHeight),
                 inputChannel(inputChannel),
                 numClasses(numClasses),
                 includeTop(includeTop),
                 weights(weights),
                 outputShape(512)
{
  alexNet = new Sequential<>();
}

AlexNet::AlexNet(const std::tuple<size_t, size_t, size_t> inputShape,
                 const size_t numClasses,
                 const bool includeTop,
                 const std::string &weights):
                 inputWidth(std::get<0> inputShape),
                 inputHeight(std::get<1> inputShape),
                 inputChannel(std::get<2> inputShape),
                 numClasses(numClasses),
                 includeTop(includeTop),
                 weights(weights),
                 outputShape(512)
{
  alexNet = new Sequential<>();
}

AlexNet::~AlexNet()
{
  delete alexNet;
}

Sequential<>* AlexNet::CompileModel()
{
  // Add Convlution Layer with inputChannels as input maps,
  // output maps = 64, kernel_size = (11, 11) stride = (4, 4)
  // and padding = (2, 2).
  alexNet->Add<Convolution>(inputChannel, 64, 11, 11,
      4, 4, 2, 2, inputWidth, inputHeight);
  // Add activation function for output.
  alexNet->Add<ReLU<>>();

  // Update inputHeight and inputWidth for next layers.
  inputHeight = ConvOutSize(inputHeight, 11, 4, 2, 2);
  inputWidth = ConvOutSize(inputWidth, 11, 4, 2, 2);

  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  alexNet->Add<MaxPooling<>>(3, 3, 2, 2, true);

  inputHeight = PoolOutSize(inputHeight, 3, 2);
  inputWidth = PoolOutSize(inputWidth, 3, 2);

  // Add Convlution Layer with inputChannels = 64,
  // output maps = 192, kernel_size = (5, 5) stride = (1, 1)
  // and padding = (2, 2).
  alexNet->Add<Convolution>(64, 192, 5, 5, 1, 1, 2, 2,
      inputWidth, inputHeight);
  // Add activation function for output.
  alexNet->Add<ReLU<>>();
  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  alexNet->Add<MaxPooling<>>(3, 3, 2, 2, true);

  // Update inputHeight and inputWidth for next layers.
  inputHeight = ConvOutSize(inputHeight, 5, 1, 2, 2);
  inputWidth = ConvOutSize(inputWidth, 5, 1, 2, 2);
  inputHeight = PoolOutSize(inputHeight, 3, 2);
  inputWidth = PoolOutSize(inputWidth, 3, 2);

  // Add Convlution Layer with input maps = 192,
  // output maps = 384, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  alexNet->Add<Convolution>(192, 384, 3, 3,
      1, 1, 1, 1, inputWidth, inputHeight);
  // Add activation function for output.
  alexNet->Add<ReLU<>>();

  // Update inputHeight and inputWidth for next layers.
  inputHeight = ConvOutSize(inputHeight, 3, 1, 1, 1);
  inputWidth = ConvOutSize(inputWidth, 3, 1, 1, 1);

  // Add Convlution Layer with input maps = 384,
  // output maps = 256, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  alexNet->Add<Convolution>(384, 256, 3, 3,
      1, 1, 1, 1, inputWidth, inputHeight);
  // Add activation function for output.
  alexNet->Add<ReLU<>>();

  // Update inputHeight and inputWidth for next layers.
  inputHeight = ConvOutSize(inputHeight, 3, 1, 1, 1);
  inputWidth = ConvOutSize(inputWidth, 3, 1, 1, 1);

  // Add Convlution Layer with input maps = 256,
  // output maps = 256, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  alexNet->Add<Convolution>(256, 256, 3, 3,
      1, 1, 1, 1, inputWidth, inputHeight);
  // Add activation function for output.
  alexNet->Add<ReLU<>>();
  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  alexNet->Add<MaxPooling<>>(3, 3, 2, 2, true);

  // Update inputHeight and inputWidth for next layers.
  inputHeight = ConvOutSize(inputHeight, 3, 1, 1, 1);
  inputWidth = ConvOutSize(inputWidth, 3, 1, 1, 1);
  inputHeight = PoolOutSize(inputHeight, 3, 2);
  inputWidth = PoolOutSize(inputWidth, 3, 2);

  if(includeTop)
  {
    
  }
}
#endif