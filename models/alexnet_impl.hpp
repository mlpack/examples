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
  
}
#endif