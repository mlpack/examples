/**
 * @file VGG16.hpp
 * @author Adithya T P (pickle-rick)
 * 
 * Implementation of VGG16 using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_VGG16_IMPL_HPP
#define MODELS_VGG16_IMPL_HPP

#include "VGG16.hpp"

VGG16::VGG16(const size_t inputWidth,
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
    vgg16 = new Sequential<>();
}

VGG16::VGG16(const std::tuple<size_t, size_t, size_t> inputShape,
             const size_t numClasses,
             const bool includeTop,
             const std::string &weights):
             inputWidth(std::get<1>(inputShape)),
             inputHeight(std::get<2>(inputShape)),
             inputChannel(std::get<0>(inputShape)),
             numClasses(numClasses),
             includeTop(includeTop),
             weights(weights),
             outputShape(512)
{
    vgg16 = new Sequential<>();
}

Sequential<>* VGG16::CompileModel()
{

    // Add VGGBlocks to make VGG16 network.
    VGGBlock(2, inputChannel, 64, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1);
    VGGBlock(2, 64, 128, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1);
    VGGBlock(3, 128, 256, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1);
    VGGBlock(3, 256, 512, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1);

    if(includeTop)
    {   
        vgg16->Add<Linear<>>(inputWidth * inputHeight * 512, 4096);
        vgg16->Add<Linear<>>(4096, 4096);
        vgg16->Add<Linear<>>(4096, numClasses);
        outputShape = numClasses;
    }

    else
    {
        // outputShape = inputWidth * inputHeight * 512;
        vgg16->Add<MaxPooling<>>(inputWidth, inputHeight, 1, 1, true);
        outputShape = 512;
    }

    return vgg16;
}

#endif