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

#ifndef VGGNET_IMPL_HPP
#define VGGNET_IMPL_HPP

#include "VGG19.hpp"

VGG19::VGG19(const size_t inputWidth,
             const size_t inputHeight,
             const size_t inputChannel,
             const size_t numClasses,
             const bool includeTop,
             const std::string pooling,
             const std::string weights
            ):
            inputWidth(inputWidth),
            inputHeight(inputHeight),
            inputChannel(inputChannel),
            numClasses(numClasses),
            includeTop(includeTop),
            pooling(pooling),
            weights(weights)
{
    VGGNet = new Sequential<>();
}

VGG19::~VGG19()
{
  delete VGGNet;
}

Sequential<>* VGG19::CompileModel()
{
    // Block 1
    VGGNet->Add<Convolution<> >(
      inputChannel,  // Number of input activation maps.
      6,  // Number of output activation maps.
      3,  // Filter width.
      3,  // Filter height.
      1,  // Stride along width.
      1,  // Stride along height.
      1,  // Padding width.
      1,  // Padding height.
      inputWidth, // Input width.
      inputHeight  // Input height.
    );
    VGGNet->Add<LeakyReLU<> >();

    VGGNet->Add<Convolution<> >(64, 64, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();

    // Add first pooling layer. Pools over 2x2 fields in the input.
    VGGNet->Add<MaxPooling<> >(
        2,  // Width of field.
        2,  // Height of field.
        2,  // Stride along width.
        2,  // Stride along height.
        true
      );
    // valid padding 
    inputWidth = std::ceil((inputWidth - 2 + 1.0) /2 );
    inputHeight = std::ceil((inputHeight - 2 + 1.0) /2 );

    // Block 2
    VGGNet->Add<Convolution<> >(64, 128, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(128, 128, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<MaxPooling<> >(2, 2, 2, 2, true);
    // valid padding 
    inputWidth = std::ceil((inputWidth - 2 + 1.0) /2 );
    inputHeight = std::ceil((inputHeight - 2 + 1.0) /2 );

    // Block 3
    VGGNet->Add<Convolution<> >(128, 256, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(256, 256, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(256, 256, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(256, 256, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<MaxPooling<> >(2, 2, 2, 2, true);
    // valid padding 
    inputWidth = std::ceil((inputWidth - 2 + 1.0) /2 );
    inputHeight = std::ceil((inputHeight - 2 + 1.0) /2 );


    // Block 4
    VGGNet->Add<Convolution<> >(256, 512, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(512, 512, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(512, 512, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(512, 512, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<MaxPooling<> >(2, 2, 2, 2, true);
    // valid padding 
    inputWidth = std::ceil((inputWidth - 2 + 1.0) /2 );
    inputHeight = std::ceil((inputHeight - 2 + 1.0) /2 );

    // Block 5
    VGGNet->Add<Convolution<> >(512, 512, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(512, 512, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(512, 512, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<Convolution<> >(512, 512, 3, 3, 1, 1, 1, 1, inputHeight, inputWidth);
    VGGNet->Add<LeakyReLU<> >();
    VGGNet->Add<MaxPooling<> >(2, 2, 2, 2, true);
    // valid padding 
    inputWidth = std::ceil((inputWidth - 2 + 1.0) / 2);
    inputHeight = std::ceil((inputHeight - 2 + 1.0) / 2);

    if (includeTop)
    {
        // Add the final dense layer.
        VGGNet->Add<Linear<> >(inputWidth * inputHeight * 512, 4096);
        VGGNet->Add<Linear<> >(4096, 4096);
        VGGNet->Add<Linear<> >(4096, numClasses);
        outputShape = numClasses;
    }
    else
    {   
        outputShape = inputWidth * inputHeight * 512;
        if (pooling == "max")
            ; // Global maax pooling

        else if (pooling == "avg")
            ; // Global avg pooling
    }
    return VGGNet;
}

size_t VGG19::GetOutputShape()
{
    return outputShape;
}

Sequential<>* VGG19::LoadModel(std::string filePath)
{
    std::cout << "Loading model ..." << std::endl;
    data::Load(filePath, "VGG19", VGGNet);
    return VGGNet;
}

void VGG19::SaveModel(std::string filePath)
{
    std::cout << "Saving model ..." << std::endl;
    data::Save(filePath, "VGG19", VGGNet);
    std::cout << "Model saved in " << filePath << std::endl;
}

Sequential<>* VGG19::GetModel()
{
    return VGGNet;
}

#endif