/**
 * @file lenet_impl.hpp
 * @author Eugene Freyman
 * @author Daivik Nema
 * @author Kartik Dutt
 * 
 * Implementation of LeNet using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_LENET_IMPL_HPP
#define MODELS_LENET_IMPL_HPP

#include "lenet.hpp"
namespace mlpack {
namespace ann {

LeNet::LeNet(const size_t inputChannel,
             const size_t inputWidth,
             const size_t inputHeight,
             const size_t numClasses,
             const std::string& weights,
             const int leNetVer) :
             LeNet(
                std::tuple<size_t, size_t, size_t>(
                    inputChannel,
                    inputWidth,
                    inputHeight),
                numClasses,
                weights,
                leNetVer)
{
  // Nothing to do here.
}

LeNet::LeNet(const std::tuple<size_t, size_t, size_t> inputShape,
             const size_t numClasses,
             const std::string &weights,
             const int leNetVer) :
             inputChannel(std::get<0>(inputShape)),
             inputWidth(std::get<1>(inputShape)),
             inputHeight(std::get<2>(inputShape)),
             numClasses(numClasses),
             weights(weights),
             leNetVer(leNetVer)
{
  leNet = new Sequential<>();
  ConvolutionBlock(inputChannel, 6, 5, 5, 1, 1, 2, 2);
  PoolingBlock(2, 2, 2, 2);
  ConvolutionBlock(6, 16, 5, 5, 1, 1, 2, 2);
  PoolingBlock(2, 2, 2, 2);
  // Add linear layer for LeNet.
  if (leNetVer == 1)
  {
    leNet->Add<Linear<>>(16 * inputWidth * inputHeight, numClasses);
  }
  else if (leNetVer == 4)
  {
    leNet->Add<Linear<>>(16 * inputWidth * inputHeight, 120);
    leNet->Add<LeakyReLU<>>();
    leNet->Add<Linear<>>(120, numClasses);
  }
  else if (leNetVer == 5)
  {
    leNet->Add<Linear<>>(16 * inputWidth * inputHeight, 120);
    leNet->Add<LeakyReLU<>>();
    leNet->Add<Linear<>>(120, 84);
    leNet->Add<LeakyReLU<>>();
    leNet->Add<Linear<>>(84, 10);
  }

  leNet->Add<LogSoftMax<>>();

  if (weights == "mnist")
  {
    LoadModel("./../weights/lenet/lenet_mnist.bin");
  }
  else if (weights != "none")
  {
    LoadModel(weights);
  }
}

Sequential<> *LeNet::LoadModel(const std::string &filePath)
{
  std::cout << "Loading model" << std::endl;
  data::Load(filePath, "LeNet", leNet);
  return leNet;
}

void LeNet::SaveModel(const std::string &filePath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filePath, "LeNet", leNet);
  std::cout << "Model saved in " << filePath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
