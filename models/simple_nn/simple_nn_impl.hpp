/**
 * @file simple_nn_impl.hpp
 * @author Eugene Freyman
 * @author Manthan-R-Sheth
 * @author Kartik Dutt
 * 
 * Implementation of simple neural network using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_SIMPLE_NN_IMPL_HPP
#define MODELS_SIMPLE_NN_IMPL_HPP

#include "simple_nn.hpp"
namespace mlpack {
namespace ann {

SimpleNN::SimpleNN(const size_t inputLength,
                   const size_t numClasses,
                   const std::tuple<int, int> hiddenOutSize,
                   const bool useBatchNorm,
                   const std::string &weights) :
                   inputLength(inputLength),
                   numClasses(numClasses),
                   H1(std::get<0>(hiddenOutSize)),
                   H2(std::get<1>(hiddenOutSize)),
                   useBatchNorm(useBatchNorm),
                   weights(weights)
{
  model = new Sequential<>();
  model->Add<Linear<>>(inputLength, H1);
  model->Add<LeakyReLU<>>();
  if (useBatchNorm)
    model->Add<BatchNorm<>>(H1);

  model->Add<Linear<>>(H1, H2);
  model->Add<LeakyReLU<>>();
  if (useBatchNorm)
    model->Add<BatchNorm<>>(H2);

  model->Add<Linear<>>(H2, numClasses);
  model->Add<LogSoftMax<>>();

  if (weights == "mnist")
  {
    LoadModel("./../weights/SimpleNN/");
  }
  else if (weights != "none")
  {
    LoadModel(weights);
  }
}

Sequential<>* SimpleNN::LoadModel(const std::string &filePath)
{
  std::cout << "Loading model" << std::endl;
  data::Load(filePath, "SimpleNN", model);
  return model;
}


void SimpleNN::SaveModel(const std::string &filePath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filePath, "SimpleNN", model);
  std::cout << "Model saved in " << filePath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
