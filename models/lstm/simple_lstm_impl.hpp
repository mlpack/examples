/**
 * @file simple_nn_impl.hpp
 * @author Mehul Kumar Nirala
 * @author Zoltan Somogyi
 * @author Kartik Dutt
 * 
 * Implementation of simple lstm using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_SIMPLE_LSTM_IMPL_HPP
#define MODELS_SIMPLE_LSTM_IMPL_HPP

#include "simple_lstm.hpp"
namespace mlpack {
namespace ann {

SimpleLSTM::SimpleLSTM(const size_t inputLength,
                       const size_t outputSize,
                       const size_t maxRho,
                       const size_t H1,
                       const std::string& weights) :
                       inputLength(inputLength),
                       maxRho(maxRho),
                       H1(H1),
                       weights(weights)
{
  model = new Sequential<>();
  model->Add<IdentityLayer<>>();
  model->Add<LSTM<>>(inputLength, H1, maxRho);
  model->Add<LeakyReLU<>>();
  model->Add<LSTM<>>(H1, H1, maxRho);
  model->Add<LeakyReLU<>>();
  model->Add<Linear<>>(H1, outputSize);

  if (weights == "google_stocks")
  {
    LoadModel("./../weights/SimpleLSTM/");
  }
  else if (weights != "none")
  {
    LoadModel(weights);
  }
}

Sequential<>* SimpleLSTM::LoadModel(const std::string &filePath)
{
  std::cout << "Loading model" << std::endl;
  data::Load(filePath, "SimpleLSTM", model);
  return model;
}


void SimpleLSTM::SaveModel(const std::string &filePath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filePath, "SimpleLSTM", model);
  std::cout << "Model saved in " << filePath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
