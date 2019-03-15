//
// Created by Soonmok on 2019-03-01.
//

#ifndef MODELS_GAN_UTILS_HPP
#define MODELS_GAN_UTILS_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/gan/gan.hpp>

using namespace mlpack;
using namespace mlpack::ann;

// Calculates mean loss over batches.
template<typename NetworkType = GAN<FFN<CrossEntropyError<> >,
                  GaussianInitialization,
                  std::function<double()> >,
         typename DataType = arma::mat>

double TestLoss(NetworkType model, DataType testSet, size_t batchSize)
{
    double loss = 0;
    size_t nofPoints = testSet.n_cols;
    size_t i;

    for (i = 0; i < (size_t)nofPoints / batchSize; i++)
    {
        loss += model.Evaluate(model.Parameters(), i, batchSize);
    }

    if (nofPoints % batchSize != 0)
    {
        loss += model.Evaluate(model.Parameters(), i, batchSize);
        loss /= (int)nofPoints / batchSize + 1;
    }
    else
        loss /= nofPoints / batchSize;

    return loss;
}

#endif //MODELS_GAN_UTILS_HPP
