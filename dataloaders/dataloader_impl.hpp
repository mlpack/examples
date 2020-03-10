/**
 * @file dataloader_impl.hpp
 * @author Eugene Freyman
 * @author Mehul Kumar Nirala.
 * @author Zoltan Somogyi
 * @author Kartik Dutt
 * 
 * Implementation of DataLoader.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_DATALOADER_IMPL_HPP
#define MODELS_DATALOADER_IMPL_HPP

#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/math/shuffle_data.hpp>
#include <mlpack/prereqs.hpp>
#include <utils/utils.hpp>
#include "dataloader.hpp"

using namespace mlpack;

template<
  typename DataSetX, typename DataSetY
>
DataLoader<
  DataSetX, DataSetY
>::DataLoader(const std::string& dataset,
              const bool shuffle,
              const double ratio,
              const std::vector<std::string> augmentation,
              const double augmentationProbability) :
              trainDatasetPath(dataset),
              ratio(ratio),
              augmentation(augmentation),
              augmentationProbability(augmentationProbability)
{
  if (trainDatasetPath == "mnist")
  {
    MNISTDataLoader();
  }

  std::cout << "Dataset Loaded." << std::endl;
}

template<
  typename DataSetX, typename DataSetY
>
DataLoader<
  DataSetX, DataSetY
>::DataLoader(const std::string& dataset,
              const double ratio,
              const size_t rho) :
              trainDatasetPath(dataset),
              ratio(ratio),
              rho(rho)
{
  if (trainDatasetPath == "google-stock-prices")
  {
    GoogleStockPricesDataloader();
  }
  else if (trainDatasetPath == "electricity-consumption")
  {
    ElectricityConsumptionDataLoader();
  }

  std::cout << "Dataset Loaded." << std::endl;
}

template<
  typename DataSetX,typename DataSetY
>
void DataLoader<
  DataSetX, DataSetY
>::MNISTDataLoader()
{
  arma::mat dataset;
  if (ratio == 0)
  {
    data::Load("./../data/mnist_train.csv", dataset, true);
    trainX = dataset.submat(1, 0, dataset.n_rows - 1, dataset.n_cols - 1) / 255.0;
    trainY = dataset.row(0) + 1;
  }
  else
  {
    data::Load("./../data/mnist_train.csv", dataset, true);
    arma::mat train, valid;
    data::Split(dataset, train, valid, ratio);

    trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1) / 255.0;
    validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1) / 255.0;

    trainY = train.row(0) + 1;
    validY = valid.row(0) + 1;
  }

  data::Load("./../data/mnist_test.csv", dataset, true);

  testX = dataset.submat(1, 0, dataset.n_rows - 1, dataset.n_cols - 1) / 255.0;
  testY = dataset.row(0) + 1;
}

template<
  typename DataSetX,typename DataSetY
>
void DataLoader<
  DataSetX, DataSetY
>::GoogleStockPricesDataloader()
{
  arma::mat dataset;
  data::Load("./../data/Google2016-2019.csv", dataset, true);
  dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

  trainCSVData = dataset.submat(arma::span(),arma::span(0, (1 - ratio) *
       dataset.n_cols));
  testCSVData = dataset.submat(arma::span(), arma::span((1 - ratio) * dataset.n_cols,
       dataset.n_cols - 1));
}

template<
  typename DataSetX,typename DataSetY
>
void DataLoader<
  DataSetX, DataSetY
>::ElectricityConsumptionDataLoader()
{
  arma::mat dataset;
  data::Load("./../data/electricity-usage.csv", dataset, true);
  dataset = dataset.submat(1, 1, 1, dataset.n_cols - 1);
  trainCSVData = dataset.submat(arma::span(),arma::span(0, (1 - ratio) *
      dataset.n_cols));
  testCSVData = dataset.submat(arma::span(), arma::span((1 - ratio) * dataset.n_cols,
      dataset.n_cols - 1));
}

#endif