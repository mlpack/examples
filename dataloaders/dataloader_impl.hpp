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
  typename DataSetX,
  typename DataSetY,
  typename ScalerType
>DataLoader<
    DataSetX, DataSetY, ScalerType
>::DataLoader()
{
  // Nothing to do here.
}

template<
  typename DataSetX,
  typename DataSetY,
  typename ScalerType
>DataLoader<
    DataSetX, DataSetY, ScalerType
>::DataLoader(const std::string &dataset,
              const bool shuffle,
              const double ratio,
              const bool useScaler,
              const std::vector<std::string> augmentation,
              const double augmentationProbability)
{
  if (dataset == "mnist")
  {
    LoadTrainCSV("./../data/mnist_train.csv", true, ratio, useScaler, true,
        1, -1, 0, 0);

    trainY = trainY + 1;
    validY = validY + 1;

    LoadTestCSV("./../data/mnist_test.csv", useScaler, true, 0, -1);
  }
}

template<
  typename DataSetX,
  typename DataSetY,
  typename ScalerType
>DataLoader<
    DataSetX, DataSetY, ScalerType
>::DataLoader(const std::string &dataset,
              const double ratio,
              const int rho,
              const bool useScaler,
              const size_t inputSize,
              const size_t outputSize)
{
  if (dataset == "electricity-consumption")
  {
    LoadTrainCSV("./../data/electricity-usage.csv", ratio, rho, useScaler,
        true, 1, 1);
  }
  else if(dataset )
}

template<
  typename DataSetX,
  typename DataSetY,
  typename ScalerType
> void DataLoader<
    DataSetX, DataSetY, ScalerType
>::LoadTrainCSV(const std::string &datasetPath,
                const bool shuffle,
                const double ratio,
                const bool useScaler,
                const bool dropHeader,
                const int startInputFeatures,
                const int endInputFeatures,
                const int startPredictionFeatures,
                const int endPredictionFeatures,
                const std::vector<std::string> augmentation,
                const double augmentationProbability)
{
  arma::mat dataset;
  data::Load(datasetPath, dataset, true);

  dataset = dataset.submat(0, size_t(dropHeader), dataset.n_rows - 1,
      dataset.n_cols - 1);

  arma::mat trainDataset, validDataset;
  data::Split(dataset, trainDataset, validDataset, ratio /* Add option for shuffle here.*/);

  if (useScaler)
  {
    scaler.Fit(trainDataset);
    scaler.Transform(trainDataset, trainDataset);
    scaler.Transform(validDataset, validDataset);
  }

  trainX = trainDataset.submat(wrapIndex(startInputFeatures, trainDataset.n_rows),
      0, wrapIndex(endInputFeatures, trainDataset.n_rows),
      trainDataset.n_cols - 1);

  trainY = trainDataset.submat(wrapIndex(startPredictionFeatures, trainDataset.n_rows),
      0, wrapIndex(endPredictionFeatures, trainDataset.n_rows),
      trainDataset.n_cols - 1);

  validX = validDataset.submat(wrapIndex(startInputFeatures, validDataset.n_rows),
      0, wrapIndex(endInputFeatures, validDataset.n_rows),
      validDataset.n_cols - 1);

  validY = trainDataset.submat(wrapIndex(startPredictionFeatures, validDataset.n_rows),
      0, wrapIndex(endPredictionFeatures, validDataset.n_rows),
      validDataset.n_cols - 1);

  // Add support for augmentation here.
  std::cout << "Training Dataset Loaded." << std::endl;
}

template<
  typename DataSetX,
  typename DataSetY,
  typename ScalerType
> void DataLoader<
    DataSetX, DataSetY, ScalerType
>::LoadTrainCSV(const std::string &datasetPath,
                const double ratio,
                const int rho,
                const bool useScaler,
                const bool dropHeader,
                const int startInputFeatures,
                const int endInputFeatures,
                const size_t inputSize,
                const size_t outputSize)
{
  arma::mat dataset;
  data::Load(datasetPath, dataset, true);

  this->ratio = ratio;
  arma::mat trainDataset, validDataset;
  data::Split(dataset, trainDataset, validDataset, ratio /* Add option for shuffle here.*/);

  dataset = dataset.submat(0, size_t(dropHeader), dataset.n_rows - 1,
      dataset.n_cols - 1);

  if (useScaler)
  {
    scaler.Fit(trainDataset);
    scaler.Transform(trainDataset, trainDataset);
    scaler.Transform(validDataset, validDataset);
  }

  CreateTimeSeriesData(trainDataset, trainX, trainY, rho,
      wrapIndex(startInputFeatures, trainDataset.n_rows),
      wrapIndex(endInputFeatures, trainDataset.n_rows),
      inputSize, outputSize);

  CreateTimeSeriesData(validDataset, validX, validY, rho,
      wrapIndex(startInputFeatures, validDataset.n_rows),
      wrapIndex(endInputFeatures, validDataset.n_rows),
      inputSize, outputSize);
}

template<
  typename DataSetX,
  typename DataSetY,
  typename ScalerType
> void DataLoader<
    DataSetX, DataSetY, ScalerType
>::LoadTestCSV(const std::string &datasetPath,
                const bool useScaler,
                const bool dropHeader,
                const int startInputFeatures,
                const int endInputFeatures)
{
  arma::mat dataset;
  data::Load(datasetPath, dataset, true);

  dataset = dataset.submat(0, size_t(dropHeader), dataset.n_rows - 1,
      dataset.n_cols - 1);

  if (useScaler)
  {
    scaler.Transform(dataset, dataset);
  }

  testX = dataset.submat(wrapIndex(startInputFeatures, dataset.n_rows),
      0, wrapIndex(endInputFeatures, dataset.n_rows), dataset.n_cols - 1);

  std::cout << "Testing Dataset Loaded." << std::endl;
}

template<
  typename DataSetX,
  typename DataSetY,
  typename ScalerType
> void DataLoader<
    DataSetX, DataSetY, ScalerType
>::LoadTestCSV(const std::string &datasetPath,
                const int rho,
                const bool useScaler,
                const bool dropHeader,
                const int startInputFeatures,
                const int endInputFeatures,
                const size_t inputSize,
                const size_t outputSize)
{
  arma::mat dataset;
  data::Load(datasetPath, dataset, true);

  if (useScaler)
  {
    scaler.Transform(dataset, dataset);
  }

  CreateTimeSeriesData(dataset, testX, testY, rho,
      wrapIndex(startInputFeatures, dataset.n_rows),
      wrapIndex(endInputFeatures, dataset.n_rows),
      inputSize, outputSize);
  // Add support for augmentation here.
}
#endif