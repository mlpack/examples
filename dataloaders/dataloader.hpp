/**
 * @file dataloader.hpp
 * @author Kartik Dutt
 * 
 * Definition of Dataloader for popular datasets.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <utils/utils.hpp>
#include <mlpack/prereqs.hpp>

#ifndef MODELS_DATALOADER_HPP
#define MODELS_DATALOADER_HPP

template<
  typename DataSetX = arma::mat,
  typename DataSetY = arma::mat,
  typename ScalerType = mlpack::data::MinMaxScaler
>
class DataLoader
{
 public:
  //! Create DataLoader object.
  DataLoader();

  /**
   * Constructor for DataLoader. This is used for loading popular Datasets such as
   * MNIST, ImageNet, Pascal VOC and many more.
   * 
   * @param datasetPath Path or name of dataset.
   * @param shuffle whether or not to shuffle the data.
   * @param ratio Ratio for train-test split.
   * @param augmentation Adds augmentation to training data only.
   * @param augmentationProbability Probability of applying augmentation on dataset.
   */
  DataLoader(const std::string &dataset,
             const bool shuffle,
             const double ratio = 0.75,
             const bool useScaler = false,
             const std::vector<std::string> augmentation =
                 std::vector<std::string>(),
             const double augmentationProbability = 0.2);

  /**
   * Constructor for DataLoader. This is used for loading popular Datasets such as
   * Google Stock Prices Dataset and many more.
   * 
   * @param datasetPath Path or name of dataset.
   * @param ratio Ratio for train-test split.
   * @param rho Lookback for dataset.
   */
  DataLoader(const std::string &dataset,
             const double ratio = 0.75,
             const int rho = 10,
             const bool useScaler = false,
             const size_t inputSize = 1,
             const size_t outputSize = 1);

  void LoadTrainCSV(const std::string &datasetPath,
                    const bool shuffle,
                    const double ratio = 0.75,
                    const bool useScaler = false,
                    const bool dropHeader = false,
                    const int startInputFeatures = -1,
                    const int endInputFeatures = -1,
                    const int startPredictionFeatures = -1,
                    const int endPredictionFeatures = -1,
                    const std::vector<std::string> augmentation =
                        std::vector<std::string>(),
                    const double augmentationProbability = 0.2);

  void LoadTrainCSV(const std::string &datasetPath,
                    const double ratio = 0.75,
                    const int rho = 10,
                    const bool useScaler = false,
                    const bool dropHeader = false,
                    const int startInputFeatures = -1,
                    const int endInputFeatures = -1,
                    const size_t inputSize = 1,
                    const size_t outputSize = 1);

  void LoadTestCSV(const std::string &datasetPath,
                   const bool useScaler = false,
                   const bool dropHeader = false,
                   const int startInputFeatures = -1,
                   const int endInputFeatures = -1);

  void LoadTestCSV(const std::string &datasetPath,
                   const int rho = 10,
                   const bool useScaler = false,
                   const bool dropHeader = false,
                   const int startInputFeatures = -1,
                   const int endInputFeatures = -1,
                   const size_t inputSize = 1,
                   const size_t outputSize = 1);

  //! Get the Training Dataset.
  DataSetX TrainX() const { return trainX; }

  //! Modify the Training Dataset.
  DataSetX &TrainX() { return trainX; }

  //! Get the Training Dataset.
  DataSetY TrainY() const { return trainY; }
  //! Modify the Training Dataset.
  DataSetY &TrainY() { return trainY; }

  //! Get the Test Dataset.
  DataSetX TestX() const { return testX; }
  //! Modify the Test Dataset.
  DataSetX &TestX() { return testX; }

  //! Get the Test Dataset.
  DataSetY TestY() const { return testY; }
  //! Modify the Training Dataset.
  DataSetY &TestY() { return testY; }

  //! Get the Validation Dataset.
  DataSetX ValidX() const { return validX; }
  //! Modify the Validation Dataset.
  DataSetX &ValidX() { return validX; }

  //! Get the Validation Dataset.
  DataSetY ValidY() const { return validY; }
  //! Modify the Validation Dataset.
  DataSetY &ValidY() { return validY; }

  //!Get the Scaler.
  ScalerType Scaler() const { return scaler; }
  //! Modify the Sclaer.
  ScalerType &Scaler() { return scaler; }

private:
  // Utility Function to wrap indices.
  size_t wrapIndex(int index, size_t length)
  {
    if (index < 0)
      return length - size_t(std::abs(index));

    return index;
  }

  //! Locally stored input for training.
  DataSetX trainX;
  //! Locally stored input for testing.
  DataSetX validX;
  //! Locally stored input for validation.
  DataSetX testX;

  //! Locally stored labels for training.
  DataSetY trainY;
  //! Locally stored labels for validation.
  DataSetY validY;
  //! Locally stored labels for testing.
  DataSetY testY;

  //! Locally Stored scalaer.
  ScalerType scaler;

  // MNIST Dataset Dataloader.
  void MNISTDataLoader();

  // Google Stock Prices Dataloader.
  void GoogleStockPricesDataloader();

  // Electricity Consumption DataLoader.
  void ElectricityConsumptionDataLoader();

  //! Locally stored path of dataset.
  std::string trainDatasetPath;

  //! Locally stored path of dataset.
  std::string testDatasetPath;

  //! Locally stored value of rho.
  size_t rho;

  //! Locally stored ratio for train-test split.
  double ratio;

  //! Locally stored augmentation.
  std::vector<std::string> augmentation;

  //! Locally stored augmented probability.
  double augmentationProbability;
};

#include "dataloader_impl.hpp" // Include implementation.

#endif
