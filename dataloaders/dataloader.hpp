/**
 * @file dataloader.hpp
 * @author Kartik Dutt
 * 
 * Definition of Dataloader to for popular datasets.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/prereqs.hpp>

#ifndef MODELS_DATALOADER_HPP
#define MODELS_DATALOADER_HPP

template<
  typename DataSetX = arma::mat,
  typename DataSetY = arma::mat
>
class DataLoader
{
 public:
  //! Create DataLoader object.
  DataLoader();

  /**
   * Constructor for DataLoader.
   * 
   * @param dataset Path or name of dataset.
   * @param ratio Ratio for train-test split.
   * @param shuffle Boolean to shuffle dataset, if true.
   * @param augmentation Adds augmentation to training data only.
   * @param augmentationProbability Probability of applying augmentation on dataset.
   */
  DataLoader(std::string dataset,
             double ratio = 0.75,
             std::vector<std::string> augmentation = std::vector<std::string>(),
             double augmentationProbability = 0.2);

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
  DataSetX ValidationX() const { return validX; }
  //! Modify the Validation Dataset.
  DataSetX &ValidationX() { return validX; }

  //! Get the Validation Dataset.
  DataSetY ValidationY() const { return validY; }
  //! Modify the Validation Dataset.
  DataSetY &ValidationY() { return validY; }

private:
  //! Locally stored input for training.
  DataSetX trainX;
  //! Locally stored input for testing.
  DataSetX validX;
  //! Locally stored input for validation.
  DataSetX testX;

  //! Locally stored labels for testing.
  DataSetY testY;
  //! Locally stored labels for training.
  DataSetY trainY;
  //! Locally stored labels for validation.
  DataSetY validY;

  // MNIST Dataset Dataloader.
  void MNISTDataLoader();

  //! Locally stored path of dataset.
  std::string datasetPath;

  //! Locally stored value of ratio for train-test split.
  double ratio;

  //! Locally stored augmentations.
  std::vector<std::string> augmentation;

  //! Locally stored augmented probability.
  double augmentationProbability;
};

#include "dataloader_impl.hpp" // Include implementation.

#endif