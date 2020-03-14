/**
 * @file utils.hpp
 * @author Eugene Freyman
 * @author Mehul Kumar Nirala
 * @author Zoltan Somogyi
 * @author Kartik Dutt
 * 
 * Utitlity functions that are useful in deep-learning.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_UTILS_HPP
#define MODELS_UTILS_HPP

#include <mlpack/prereqs.hpp>

using namespace mlpack;

/**
 * Returns MSE metric (average).
 * 
 * @param predictions predicted values.
 * @param groundTruth groung truth values.
 * @return MSE between ground truth and predictions.
 */
template<
typename PredType = arma::cube,
typename GroundTruthType = arma::cube
>
double MSE(const PredType predictions, const GroundTruthType groundTruth)
{
  return metric::SquaredEuclideanDistance::Evaluate(predictions, groundTruth) /
      (groundTruth.n_elem);
}

/**
 * Returns the accuracy (percentage of correct answers).
 * 
 * @param predictions predicted values.
 * @param groundTruth groung truth values.
 * @return percentage of correct answers.
 */
template<
typename PredType = arma::Row<size_t>,
typename GroundTruthType = arma::Row<size_t>
>
double Accuracy(const PredType predictions, const GroundTruthType groundTruth)
{
  // Calculating how many predicted classes are coincide with real labels.
  size_t success = 0;
  for (size_t j = 0; j < groundTruth.n_cols; j++)
      success += (predictions(j) == std::round(groundTruth(j)));

  // Calculating percentage of correctly classified data points.
  return ((double)success / (double)groundTruth.n_cols) * 100.0;
}

/**
 * Returns labels bases on predicted probability (or log of probability)
 * of classes.
 * 
 * @param predOut matrix contains probabilities (or log of probability) of
 * classes. Each row corresponds to a certain class, each column corresponds
 * to a data point.
 * @return a row vector of data point's classes. The classes starts from 1 to
 * the number of rows in input matrix.
 */
arma::Row<size_t> GetLabels(const arma::mat &predOut)
{
  arma::Row<size_t> pred(predOut.n_cols);

  // Class of a j-th data point is chosen to be the one with maximum value
  // in j-th column plus 1 (since column's elements are numbered from 0).
  for (size_t j = 0; j < predOut.n_cols; ++j)
  {
    pred(j) = arma::as_scalar(arma::find(
        arma::max(predOut.col(j)) == predOut.col(j), 1)) + 1;
  }

  return pred;
}

/**
 * Creates Time-Series Dataset for Time-Series Prediction.
 * 
 * @param dataset Time-Series Dataset.
 * @param X Input features will be stored in this variable.
 * @param Y Output will be stored in this variable.
 * @param rho Maximum rho used in LSTMs / RNNs.
 * @param inputFeatureColumnStart Starting column number for input features
 *         in dataset.
 * @param inputFeatureColumnEnd Last column number for input features in dataset.
 */
template <typename InputDataType = arma::mat,
          typename DataType = arma::cube,
          typename LabelType = arma::cube>
void CreateTimeSeriesData(InputDataType dataset,
                          DataType &X,
                          LabelType &y,
                          const size_t rho,
                          const size_t inputFeatureStart = 0,
                          const size_t inputFeatureEnd = 0,
                          const size_t inputSize = 0,
                          const size_t outputSize = 0)
{
  X.set_size(inputSize, dataset.n_cols - rho + 1, rho);
  y.set_size(outputSize, dataset.n_cols - rho + 1, rho);

  for (size_t i = 0; i < dataset.n_cols - rho; i++)
  {
    X.subcube(arma::span(), arma::span(i), arma::span()) =
        dataset.submat(arma::span(inputFeatureStart, inputFeatureEnd),
        arma::span(i, i + rho - 1));
    y.subcube(arma::span(), arma::span(i), arma::span()) =
        dataset.submat(arma::span(inputFeatureStart,
        inputFeatureEnd), arma::span(i + 1, i + rho));
  }
}

#endif
