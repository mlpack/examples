/**
 * Utitlity functions that is useful for solving Kaggle problems
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 * 
 * @author Eugene Freyman
 */
 
#ifndef MODELS_KAGGLE_UTILS_HPP
#define MODELS_KAGGLE_UTILS_HPP

#include <mlpack/prereqs.hpp>

/**
 * Returns labels bases on predicted probability (or log of probability)  
 * of classes.
 * @param predOut matrix contains probabilities (or log of probability) of
 * classes. Each row corresponds to a certain class, each column corresponds
 * to a data point.
 * @return a row vector of data point's classes. The classes starts from 1 to
 * the number of rows in input matrix.
 */
arma::Row<size_t> getLabels(const arma::mat& predOut) 
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
 * Returns the accuracy (percentage of correct answers).
 * @param predLabels predicted labels of data points.
 * @param realY real labels (they are double because we usually read them from
 * CSV file that contain many other double values).
 * @return percentage of correct answers.
 */
double accuracy(arma::Row<size_t> predLabels, const arma::mat& realY)
{
  // Calculating how many predicted classes are coincide with real labels.
  size_t success = 0;
  for (size_t j = 0; j < realY.n_cols; j++) {
    if (predLabels(j) == std::round(realY(j))) {
      ++success;
    }  
  }
  
  // Calculating percentage of correctly classified data points.
  return (double)success / (double)realY.n_cols * 100.0;
}

/**
 * Saves prediction into specifically formated CSV file, suitable for 
 * most Kaggle competitions.
 * @param filename the name of a file.
 * @param header the header in a CSV file.
 * @param predLabels predicted labels of data points. Classes of data points
 * are expected to start from 1. At the same time classes of data points in
 * the file are going to start from 0 (as Kaggle usually expects)
 */
void save(const std::string filename, std::string header, 
  const arma::Row<size_t>& predLabels)
{
	std::ofstream out(filename);
	out << header << std::endl;
	for (size_t j = 0; j < predLabels.n_cols; ++j)
	{
	  // j + 1 because Kaggle indexes start from 1
	  // pred - 1 because 1st class is 0, 2nd class is 1 and etc.
		out << j + 1 << "," << std::round(predLabels(j)) - 1;
    // to avoid an empty line in the end of the file
		if (j < predLabels.n_cols - 1)
		{
		  out << std::endl;
		}
	}
	out.close();
}

#endif