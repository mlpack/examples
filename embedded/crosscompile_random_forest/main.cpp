/**
 * This is a super simple example using random forest. The idea is show how we
 * can cross compile this binary and use it on an embedded Linux device.
 *
 * It is up to the user to built something interesting out of this example, the
 * following is just a starting point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @author Omar Shrit
 */

#include <mlpack.hpp>

using namespace mlpack;

int main(int argc, char** argv)
{
  arma::mat dataset;
  data::Load("../../data/covertype-small.csv", dataset);

  // Labels are the last row.
  // The dataset stores labels from 1 through 7, but we need 0 through 6
  // (in mlpack labels are zero-indexed), so we subtract 1.
  arma::Row<size_t> labels = arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1)) - 1;
  dataset.shed_row(dataset.n_rows - 1);

  arma::mat trainSet, testSet;
  arma::Row<size_t> trainLabels, testLabels;

  // Split dataset randomly into training set and test set.
  data::Split(dataset, labels, trainSet, testSet, trainLabels, testLabels, 0.3
      /* Percentage of dataset to use for test set. */);

  RandomForest<> rf(trainSet, trainLabels, 7 /* Number of classes in dataset */, 10 /* 10 trees */);
  // Predict the labels of the test points.
  arma::Row<size_t> output;
  rf.Classify(testSet, output);
  // Now print the accuracy. The 'probabilities' output could also be used to
  // generate an ROC curve.
  const size_t correct = arma::accu(output == testLabels);
  std::cout << correct
            << " correct out of "
            << testLabels.n_elem << "\n"
            << 100.0 * correct / testLabels.n_elem
            << "%)." << std::endl;
}
