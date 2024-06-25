#include <mlpack.hpp>

using namespace mlpack;

int main()
{
  arma::mat dataset;
  data::Load("../../../data/covertype-small.csv", dataset);

  // Labels are the last row.\n",
  // The dataset stores labels from 1 through 7, but we need 0 through 6\n",
  // (in mlpack labels are zero-indexed), so we subtract 1.\n",
  arma::Row<size_t> labels = arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1)) - 1;
  dataset.shed_row(dataset.n_rows - 1);

  arma::mat trainSet, testSet;
  arma::Row<size_t> trainLabels, testLabels;

  // Split dataset randomly into training set and test set.\n",
  data::Split(dataset, labels, trainSet, testSet, trainLabels, testLabels, 0.3 
      /* Percentage of dataset to use for test set. */);

  RandomForest<> rf(trainSet, trainLabels, 7 /* Number of classes in dataset */, 10 /* 10 trees */);
  // Predict the labels of the test points.,
  arma::Row<size_t> output;
  rf.Classify(testSet, output);
  // Now print the accuracy. The 'probabilities' output could also be used to\n",
  // generate an ROC curve.\n",
  const size_t correct = arma::accu(output == testLabels);
  std::cout << correct
            << " correct out of "
            << testLabels.n_elem << "\n"
            << 100.0 * correct / testLabels.n_elem
            << "%)." << std::endl;
}
