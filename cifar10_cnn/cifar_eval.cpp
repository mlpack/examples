#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

using namespace mlpack;
using namespace arma;
using namespace std;

// Utility function to generate class labels from probabilities.
Row<size_t> getLabels(const mat& yPreds)
{
  Row<size_t> yLabels(yPreds.n_cols);
  for (uword i = 0; i < yPreds.n_cols; ++i)
  {
    yLabels(i) = yPreds.col(i).index_max();
  }
  return yLabels;
}

int main()
{
  FFN<NegativeLogLikelihood, RandomInitialization> model;
  // Load pretrained model weights for inference.
  data::Load("cifarNet.xml", "model", model);

  cout << "Starting Prediction on testset ..." << endl;
  // Matrix for storing test feeature & labels.
  mat testData, testY;
  // Load the test data.
  data::Load("../data/cifar-10_test.csv", testData, true);
  // Drop the header column.
  testData.shed_col(0);
  // Remove labels before predicting.
  testY = testData.row(testData.n_rows - 1);
  testData.shed_row(testData.n_rows - 1);

  mat testPredProbs;
  // Get predictions on test data points.
  model.Predict(testData, testPredProbs);

  // Generate labels for the test dataset.
  arma::Row<size_t> testPreds = getLabels(testPredProbs);

  // Calculate accuracy on test dataset using the labels.
  double testAccuracy = arma::accu(testPreds == testY) /
      (double) testY.n_elem * 100;

  cout << "Accuracy: test = " << testAccuracy << "%" << endl;
}
