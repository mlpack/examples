#include<iostream>
#include <mlpack/prereqs.hpp>
#include<mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
using namespace std;
using namespace mlpack;
using namespace mlpack::tree;


int main(){

  double RATIO = 0.1;

  arma::mat dataset;
  data::Load("../data/iris_data.csv", dataset, true);

  arma::mat train, valid;
  data::Split(dataset, train, valid, RATIO);

  arma::mat trainX = train.submat(1, 0, train.n_rows - 1,
      train.n_cols - 1);
  arma::mat validX = valid.submat(1, 0, valid.n_rows - 1,
      valid.n_cols - 1);


  arma::Row<size_t> trainY , validY;

  trainY = arma::conv_to<arma::Row<size_t>>::from(
          train.row(train.n_rows-1));

  validY = arma::conv_to<arma::Row<size_t>>::from(
          valid.row(train.n_rows-1));
    const size_t numClasses = arma::max(arma::max(trainY)) + 1;
    const size_t minLeafSize = 3;
    const size_t maxDepth = 15;
    const double minimumGainSplit =-1;

    DecisionTree<> model = DecisionTree<>(  trainX, 
                                            trainY, 
                                            numClasses, 
                                            minLeafSize, 
                                            minimumGainSplit, 
                                            maxDepth); 

  arma::Row<size_t> predictions;
  arma::mat probabilities;

  model.Classify(validX, predictions, probabilities);

  size_t correct = 0;
      for (size_t i = 0; i < validY.n_cols; ++i)
        if (predictions[i] == validY[i])
          ++correct;

      // Print number of correct points.
      cout << double(correct) / double(validX.n_cols) * 100 << "% "
          << "correct on test set (" << correct << " / " << validX.n_cols
          << ")." << endl;
	
	return 0;
}

