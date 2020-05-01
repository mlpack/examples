#include<iostream>
#include <mlpack/prereqs.hpp>
#include<mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
using namespace std;
using namespace mlpack;
using namespace mlpack::tree;


int main(){

  // Dataset is randomly split into validation
  // and training parts in the following ratio.
  double RATIO = 0.1;
	
	// Labeled dataset that contains data for training is loaded from CSV file,
  // rows represent features, columns represent data points.
  arma::mat dataset;
  data::Load("../data/iris_data.csv", dataset, true);

  // The dataset is divided into 2 datasets, i.e train and valid
	arma::mat train, valid;
  data::Split(dataset, train, valid, RATIO);

  // trainX and validX contain indepedent variables (i.e features) 
	// on which model will be trained 
	arma::mat trainX = train.submat(1, 0, train.n_rows - 1,
      train.n_cols - 1);
  arma::mat validX = valid.submat(1, 0, valid.n_rows - 1,
      valid.n_cols - 1);


  arma::Row<size_t> trainY , validY;

	// trainY and validY contain dependent features.
  trainY = arma::conv_to<arma::Row<size_t>>::from(
          train.row(train.n_rows-1));
  validY = arma::conv_to<arma::Row<size_t>>::from(
          valid.row(train.n_rows-1));

	// Number of classes in the dataset.			
  const size_t numClasses = arma::max(arma::max(trainY)) + 1;
	// Minimum leaf size for decision trees.
  const size_t minLeafSize = 3;
	// Maximum depth of decision tree.
  const size_t maxDepth = 15;

	// Minimum gain required for split
  const double minGainSplit =-1;

	// Building the model.
  DecisionTree<> model = DecisionTree<>(    trainX,				// Independent variables
                                            trainY,				// Dependent variables
                                            numClasses, 	// number of classes
                                            minLeafSize,	// minimum leaf size
                                            minGainSplit,	// minimum gain split 
                                            maxDepth); 		// maximum depth

  // Predictions made by model.
	arma::Row<size_t> predictions;
	// probabilites for each data-point for each class.
  arma::mat probabilities;

  // classify function to predict on un-seen data.
	model.Classify(validX, predictions, probabilities);

  // calculating number of correct points.

  const size_t correct = arma::accu(predictions == validY);

  // Print number of correct points.
  cout<< double(correct) / double(validX.n_cols) * 100 << "% "
      << "correct on test set (" << correct << " / " << validX.n_cols
      << ")." << endl;
	
	return 0;
}

