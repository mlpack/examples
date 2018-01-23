/**
 * An example of using Feed Forward Neural Network (FFN) for 
 * solving Digit Recognizer problem from Kaggle website.
 * 
 * The full description of a problem as well as datasets for training 
 * and testing are available here https://www.kaggle.com/c/digit-recognizer
 * 
 * @author Eugene Freyman
 */

#include <iostream>
#include <fstream>

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

using namespace arma;

/************************* Utility functions *************************/

// Saves prediction into specifically formated CSV file suitable for Kaggle
void save(const std::string filename, const mat& pred)
{
	std::ofstream out(filename);
	out << "ImageId,Label" << endl;
	for (int j = 0; j < pred.n_cols; ++j)
	{
	  // j + 1 because Kaggle indexes start from 1
	  // pred - 1 because 1st class is 0, 2nd class is 1 and etc.
		out << j + 1 << "," << std::round(pred(0, j)) - 1;
    // to avoid an empty line in the end of the file
		if (j < pred.n_cols - 1)
		{
		  out << endl;
		}
	}
	out.close();
}

// Returns labels bases on predicted probability of classes
// from 1 to 10
mat getLabels(const mat& predOut) 
{
  // Variable for storing predicted labels
  mat pred = arma::zeros<mat>(1, predOut.n_cols);
  
  // predOut contains 10 rows, each corresponds a class and contains
  // the log of probability of data point to belong to that class. Class of 
  // a data point is chosen to be the one with maximum of log of probability.
  for (int j = 0; j < predOut.n_cols; ++j)
  {
    pred(j) = arma::as_scalar(arma::find(
        arma::max(predOut.col(j)) == predOut.col(j), 1)) + 1;
  }
  
  return pred;
}

// Returns accuracy (percentage of correct answers).
double accuracy(const mat& predOut, const mat& realY)
{
  // Getting labels from predicted output.
  mat pred = getLabels(predOut);
  
  // Calculating how many predicted classes are coincide with real labels.
  int success = 0;
  for (int j = 0; j < realY.n_cols; j++) {
    if (std::round(pred(j)) == std::round(realY(j))) {
      ++success;
    }  
  }
  
  // Calculating percentage of correct classified data points.
  return (double)success / (double)realY.n_cols * 100.0;
}

/************************* Entry point *************************/
int main()
{
  // Dataset is randomly split into training 
  // and validation parts with following ratio.
  constexpr double RATIO = 0.9;
  // The number of neurons in the first layer.
  constexpr int H1 = 100;
  // The number of neurons in the second layer.
  constexpr int H2 = 200;
  
  // The solution is done in several approaches (CYCLES), each approach 
  // uses previous results as starting point and have a different optimizer 
  // options (here the step size is different).
  
  // Number of iteration per cycle. 
  constexpr int ITERATIONS_PER_CYCLE = 5000;
  
  // Number of cycles.
  constexpr int CYCLES = 20;
  
  // Initial step size of an optimizer.
  constexpr double STEP_BEGIN = 1e-2;
  
  // Final step size of an optimizer. Between those two points step size is
  // vary linearly.
  constexpr double STEP_END = 1e-3;
  
  std::cout << "Reading data ..." << std::endl;
  
  // Labeled dataset that contains data for training is loaded from CSV file,
  // rows represent features, columns represent data points.
  // Originally on Kaggle dataset CSV file has header, so it's necessary to
  // get rid of the this row. In Armadillo representation it's the first column.
  mat tempDataset;
  //TODO: Here you should put proper path to train.csv file, which could
  //be downloaded from https://www.kaggle.com/c/digit-recognizer/data
  data::Load("train.csv", tempDataset, true);
  mat dataset = tempDataset.submat(0, 1, 
    tempDataset.n_rows - 1, tempDataset.n_cols - 1);
  
  const int N = dataset.n_cols;
  
  // The index that splits dataset on two parts: for training and validating.
  const int SPLITTER = N * 0.8;
  
  // Generating row of indexes of features (rows of dataset from 1 to 
  // dataset.n_rows - 1).
  const Row<uword> featureIdx = regspace<Row<uword>>(1, dataset.n_rows - 1);

  // Generating shuffled row of indexes of data poits (columns of dataset from
  // 0 to dataset.n_cols - 1).
  const Row<uword> idx = shuffle(regspace<Row<uword>>(0,  N - 1));
  
  // Getting indexes of training subset of data points.
  const Row<uword> trainIdx = idx.subvec(0, SPLITTER - 1);
  // Getting indexes of validating subset of data points.
  const Row<uword> validIdx = idx.subvec(SPLITTER, idx.n_elem - 1);
  
  // Getting training dataset with features (subset of original dataset
  // with selected indexes for training).
  const mat trainX = dataset.submat(featureIdx, trainIdx);
  // Getting validating dataset with features (subset of original dataset
  // with selected indexes for validating).
  const mat validX = dataset.submat(featureIdx, validIdx);

  // According to NegativeLogLikelihood output layer of NN, labels should
  // specify class of a data point and be in the interval from 1 to 
  // number of classes (in this case from 1 to 10).
  
  // Creating labels for training.
  mat trainY(1, trainIdx.n_cols);
  for (int j = 0; j < trainIdx.n_cols; ++j)
  {
    trainY(0, j) = dataset(0, trainIdx(j)) + 1;
  }
  
  // Creating labels for validating.
  mat validY(1, validIdx.n_cols);
  for (int j = 0; j < validIdx.n_cols; ++j)
  {
    validY(0, j) = dataset(0, validIdx(j)) + 1;
  }
  
  // Specifying the NN model. NegativeLogLikelihood is the output layer that
  // is used for classification problem. RandomInitialization means that 
  // initial weights in neurons are generated randomly in the interval 
  // from -1 to 1.
  FFN<NegativeLogLikelihood<>, RandomInitialization> model;
  // This is intermediate layer that is needed for connection between input
  // data and sigmoid layer. Parameters specify the number of input features
  // and number of neurons in the next layer.
  model.Add<Linear<> >(trainX.n_rows, H1);
  // The first sigmoid layer.
  model.Add<SigmoidLayer<> >();
  // Intermediate layer between sigmoid layers.
  model.Add<Linear<> >(H1, H2);
  // The second sigmoid layer.
  model.Add<SigmoidLayer<> >();
  // Dropout layer for regularization. First parameter is the probability of
  // setting a specific value to 0.
  // model.Add<Dropout<> >(0.3, true);
  // Intermediate layer.
  model.Add<Linear<> >(H2, 10);
  // LogSoftMax layer is used together with NegativeLogLikelihood for mapping
  // output values to log of probabilities of being a specific class.
  model.Add<LogSoftMax<> >();
  
  std::cout << "Training ..." << std::endl;  
  
  // Cycles for monitoring a solution and applying different step size of 
  // the Adam optimizer.
  for (int i = 0; i <= CYCLES; i++) {
    
    // Calculating a step size as linearly distributed between specified 
    // STEP_BEGIN and STEP_END values.
    double stepRatio = (double)i / (double)CYCLES;
    double step = STEP_BEGIN + stepRatio * (STEP_END - STEP_BEGIN);
    
    // Setting parameters of Adam optimizer.
    Adam opt(step, 50, 0.9, 0.999, 1e-8, ITERATIONS_PER_CYCLE, 1e-8, true);  
    
    // Train neural network. If this is the first iteration, weights are 
    // random, using current values as starting point otherwise.
    model.Train(trainX, trainY, opt);  
    
    mat predOut;
    // Getting predictions on training data points.
    model.Predict(trainX, predOut);
    // Calculating accuracy on training data points.
    double trainAccuracy = accuracy(predOut, trainY);
    // Getting predictions on validating data points.
    model.Predict(validX, predOut);
    // Calculating accuracy on validating data points.
    double validAccuracy = accuracy(predOut, validY);

    std::cout << i << ", step = " << step << ", accuracy"
      " train = "<< trainAccuracy << "%," << 
      " valid = "<< validAccuracy << "%" <<  std::endl;
  }
  
  std::cout << "Predicting ..." << std::endl;  
  
  // Loading test dataset (the one whose predicted labels 
  // should be sent to Kaggle website).
  // As before, it's necessary to get rid of header
  
  //TODO: Here you should put proper path to test.csv file, which could
  //be downloaded from https://www.kaggle.com/c/digit-recognizer/data
  data::Load("test.csv", tempDataset, true);
  mat testX = tempDataset.submat(0, 1, 
    tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  mat testPredOut;
  // Getting predictions on test data points 
  model.Predict(testX, testPredOut);
  // Generating labels for test dataset
  mat testPred = getLabels(testPredOut);
  cout << "Saving predicted labels to \"results.csv\" ..." << endl;
  
  // TODO: result.csv could be uploaded to 
  // https://www.kaggle.com/c/digit-recognizer/submissions for competition
  save("results.csv", testPred);
 
  std::cout << "Finished" << std::endl;
}