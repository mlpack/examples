/**
* @file  
* @author Omar Shrit
*
* Using Decision Tree for Loan Default Prediction
*
* This example is trimmed from the original one to reduce depencies 
* on a couple of functions that needs Python. If you are interested in the
* full example please refer to the Jupyter notebook directory.
*
* What is our objective ?
* To reliably predict wether a person's loan payment will be defaulted based
* on features such as Salary, Account Balance etc.
*
* Getting to know the dataset!
* LoanDefault dataset contains historic data for loan defaultees, along with their
* associated financial background, it has the following features.
* Employed - Employment status of the borrower, (1 - Employed | 0 - Unemployed).
* Bank Balance - Account Balance of the borrower at the time of repayment /
* default.
* Annual Salary - Per year income of the borrower at the time of repayment /
* default.
* Default - Target variable, indicated if the borrower repayed the loaned
* amount within the stipulated time period, (1 - Defaulted | 0 - Re-Paid).

* Approach
* This is an trivial example for dataset containing class imbalance,
* considering most of the people will be repaying their loan without default.
* So, we have to explore our data to check for imbalance, handle it using
* various techniques.
* Explore the correlation between various features in the dataset
* Split the preprocessed dataset into train and test sets respectively.
* Train a DecisionTree (Classifier) using mlpack.
* Finally we'll predict on the test set and using various evaluation metrics
* such as Accuracy, F1-Score, ROC AUC to judge the performance of our model
* on unseen data.

* In this example we'll be implementing 4 parts i.e modelling on imbalanced,
* oversampled, SMOTE & undersampled data respectively."
*/
#include <mlpack.hpp>

using namespace mlpack;
using namespace mlpack::data;

/**
 * Utility functions and definitions for evaluation metrics.
 *
 * True Positive - The actual value was true & the model predicted true.
 * False Positive - The actual value was false & the model predicted true.
 * True Negative - The actual value was false & the model predicted false.
 * False Negative - The actual value was true & the model predicted false.
 *
 * `Accuracy`: is a metric that generally describes how the model performs
 * across all classes. It is useful when all classes are of equal importance.
 * It is calculated as the ratio between the number of correct predictions to
 * the total number of predictions.
 *
 * $$Accuracy = frac{True_{positive} + True_{negative}}{True_{positive} +
 *  True_{negative} + False_{positive} + False_{negative}}$$
 * 
 * `Precision`: is calculated as the ratio between the number of positive
 * samples correctly classified to the total number of samples classified
 * as Positive. The precision measures the model's accuracy in classifying
 * a sample as positive.
 *
 * $$Precision = frac{True_{positive}}{True_{positive} + False_{positive}}$$
 *
 * `Recall`: is calulated as the ratio between the number of positive samples
 * correctly classified as Positive to the total number of Positive samples.
 * The recall measures the model's ability to detect Positive samples. The
 * higher the recall, the more positive samples detected.
 *
 * $$Recall = frac{True_{positive}}{True_{positive} + False_{negative}}$$
 *
 * The decision of whether to use precision or recall depends on the type of
 * problem begin solved. If the goal is to detect all positive samples then
 * use recall. Use precision if the problem is sensitive to classifying a
 * sample as Positive in general.

 * ROC graph has the True Positive rate on the y axis and the False Positive
 * rate on the x axis.
 * ROC Area under the curve in the graph is the primary metric to determine
 * if the classifier is doing well, the higher the value the higher the model
 * performance."
 */

double ComputeAccuracy(const arma::Row<size_t>& yPreds, const arma::Row<size_t>& yTrue)
{
  const size_t correct = arma::accu(yPreds == yTrue);
  return (double)correct / (double)yTrue.n_elem;
}

double ComputePrecision(const size_t truePos, const size_t falsePos)
{
  return (double)truePos / (double)(truePos + falsePos);
}

double ComputeRecall(const size_t truePos, const size_t falseNeg)
{
  return (double)truePos / (double)(truePos + falseNeg);
}

double ComputeF1Score(const size_t truePos, const size_t falsePos, const size_t falseNeg)
{
  double prec = ComputePrecision(truePos, falsePos);
  double rec = ComputeRecall(truePos, falseNeg);
  return 2 * (prec * rec) / (prec + rec);
}

void ClassificationReport(const arma::Row<size_t>& yPreds, const arma::Row<size_t>& yTrue)
{
  arma::Row<size_t> uniqs = arma::unique(yTrue);
  std::cout << std::setw(29) << "precision" << std::setw(15) << "recall" 
            << std::setw(15) << "f1-score" << std::setw(15) << "support" 
            << std::endl << std::endl;
  
  for (auto val: uniqs)
  {
    size_t truePos = arma::accu(yTrue == val && yPreds == val && yPreds == yTrue);
    size_t falsePos = arma::accu(yPreds == val && yPreds != yTrue);
    size_t trueNeg = arma::accu(yTrue != val && yPreds != val && yPreds == yTrue);
    size_t falseNeg = arma::accu(yPreds != val && yPreds != yTrue);
    
    std::cout << std::setw(15) << val
              << std::setw(12) << std::setprecision(2) << ComputePrecision(truePos, falsePos) 
              << std::setw(16) << std::setprecision(2) << ComputeRecall(truePos, falseNeg) 
              << std::setw(14) << std::setprecision(2) << ComputeF1Score(truePos, falsePos, falseNeg)
              << std::setw(16) << truePos
              << std::endl;
  }
}

int main() 
{
  arma::mat loanData;
  data::Load("../../../data/LoanDefault.csv", loanData);

  // Split the data into features (X) and target (y) variables, targets are the last row.
  arma::Row<size_t> targets = arma::conv_to<arma::Row<size_t>>::from(loanData.row(loanData.n_rows - 1));
  // Targets are dropped from the loaded matrix.
  loanData.shed_row(loanData.n_rows-1);

  // Train Test Split
  /** 
   * The data set has to be split into a training set and a test set. Here the dataset has 10000
   * observations and the test Ratio is taken as 25% of the total observations. This indicates
   * the test set should have 25% * 10000 = 2500 observations and trainng test should have 7500
   * observations respectively. This can be done using the `data::Split()` api from mlpack.
   **/

  // Split the dataset into train and test sets using mlpack.
  arma::mat Xtrain, Xtest;
  arma::Row<size_t> Ytrain, Ytest;
  Split(loanData, targets, Xtrain, Xtest, Ytrain, Ytest, 0.25);

  /**
   * Decision trees start with a basic question, From there you can ask a
   * series of questions to determine an answer. These questions make up
   * the decision nodes in the tree, acting as a means to split the data.
   * Each question helps an individual to arrive at a final decision, which
   * would be denoted by the leaf node. Observations that fit the criteria
   * will follow the “Yes” branch and those that don’t will follow the
   * alternate path. Decision trees seek to find the best split to subset
   * the data. To create the model we'll be using `DecisionTree<>` API from
   * mlpack.
   */
  // Create and train Decision Tree model using mlpack.
  DecisionTree<> dt(Xtrain, Ytrain, 2);
  // Classify the test set using trained model & get the probabilities.
  arma::Row<size_t> output;
  arma::mat probs;
  dt.Classify(Xtest, output, probs);


  // Save the yTest and probabilities into csv for generating ROC AUC plot.
  data::Save("./data/probabilities.csv", probs);
  data::Save("./data/ytest.csv", Ytest);
  // Model evaluation metrics.
  std::cout << "Accuracy: " << ComputeAccuracy(output, Ytest) << std::endl;
  ClassificationReport(output, Ytest);
}

