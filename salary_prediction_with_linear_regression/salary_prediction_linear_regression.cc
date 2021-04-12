/**
 * @file salary_prediction_linear_regression.cc
 *
 * A simple example usage of Linear Regression
 * applied to Salary dataset
 */
#include<mlpack/core.hpp>
#include<mlpack/core/data/split_data.hpp>
#include<mlpack/methods/linear_regression/linear_regression.hpp>

// Header file for visualization
#include<matplotlibcpp.h>

using namespace mlpack;
using namespace mlpack::regression;
namespace plt = matplotlibcpp;

int main() {

  // Loading data from csv into matrix
  arma::mat inputs;
  data::Load("Salary.csv", inputs);

  // Dropping first row as they represent headers
  inputs.shed_col(0);

  // Print the first 5 rows of the input data
  std::cout<<inputs.submat(0, 0, inputs.n_rows-1, 5).t()<<std::endl;

  // Plot the input data

  std::vector<double> x = arma::conv_to<std::vector<double>>::from(inputs.row(0));
  std::vector<double> y = arma::conv_to<std::vector<double>>::from(inputs.row(1));

  plt::scatter(x, y, 5);
  plt::show();

  // Split the data into features (X) and target (y) variables
  // Labels are the last row
  arma::rowvec targets = arma::conv_to<arma::rowvec>::from(inputs.row(inputs.n_rows - 1));
  // Labels are dropped from the originally loaded data to be used as features
  inputs.shed_row(inputs.n_rows - 1);

  // Split the dataset using mlpack
  //arma::mat Xtrain, Xtest;
  //arma::Row<size_t> Ytrain, Ytest;
  //data::Split(inputs, targets, Xtrain, Xtest,Ytrain, Ytest, 0.4);

  // Create and Train Linear Regression model
  LinearRegression lr(inputs, targets, 0.5);
  
  arma::rowvec y_preds;
  lr.Predict(inputs, y_preds);

  std::vector<double> y_p = arma::conv_to<std::vector<double>>::from(y_preds);

  plt::scatter(x, y, 5);
  plt::plot(x,y_p);
  plt::show();

  return 0;
}
