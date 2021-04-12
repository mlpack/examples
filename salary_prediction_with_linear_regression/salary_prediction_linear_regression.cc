/**
 * @file salary_prediction_linear_regression.cc
 *
 * A simple example usage of Linear Regression
 * applied to Salary dataset
 */
#include<mlpack/core.hpp>
#include<mlpack/core/data/split_data.hpp>
#include<mlpack/methods/linear_regression/linear_regression.hpp>
#include<cmath>

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
  std::cout<<std::setw(18)<<"Years Of Experience"<<std::setw(10)<<"Salary"<<std::endl;
  std::cout<<inputs.submat(0, 0, inputs.n_rows-1, 5).t()<<std::endl;

  // Plot the input data

  std::vector<double> x = arma::conv_to<std::vector<double>>::from(inputs.row(0));
  std::vector<double> y = arma::conv_to<std::vector<double>>::from(inputs.row(1));

  plt::scatter(x, y, 5);
  plt::show();

  // Split the data into features (X) and target (y) variables
  // Labels are the last row
  arma::Row<size_t> targets = arma::conv_to<arma::Row<size_t>>::from(inputs.row(inputs.n_rows - 1));
  // Labels are dropped from the originally loaded data to be used as features
  inputs.shed_row(inputs.n_rows - 1);

  // Split the dataset using mlpack
  arma::mat Xtrain;
  arma::mat Xtest;
  arma::Row<size_t> Ytrain;
  arma::Row<size_t> Ytest;
  data::Split(inputs, targets, Xtrain, Xtest, Ytrain, Ytest, 0.4);

  arma::rowvec y_train = arma::conv_to<arma::rowvec>::from(Ytrain);
  arma::rowvec y_test = arma::conv_to<arma::rowvec>::from(Ytest);

  // Create and Train Linear Regression model
  LinearRegression lr(Xtrain, y_train, 0.5);
  
  arma::rowvec y_preds;
  lr.Predict(Xtest, y_preds);

  std::vector<double> x_test = arma::conv_to<std::vector<double>>::from(Xtest);
  std::vector<double> y_t = arma::conv_to<std::vector<double>>::from(y_test);
  std::vector<double> y_p = arma::conv_to<std::vector<double>>::from(y_preds);

  plt::scatter(x_test, y_t, 5);
  plt::plot(x_test,y_p);
  plt::show();

  std::cout<<"Mean Absolute Error: "<<arma::mean(arma::abs(y_preds - y_test))<<std::endl;
  std::cout<<"Mean Squared Error: "<<arma::mean(arma::pow(y_preds - y_test,2))<<std::endl;
  std::cout<<"Root Mean Squared Error: "<<sqrt(arma::mean(arma::pow(y_preds - y_test,2)))<<std::endl;

  return 0;
}
