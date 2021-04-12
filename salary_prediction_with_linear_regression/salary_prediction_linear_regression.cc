/**
 * @file salary_prediction_linear_regression.cc
 *
 * A simple example usage of Linear Regression
 * applied to Salary dataset
 */
#include<mlpack/core.hpp>
#include<mlpack/methods/lars/lars.hpp>

// Header file for visualization
#include<matplotlibcpp.h>

using namespace mlpack;
using namespace mlpack::regression;
namespace plt = matplotlibcpp;

int main() {

  // Loading data from csv into matrix
  arma::mat input;
  data::Load("Salary.csv", input);

  // Dropping first row as they represent headers
  input.shed_col(0);

  // Print the first 5 rows of the input data
  //std::cout<<input.submat(0, 0, input.n_rows-1, 5).t()<<std::endl;

  // Plot the input data

  std::vector<double> x = arma::conv_to<std::vector<double>>::from(input.row(0));
  std::vector<double> y = arma::conv_to<std::vector<double>>::from(input.row(1));

  plt::scatter(x, y, 5);
  plt::show();

  arma::rowvec targets = arma::conv_to<arma::rowvec>::from(input.row(input.n_rows - 1));
  input.shed_row(input.n_rows - 1);
  return 0;
}
