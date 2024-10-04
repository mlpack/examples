/**
 * Predicting Salary using Linear Regression.
 * Objective:
 * We have to predict the salary of an employee given how many years of experience they have.
 * Approach:
 * So in this example, we will train a Linear Regression model to learn the
 * correlation between the number of years of experience of each employee and
 * their respective salary.
 * Once the model is trained, we will be able to do some sample predictions.
*/
#include <mlpack.hpp>
#include <cmath>

using namespace mlpack;

int main()
{
  /* Dataset:
   * Salary_Data.csv has 2 columns — “Years of Experience” (feature) and “Salary”
   * (target) for 30 employees in a company.
   *
   * Please if the data set contains header, please consider removing the
   * header, before loading the dataset, otherwise Load function may not work 
   * correctly.
   */
  arma::mat inputs;
  data::Load("../../../data/Salary_Data.csv", inputs);

  // Split the data into features (X) and target (y) variables
  // targets are the last row.
  arma::Row<size_t> targets = 
      arma::conv_to<arma::Row<size_t>>::from(inputs.row(inputs.n_rows - 1));
  // Labels are dropped from the originally loaded data to be used as features.
  inputs.shed_row(inputs.n_rows - 1);

  /*
   * The dataset has to be split into a training set and a test set.
   * This can be done using the `data::Split()` api from mlpack.
   * Here the dataset has 30 observations and the `testRatio` is taken as 40%
   * of the total observations.
   * This indicates the test set should have 40% * 30 = 12 observations and
   * training test should have 18 observations respectively.
   * Split the dataset into train and test sets using mlpack.
   */
  arma::mat Xtrain, Xtest;
  arma::Row<size_t> Ytrain, Ytest;
  data::Split(inputs, targets, Xtrain, Xtest, Ytrain, Ytest, 0.4);
  
  // Convert armadillo Rows into rowvec. (Required by mlpacks'
  // LinearRegression API in this format).
  arma::rowvec yTrain = arma::conv_to<arma::rowvec>::from(Ytrain);
  arma::rowvec yTest = arma::conv_to<arma::rowvec>::from(Ytest);

  /*
   * Regression analysis is the most widely used method of prediction. Linear
   * regression is used when the dataset has a linear correlation and as the
   * name suggests, simple linear regression has one independent variable
   * (predictor) and one dependent variable(response).
   * The simple linear regression equation is represented as
   * $y = a+bx$ where $x$ is the explanatory variable, $y$ is the dependent
   * variable, $b$ is coefficient and $a$ is the intercept.
   * To perform linear regression we'll be using `LinearRegression()`
   * api from mlpack.
   */

   //Create and Train Linear Regression model.
  LinearRegression lr(Xtrain, yTrain, 0.5);

  // Make predictions for test data points.
  arma::rowvec yPreds;
  lr.Predict(Xtest, yPreds);
 
  /*
   * Evaluation Metrics for Regression model.
   * In the Previous cell we have visualized our model performance by plotting.
   * the best fit line. Now we will use various evaluation metrics to understand
   * how well our model has performed.
   * Mean Absolute Error (MAE) is the sum of absolute differences between actual
   * and predicted values, without considering the direction.
   * $$ MAE = \\frac{\\sum_{i=1}^n\\lvert y_{i} - \\hat{y_{i}}\\rvert} {n} $$
   * Mean Squared Error (MSE) is calculated as the mean or average of the
   * squared differences between predicted and expected target values in a
   * dataset, a lower value is better
   * $$ MSE = \\frac {1}{n} \\sum_{i=1}^n (y_{i} - \\hat{y_{i}})^2 $$
   * Root Mean Squared Error (RMSE), Square root of MSE yields root mean square
   * error (RMSE) it indicates the spread of the residual errors. It is always
   * positive, and a lower value indicates better performance.
   * $$ RMSE = \\sqrt{\\frac {1}{n} \\sum_{i=1}^n (y_{i} - \\hat{y_{i}})^2} $$
   */
  std::cout << "Mean Absolute Error: "
      << arma::mean(arma::abs(yPreds - yTest)) << std::endl;
  std::cout << "Mean Squared Error: "
      << arma::mean(arma::pow(yPreds - yTest,2)) << std::endl;
  std::cout << "Root Mean Squared Error: "
      << sqrt(arma::mean(arma::pow(yPreds - yTest,2))) << std::endl;
}
