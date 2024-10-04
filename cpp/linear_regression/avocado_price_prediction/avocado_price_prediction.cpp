/**
* Predicting Avocado's Average Price using Linear Regression
* Our target is to predict the future price of avocados depending on various
* features (Type, Region, Total Bags, ...).
* Approach
* 
* In this example, we will be using one hot encoding to encode categorical
* features. Then, we will use LinearRegression API from mlpack to learn
* the correlation between various features and the target i.e AveragePrice.
* After training the model, we will use it to do some predictions, followed by
* various evaluation metrics to quantify how well our model behaves.
*/

#include <mlpack.hpp>

using namespace mlpack;

int main()
{
  /** Dataset
   * 
   * Avocado Prices dataset has the following features:
   * PLU - Product Lookup Code in Hass avocado board.
   * Date - The date of the observation.
   * AveragePrice - Observed average price of single avocado.
   * Total Volume - Total number of avocado's sold.
   * 4046 - Total number of avocado's with PLU 4046 sold.
   * 4225 - Total number of avocado's with PLU 4225 sold.
   * 4770 - Total number of avocado's with PLU 4770 sold.
   * Total Bags = Small Bags + Large Bags + XLarge Bags.
   * Type - Conventional or organic.
   * Year - Year of observation.
   * Region - City or region of observation.
   *
   * 9 Avocado type and 11 region of observation are categorical string,
   * but armadillo matrices can contain only numeric information
   * Therefore, we explicitly define them as categorical in `datasetInfo`
   * this allows mlpack to map numeric values to each of those values,
   * which can later be unmapped to strings.
  */
  /** PLEASE, delete the header of the dataset once you have downloaded the
   * datset to your data/ directory. **/
  // Load the dataset into armadillo matrix.
  arma::mat matrix;
  data::DatasetInfo info;
  info.Type(9) = data::Datatype::categorical;
  info.Type(11) = data::Datatype::categorical;
  data::Load("../../../data/avocado.csv", matrix, info);

  arma::mat output;
  data::OneHotEncoding(matrix, output, info);
  arma::rowvec targets = arma::conv_to<arma::rowvec>::from(output.row(0));

  // Labels are dropped from the originally loaded data to be used as features.
  output.shed_row(0);
  
  // Train Test Split,
  // The dataset has to be split into a training set and a test set. Here the
  // dataset has 18249 observations and the `testRatio` is set to 20% of the
  // total observations. This indicates the test set should have
  // 20% * 18249 = 3649 observations and training test should have
  // 14600 observations respectively.
  arma::mat Xtrain, Xtest;
  arma::rowvec Ytrain, Ytest;
  data::Split(output, targets, Xtrain, Xtest, Ytrain, Ytest, 0.2);

  /* Training the linear model.
   * Regression analysis is the most widely used method of prediction.
   * Linear regression is used when the dataset has a linear correlation
   * and as the name suggests, multiple linear regression has one independent
   * variable (predictor) and one or more dependent variable(response).
   * The simple linear regression equation is represented as
   * y = $a + b_{1}x_{1} + b_{2}x_{2} + b_{3}x_{3} + ... + b_{n}x_{n}$
   * where $x_{i}$ is the ith explanatory variable, y is the dependent
   * variable, $b_{i}$ is ith coefficient and a is the intercept.
   * To perform linear regression we'll be using the `LinearRegression` class from mlpack.
   * Create and train Linear Regression model.
  */
  LinearRegression lr(Xtrain, Ytrain, 0.5);

  arma::rowvec Ypreds;
  lr.Predict(Xtest, Ypreds);

  /*
   * Model Evaluation,
   * To evaulate the model we use Mean Absolute Error (MAE) which 
   * is the sum of absolute differences between actual
   * and predicted values, without considering the direction.
   * MAE = \\frac{\\sum_{i=1}^n\\lvert y_{i} - \\hat{y_{i}}\\rvert} {n}
   * Mean Squared Error (MSE) is calculated as the mean or average of the
   * squared differences between predicted and expected target values in
   * a dataset, a lower value is better
   * MSE = \\frac {1}{n} \\sum_{i=1}^n (y_{i} - \\hat{y_{i}})^2,
   * Root Mean Squared Error (RMSE), Square root of MSE yields root mean square
   * error (RMSE) it indicates the spread of the residual errors. It is always
   * positive, and a lower value indicates better performance.
   * RMSE = \\sqrt{\\frac {1}{n} \\sum_{i=1}^n (y_{i} - \\hat{y_{i}})^2}
  */
  // Model evaluation metrics.
  // From the above metrics, we can notice that our model MAE is ~0.2,
  // which is relatively small compared to our average price of $1.405,
  // from this and the above plot we can conclude our model is a reasonably
  // good fit.
 
  std::cout << "Mean Absolute Error: "
            << arma::mean(arma::abs(Ypreds - Ytest)) << std::endl;
  std::cout << "Mean Squared Error: "
            << arma::mean(arma::pow(Ypreds - Ytest, 2)) << std::endl;
  std::cout << "Root Mean Squared Error: "
            << sqrt(arma::mean(arma::pow(Ypreds - Ytest, 2))) << std::endl;
} 
