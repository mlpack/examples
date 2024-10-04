/**
* Predicting  California House Prices with  Linear Regression
*
* Objective
*
* To predict California Housing Prices using the most simple Linear Regression
* Model and see how it performs. To understand the modeling workflow using mlpack.
*
* Approach
*
* Here, we will try to recreate the workflow from the book mentioned above. 
* Pre-Process the data for the Ml Algorithm.
* Create new features. 
* Splitting the data.
* Training the ML model using mlpack.
* Residuals, Errors and Conclusion.
*/

#include <mlpack.hpp>

using namespace mlpack;

int main()
{
  /**
   * Dataset structure 
   *
   * This dataset is a modified version of the California Housing
   * dataset available from Luís Torgo's page (University of Porto).
   * Luís Torgo obtained it from the StatLib repository. The dataset
   * may also be downloaded from StatLib mirrors.
   *
   * Longitude : Longitude coordinate of the houses.
   * Latitude : Latitude coordinate of the houses.
   * Housing Median Age : Average lifespan of houses.
   * Total Rooms : Number of rooms in a location.
   * Total Bedrooms : Number of bedroooms in a location.
   * Population : Population in that location.
   * Median Income : Median Income of households in a location.
   * Median House Value : Median House Value in a location.
   * Ocean Proximity : Closeness to shore. 
   *
   * we need to load the dataset as an Armadillo matrix for further operations.
   * Our dataset has a total of 9 features: 8 numerical and
   * 1 categorical(ocean proximity). We need to map the
   * categorical features, as armadillo operates on numeric
   * values only.
   */
  arma::mat dataset;
  data::DatasetInfo info;
  info.Type(9) = mlpack::data::Datatype::categorical;
  // Please remove the header of the file if exist, otherwise the results will
  // not work
  data::Load("../../../data/housing.csv", dataset, info);

  arma::mat encoded_dataset; 
  // Here, we chose our pre-built encoding method "One Hot Encoding" to deal
  // with the categorical values.
  data::OneHotEncoding(dataset, encoded_dataset, info);
  // The dataset needs to be split into a training and testing set before we learn any model.
  // Labels are median_house_value which is row 8
  arma::rowvec labels =
      arma::conv_to<arma::rowvec>::from(encoded_dataset.row(8));
  encoded_dataset.shed_row(8);

  arma::mat trainSet, testSet;
  arma::rowvec trainLabels, testLabels;
  data::Split(encoded_dataset, labels, trainSet, testSet, trainLabels, testLabels,
      0.2 /* Percentage of dataset to use for test set. */);

  // Training the linear model
  /* Regression analysis is the most widely used method of prediction.
   * Linear regression is used when the dataset has a linear correlation
   * and as the name suggests, multiple linear regression has one independent
   * variable (predictor) and one or more dependent variable(response).
   */
  
  /**
   * The simple linear regression equation is represented as
   * y = $a + b_{1}x_{1} + b_{2}x_{2} + b_{3}x_{3} + ... + b_{n}x_{n}$
   * where:
   * $x_{i}$ is the ith explanatory variable,
   * y is the dependent variable,
   * $b_{i}$ is ith coefficient and a is the intercept.
   */
  
  /* To perform linear regression we'll be using the `LinearRegression`
   * class from mlpack.
   */
  LinearRegression lr(trainSet, trainLabels, 0.5);

  // The line above creates and train the model.
  // Let's create a output vector for storing the results.
  arma::rowvec output;
  lr.Predict(testSet, output);
  lr.ComputeError(trainSet, trainLabels);
  std::cout << lr.ComputeError(trainSet, trainLabels);
 
  // Let's manually check some predictions.
  std::cout << testLabels[1] << std::endl;
  std::cout << output[1] << std::endl;
  std::cout << testLabels[7] << std::endl;
  std::cout << output[7] << std::endl;
  arma::mat preds;
  preds.insert_rows(0, testLabels);
  preds.insert_rows(1, output);
  
  arma::mat diffs = preds.row(1) - preds.row(0);
  data::Save("preds.csv", preds);
  data::Save("predsDiff.csv", diffs);

  /**
   * Model Evaluation
   * Evaluation Metrics for Regression model
   * In the previous cell we have visualized our model performance by plotting
   * the best fit line. Now we will use various evaluation metrics to understand
   * how well our model has performed.
   * Mean Absolute Error (MAE) is the sum of absolute differences between actual
   * and predicted values, without considering the direction.
   * MAE = \\frac{\\sum_{i=1}^n\\lvert y_{i} - \\hat{y_{i}}\\rvert} {n}
   * Mean Squared Error (MSE) is calculated as the mean or average of the
   * squared differences between predicted and expected target values in a
   * dataset, a lower value is better
   * MSE = \\frac {1}{n} \\sum_{i=1}^n (y_{i} - \\hat{y_{i}})^2
   * Root Mean Squared Error (RMSE), Square root of MSE yields
   * root mean square error (RMSE) it indicates the spread of
   * the residual errors. It is always positive, and a lower
   * value indicates better performance.
   * RMSE = \\sqrt{\\frac {1}{n} \\sum_{i=1}^n (y_{i} - \\hat{y_{i}})^2}
   */
  std::cout << "Mean Absolute Error: " 
            << arma::mean(arma::abs(output - testLabels)) << std::endl;
  std::cout << "Mean Squared Error: "
            << arma::mean(arma::pow(output - testLabels,2)) << std::endl;
  std::cout << "Root Mean Squared Error: "
            << sqrt(arma::mean(arma::pow(output - testLabels,2))) << std::endl;
}
    