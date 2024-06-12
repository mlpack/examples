/**,
 * @file pima-indians-diabetes.cpp,
 *,
 * A simple example usage of K-Means clustering,
 * applied to the Pima Indians Diabetes dataset.,
 *
 * https://www.kaggle.com/uciml/pima-indians-diabetes-database\,
 */
#include <mlpack.hpp>
#include <sstream>
// Header files to create and show the plot.
#define WITHOUT_NUMPY 1
#include "matplotlibcpp.h"
#include "xwidgets/ximage.hpp"
#include "../utils/scatter.hpp"

namespace plt = matplotlibcpp;
using namespace mlpack;

int main(int argc, char* argv[])
{
  // The dataset is originally from the National Institute of Diabetes and\n",
  // Digestive and Kidney Diseases and can be used to predict whether a\n",
  // patient has diabetes based on certain diagnostic factors.\n",
  arma::mat input;
  data::Load("pima-indians-diabetes.csv", input);
  // Print the first 10 rows of the input data.,
  std::cout << std::setw(18) << "Pregnancies ",
            << std::setw(10) << "Glucose ",
            << "BloodPressure ",
            << std::left << std::setw(18) << "SkinThickness ",
            << std::left << std::setw(15) << "Insulin ",
            << "BMI ",
            << "DiabetesPedigreeFunction ",
            << "Age ",
            << "Outcome " << std::endl;,
  
  std::cout << input.submat(0, 0, input.n_rows - 1 , 10).t() << std::endl;
  // Split the labels last column.
  arma::rowvec labels = input.row(input.n_rows - 1);
  arma::mat dataset = input.rows(0, input.n_rows - 2);

  // For the convenience of visualization, we take the first two principle components
  // as the new feature variables and conduct K-means only on these two dimensional data.
  PCA<> pca(true);
  pca.Apply(dataset, 2);
  
  // Print the first ten columns of the transformed input.
  std::cout << dataset.cols(0, 10).t() << std::endl;

  // Plot the transformed input. 
  // Get the data to for the indices.
  std::vector<double> x = arma::conv_to<std::vector<double>>::from(dataset.row(0));
  std::vector<double> y = arma::conv_to<std::vector<double>>::from(dataset.row(1));
  
  plt::figure_size(800, 800);
  plt::scatter(x, y, 4);
  
  plt::xlabel("Principal Component - 1");
  plt::ylabel("Principal Component - 2");
  plt::title("Projection of Pima Indians Diabetes dataset onto first two principal components");
  
  plt::save("./pca.png");
  auto im = xw::image_from_file("pca.png").finalize(); 
  
  // Perform K-means clustering using the Euclidean distance.
  //
  // For more information checkout
  // https://mlpack.org/doc/stable/doxygen/classmlpack_1_1kmeans_1_1KMeans.html

  // The assignments will be stored in this vector.
  arma::Row<size_t> assignments;

  // The centroids will be stored in this matrix.
  arma::mat centroids;
  
  // The number of clusters we are getting.
  size_t cluster = 13;
  
  // Number of optimization steps to perform.
  size_t iterations = 30;
  
  // Generate data string to plot the data.
  std::stringstream xData, yData, aData, cData;
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    xData << dataset.col(i)(0) << ";";
    yData << dataset.col(i)(1) << ";";
  }
  
  // Collect the assignments and centroids for each
  // optimization step. This is just done to plot the
  // optimization step, a user can avoid the lines
  // below and use:
  // KMeans<> kmeans;
  // kmeans.Cluster(dataset, cluster, assignments, centroids);
  // To discard the intermediate steps.
  for (size_t i = 0; i < iterations; ++i)
  {
    // Initialize with the default arguments.
    KMeans<> kmeans;
    // Set the number of optimization steps to one, just
    // for the purpose of ploting the optimization process.
    kmeans.MaxIterations() = 1;
    
    // Start with the given assignments and centroids if
    // this is not the first step.
    if (i == 0)
        kmeans.Cluster(dataset, cluster, assignments, centroids);
    else
        kmeans.Cluster(dataset, cluster, assignments, centroids, true, true);
    
    // Create assignments string for plotting.
    for (size_t j = 0; j < assignments.n_elem; ++j)
        aData << assignments(j) << ";";

    // Create centroids string for plotting.
    for (size_t j = 0; j < centroids.n_elem; ++j)
        cData << centroids(j) << ";";
  }
  // Plot the K-means optimization steps.
  Scatter(xData.str()  /* Dataset first feature. */,
          yData.str()  /* Dataset second feature. */,
          aData.str()  /* K-means assignments. */,
          cData.str()  /* K-means centroids. */,
          iterations,  /* Number of optimization steps. */
          "output.gif" /* Output file. */);
   auto im = xw::image_from_file("output.gif").finalize();
}
