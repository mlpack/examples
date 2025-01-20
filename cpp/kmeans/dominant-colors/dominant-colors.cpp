/**
 * @file dominant-colors.cpp
 * @author Omar Shrit
 *
 * A simple example usage of K-means clustering
 * to find the most dominant colors in an image.
 *
 * The dominant colors are colors that are represented
 * most in the image.
 */

// Enable image load/save support.
#define HAS_STB
#include <mlpack.hpp>
#include <sstream>
// Header files to create and show images.
#include "../../../utils/stackedbar.hpp"
using namespace mlpack;

// Before we apply K-means on an image we have to be aware that the RGB color space has some shortages. In fact, it's
// tempting to simply compare the euclidean distance difference between the red, green, and blue aspects of an RGB.
// Unfortunately RGB was intended for convenient use with electronic systems, so is not very similar to average human
// perception. Applying K-means using the euclidean distance quickly reveals sporadic and often drastically different
// results than one would expect of visually similar colors.  There are several ways to tackle the issue and to calculate
// the perceived difference in color. The most popular method is known as CIE 1976, or more commonly just CIE76. This
// method uses the Euclidean distance, however, the trick is to first convert to the CIE*Lab color space.

// Function to convert RGB into CIE*Lab color space.
void rgb2lab(const double R,
             const double G,
             const double B,
             double& ls,
             double& as,
             double& bs )
{
  double varR = R / 255.0;
  double varG = G / 255.0;
  double varB = B / 255.0;

  if (varR > 0.04045)
      varR = std::pow(((varR + 0.055) / 1.055), 2.4 );
  else
      varR /= 12.92;

  if (varG > 0.04045)
      varG = std::pow(((varG + 0.055) / 1.055), 2.4);
  else
      varG /= 12.92;

  if (varB > 0.04045)
      varB = std::pow(((varB + 0.055 ) / 1.055), 2.4);
  else
      varB = varB / 12.92;

  varR *= 100.;
  varG *= 100.;
  varB *= 100.;

  double X = varR * 0.4124 + varG * 0.3576 + varB * 0.1805;
  double Y = varR * 0.2126 + varG * 0.7152 + varB * 0.0722;
  double Z = varR * 0.0193 + varG * 0.1192 + varB * 0.9505;

  double varX = X / 95.047;
  double varY = Y / 100.000;
  double varZ = Z / 108.883;

  if (varX > 0.008856)
      varX = std::pow(varX, 1.0 / 3.0);
  else
      varX = (7.787 * varX) + (16.0 / 116.0);
  
  if (varY > 0.008856)
      varY = std::pow(varY, 1.0 / 3.0);
  else
      varY = (7.787 * varY) + (16.0 / 116.0);
  
  if (varZ > 0.008856)
      varZ = std::pow(varZ, 1.0 / 3.0);
  else
      varZ = (7.787 * varZ) + (16.0 / 116.0);

  ls = (116.0 * varY) - 16.0;
  as = 500.0 * (varX - varY);
  bs = 200.0 * (varY - varZ);
}

// Function to convert CIE*Lab into RGB color space.
void lab2rgb(const double ls,
             const double as,
             const double bs,
             double& R,
             double& G,
             double& B )
{
  double varY = (ls + 16.0) / 116.0;
  double varX = as / 500.0 + varY;
  double varZ = varY - bs / 200.0;

  if (std::pow(varY, 3.0) > 0.008856)
      varY = std::pow(varY, 3.0);
  else
      varY = (varY - 16.0 / 116.0) / 7.787;
  
  if (std::pow(varX, 3.0) > 0.008856)
      varX = std::pow(varX, 3.0);
  else
      varX = (varX - 16.0 / 116.0) / 7.787;
  
  if (std::pow(varZ, 3.0) > 0.008856)
      varZ = std::pow(varZ, 3);
  else
      varZ = (varZ - 16.0 / 116.0) / 7.787;

  double X = 95.047 * varX;
  double Y = 100.000 * varY;
  double Z = 108.883 * varZ;

  varX = X / 100.0;
  varY = Y / 100.0;
  varZ = Z / 100.0;

  double varR = varX * 3.2406 + varY * -1.5372 + varZ * -0.4986;
  double varG = varX * -0.9689 + varY * 1.8758 + varZ * 0.0415;
  double varB = varX * 0.0557 + varY * -0.2040 + varZ * 1.0570;

  if (varR > 0.0031308)
      varR = 1.055 * std::pow(varR, (1.0 / 2.4)) - 0.055;
  else
      varR *= 12.92;
  
  if (varG > 0.0031308)
      varG = 1.055 * std::pow(varG, (1.0 / 2.4)) - 0.055;
  else
      varG *= 12.92;
  if (varB > 0.0031308)
      varB = 1.055 * std::pow(varB, (1.0 / 2.4)) - 0.055;
  else
      varB = 12.92 * varB;

  R = varR * 255.0;
  G = varG * 255.0;
  B = varB * 255.0;
}

// Function to convert RGB matrix into CIE*Lab color space.
void rgb2labMatrix(arma::mat& matrix)
{
  for (size_t i = 0; i < matrix.n_cols; ++i)
  {
      rgb2lab(matrix.col(i)(0),
              matrix.col(i)(1),
              matrix.col(i)(2),
              matrix.col(i)(0),
              matrix.col(i)(1),
              matrix.col(i)(2));
  }
}

// Function to convert CIE*Lab matrix into RGB color space.
void lab2rgbMatrix(arma::mat& matrix)
{
  for (size_t i = 0; i < matrix.n_cols; ++i)
  {
      lab2rgb(matrix.col(i)(0),
              matrix.col(i)(1),
              matrix.col(i)(2),
              matrix.col(i)(0),
              matrix.col(i)(1),
              matrix.col(i)(2));
  }
}

// Helper function to create the color string from the K-means centroids.
void GetColorBarData(std::string& values,
                     std::string& colors,
                     const size_t cluster,
                     const arma::Row<size_t>& assignments,
                     const arma::mat& centroids)
{
  arma::uvec h = arma::histc(arma::conv_to<arma::vec>::from(assignments), arma::linspace<arma::vec>(0, cluster - 1, cluster));
  arma::uvec indices = arma::sort_index(h);

  std::stringstream valuesString;
  std::stringstream colorsString;
  for (size_t i = 0; i < indices.n_elem; ++i)
  {
      colorsString << (int)centroids.col(indices(i))(0) << ";"
                   << (int)centroids.col(indices(i))(1) << ";"
                   << (int)centroids.col(indices(i))(2) << ";";

      valuesString << h(indices(i)) << ";";
  }
  
  values = valuesString.str();
  colors = colorsString.str();
}

void dominantColors(std::string PathToImage, std::string PathToColorBars)
{
  // Load the example image.
  arma::Mat<unsigned char> imageMatrix;
  data::ImageInfo info;
  data::Load(PathToImage, imageMatrix, info, false);
  // Print the image shape.
  std::cout << "Image info -"
            << " Width:" << info.Width()
            << " Height: " << info.Height()
            << " Channels: " << info.Channels() << std::endl;
  // Each column of the image matrix contains an image that
  // is vectorized in the format of [R, G, B, R, G, B, ..., R, G, B].
  // Here we transform the image data into the expected format:
  // [[R, G, B],
  //  [R, G, B],
  //  ...
  //  [R, G, B]]
  arma::mat imageData = arma::conv_to<arma::mat>::from(
      arma::reshape(imageMatrix, info.Channels(), imageMatrix.n_elem / 3));

  // Remove the alpha channel if the image comes with one.
  if (info.Channels() > 3)
      imageData.shed_row(3);

  // Convert from RGB to CIE*Lab color space.
  rgb2labMatrix(imageData);

  // Perform K-means clustering using the Euclidean distance.
  // The assignments will be stored in this vector.
  arma::Row<size_t> assignments;
  
  // The centroids will be stored in this matrix.
  arma::mat centroids;

  // The number of clusters we are getting (colors).
  // For the image we like the see the first 5 dominate colors.
  size_t cluster = 5;

  // Initialize with the default arguments.
  KMeans<> kmeans;
  kmeans.Cluster(imageData, cluster, assignments, centroids);

  // Convert back from CIE*Lab to RGB color space to plot the result.
  lab2rgbMatrix(centroids);

  // Create color bar data using the centroids matrix and assignments vector.
  // In our case which the centroids matrix contains the dominant colors in
  // RGB color space, and the assignments vector contains the associated
  // dominant color for each pixel in the image.
  std::string values, colors;
  GetColorBarData(values, colors, cluster, assignments, centroids);
  
  // Show the dominant colors.
  StackedBar(values, colors, PathToColorBars);
}

int main()
{
  dominantColors("../../../data/jurassic-park.png", "jurassic-park-colors.png");
  dominantColors("../../../data/the-grand-budapest-hotel.png",
      "the-grand-budapest-hotel-colors.png");
  dominantColors("../../../data/the-godfather.png", "the-godfather-colors.png");
}
