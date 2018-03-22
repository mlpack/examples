/**
 * An example of using Convolution Neural Network (CNN) for 
 * solving Digit Recognizer problem from Kaggle website.
 * 
 * The full description of a problem as well as datasets for training 
 * and testing are available here https://www.kaggle.com/c/digit-recognizer
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 * 
 * The file works as basic example implementation of CNNs.
 * The file is shamelessly copied from DigitRecognizer.cpp .
 * The model has been taken from CNN test.
 * @author Eugene Freyman
 * @author Marcus Edel
 * @author Abhinav Moudgil
 * @author Sudhanshu Ranjan
 *
 */

#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/adam/adam_update.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <Kaggle/kaggle_utils.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

using namespace arma;
using namespace std;

int main()
{
  // Dataset is randomly split into training 
  // and validation parts with following ratio.
  constexpr double RATIO = 0.2;
  
  // The solution is done in several approaches (CYCLES), so each approach 
  // uses previous results as starting point and have a different optimizer 
  // options (here the step size is different).
  
  // Number of iteration per cycle. 
  constexpr int ITERATIONS_PER_CYCLE = 10000;
  
  // Number of cycles.
  constexpr int CYCLES = 100;
  
  // Step size of an optimizer.
  constexpr double STEP_SIZE = 5e-3;
  
  // Number of data points in each iteration of SGD
  constexpr int BATCH_SIZE = 50;
  
  cout << "Reading data ..." << endl;
  
  // Labeled dataset that contains data for training is loaded from CSV file,
  // rows represent features, columns represent data points.
  mat tempDataset;
  // The original file could be download from 
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("train.csv", tempDataset, true);

  std::cout<<"Data loaded : \n";
  // Originally on Kaggle dataset CSV file has header, so it's necessary to
  // get rid of the this row, in Armadillo representation it's the first column.
  mat dataset = tempDataset.submat(0, 1, 
    tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  // Splitting the dataset on training and validation parts.
  mat train, valid;
  data::Split(dataset, train, valid, RATIO);
  
  // Getting training and validating dataset with features only.
  mat trainX;
  trainX.set_size(train.n_rows - 1, train.n_cols);
  mat validX;
  validX.set_size(valid.n_rows - 1, valid.n_cols);
  for(size_t i = 0; i < train.n_cols; i++)
    trainX.col(i) = train.submat(1, i, train.n_rows -1, i), trainX.col(i) /= norm(trainX.col(i), 2);
  for(size_t i = 0; i < valid.n_cols; i++)
    validX.col(i) = valid.submat(1, i, valid.n_rows -1, i), validX.col(i) /= norm(validX.col(i), 2);

  // According to NegativeLogLikelihood output layer of NN, labels should
  // specify class of a data point and be in the interval from 1 to 
  // number of classes (in this case from 1 to 10).
  
  // Creating labels for training and validating dataset.
  const mat trainY = train.row(0) + 1;
  const mat validY = valid.row(0) + 1;

  std::cout<<"Splitting done\n";

  // CNNs are implemented as feed forward network in mlpack
  // First template parameter is output layer.
  // Second template parameter is initialization for model.
  FFN<NegativeLogLikelihood<>, RandomInitialization> model;

  // At this point you would be thinking of converting
  // arma::mat input to arma::cube but that is not needed.

  // Convolution rules are implemented using arma::mat and
  // passed from one layer to another using a 2d matrix.
  // Same can be said for pooling layers.
  
  // The convolution layers internally use memptr for reshaping
  // to arma::cube and then back to arma::mat and these implementation
  // details can be found in convolution.cpp and convolution_impl.cpp

  /*
   * Construct a convolutional neural network with a 28x28x1 input layer,
   * 24x24x8 convolution layer, 12x12x8 pooling layer, 8x8x12 convolution layer
   * and a 4x4x12 pooling layer which is fully connected with the output layer.
   * The network structure looks like:
   *
   * Input    Convolution  Pooling      Convolution  Pooling      Output
   * Layer    Layer        Layer        Layer        Layer        Layer
   *
   *          +---+        +---+        +---+        +---+
   *          | +---+      | +---+      | +---+      | +---+
   * +---+    | | +---+    | | +---+    | | +---+    | | +---+    +---+
   * |   |    | | |   |    | | |   |    | | |   |    | | |   |    |   |
   * |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> +-+ |   +--> |   |
   * |   |      +-+   |      +-+   |      +-+   |      +-+   |    |   |
   * +---+        +---+        +---+        +---+        +---+    +---+
   */

  // The convolution layer takes following parameters: 
  // inSize  The number of input maps.
  // outSize The number of output maps.
  // kW  Width of the filter/kernel.
  // kH  Height of the filter/kernel.
  // dW  Stride of filter application in the x direction.
  // dH  Stride of filter application in the y direction.
  // padW  Padding width of the input.
  // padH  Padding height of the input.
  // inputWidth  The widht of the input data.
  // inputHeight The height of the input data. 
  
  // For first layer we have inSize as 1 and outsize as 8
  // (28−5+2*0)/1+1 = 24 (W−F+2P)/S+1 
  // Refer : http://cs231n.github.io/convolutional-networks/#conv
  model.Add<Convolution<> >(1, 8, 5, 5, 1, 1, 0, 0, 28, 28);
  // Activation function (remember everythin is passed as 2d matrices).
  model.Add<ReLULayer<> >();
  
  // Maxpooling, parameters :
  // kW  Width of the pooling window.
  // kH  Height of the pooling window.
  // dW  Width of the stride operation.
  // dH  Width of the stride operation.
  // floor Rounding operator (floor or ceil). 
  // (24 - 2) / 2 + 1 = 12 
  model.Add<MaxPooling<> >(8, 8, 2, 2);
  
  // All the next layers are similarly made.
  model.Add<Convolution<> >(8, 12, 2, 2);
  model.Add<ReLULayer<> >();
  model.Add<MaxPooling<> >(2, 2, 2, 2);

  // Linear transformation (remember everythin is passed as 2d matrices).
  model.Add<Linear<> >(192, 20);
  model.Add<ReLULayer<> >();
  model.Add<Linear<> >(20, 10);

  // Use softmax for final layer.
  model.Add<LogSoftMax<> >();

  // Setting parameters Stochastic Gradient Descent (SGD) optimizer.
  SGD<AdamUpdate> optimizer(
    // Step size of the optimizer.
    STEP_SIZE,
    // Batch size. Number of data points that are used in each iteration.
    BATCH_SIZE, 
    // Max number of iterations
    ITERATIONS_PER_CYCLE,
    // Tolerance, used as a stopping condition. This small number 
    // means we never stop by this condition and continue to optimize 
    // up to reaching maximum of iterations.
    1e-8, 
    // Shuffle. If optimizer should take random data points from the dataset at
    // each iteration.
    true, 
    // Adam update policy.
    AdamUpdate(1e-8, 0.9, 0.999)); 

  std::cout<<"Model made and optimizer done.\n";
  
  // Cycles for monitoring the process of a solution.
  for (int i = 0; i <= CYCLES; i++) {
    
    // Train neural network. If this is the first iteration, weights are 
    // random, using current values as starting point otherwise.
    model.Train(trainX, trainY, optimizer); 
    
    // Don't reset optimizer's parameters between cycles.
    optimizer.ResetPolicy() = false;
    
    mat predOut;
    // Getting predictions on training data points.
    model.Predict(trainX, predOut);
    // Calculating accuracy on training data points.
    Row<size_t> predLabels = getLabels(predOut);
    double trainAccuracy = accuracy(predLabels, trainY);
    // Getting predictions on validating data points.
    model.Predict(validX, predOut);
    // Calculating accuracy on validating data points.
    predLabels = getLabels(predOut);
    double validAccuracy = accuracy(predLabels, validY);

    cout << i << " - accuracy: train = "<< trainAccuracy << "%," << 
      " valid = "<< validAccuracy << "%" <<  endl;
  }
  
  cout << "Predicting ..." << endl;  
  
  // Loading test dataset (the one whose predicted labels 
  // should be sent to Kaggle website).
  // As before, it's necessary to get rid of header.
  
  // The original file could be download from 
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("test.csv", tempDataset, true);
  mat testX = tempDataset.submat(0, 1, 
    tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  mat testPredOut;
  // Getting predictions on test data points .
  model.Predict(testX, testPredOut);
  // Generating labels for the test dataset.
  Row<size_t> testPred = getLabels(testPredOut);
  cout << "Saving predicted labels to \"results.csv\" ..." << endl;
  
  // Saving results into Kaggle compatibe CSV file.
  save("results.csv", "ImageId,Label", testPred);
  cout << "Results were saved to \"results.csv\" and could be uploaded to " 
    << "https://www.kaggle.com/c/digit-recognizer/submissions for a competition" 
    << endl;
  cout << "Finished" << endl;

}