/**
 * @file model.cpp
 * @author Mehul Kumar Nirala
 *
 * An example model using VGG19.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "../../VGG19.hpp"
#include <ensmallen.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;

using namespace arma;
using namespace std;
using namespace ens;

// Convenience typedefs
typedef FFN<NegativeLogLikelihood<>, XavierInitialization > VGGModel;

// Calculates Log Likelihood Loss over batches.
template<typename NetworkType = FFN<NegativeLogLikelihood<>, XavierInitialization >,
         typename DataType = arma::mat>
double NLLLoss(NetworkType& model, DataType& testX, DataType& testY, size_t batchSize)
{
  double loss = 0;
  size_t nofPoints = testX.n_cols;
  size_t i;

  for (i = 0; i < (size_t)(nofPoints / batchSize); i++)
  {
    loss += model.Evaluate(testX.cols(batchSize * i, batchSize * (i + 1) - 1),
        testY.cols(batchSize * i, batchSize * (i + 1) - 1));
  }

  if (nofPoints % batchSize != 0)
  {
    loss += model.Evaluate(testX.cols(batchSize * i, nofPoints - 1),
        testY.cols(batchSize * i, nofPoints - 1));
    loss /= (int)nofPoints / batchSize + 1;
  }
  else
    loss /= nofPoints / batchSize;

  return loss;
}


int main()
{
  // INPUT
  const size_t inputWidth = 28, inputHeight = 28, inputChannel = 1;
  // Dataset is randomly split into validation
  // and training parts with following ratio.
  constexpr double RATIO = 0.1;

  // Number of iteration per cycle.
  constexpr int ITERATIONS_PER_CYCLE = 10;

  // Number of cycles.
  constexpr int CYCLES = 32;

  // Step size of the optimizer.
  constexpr double STEP_SIZE = 1.2e-3;

  // Number of data points in each iteration of SGD.
  constexpr int BATCH_SIZE = 32;

  constexpr int numClasses = 10;

  constexpr bool saveModel = true;

  cout << "Reading data ..." << endl;

  // Labeled dataset that contains data for training is loaded from CSV file.
  // Rows represent features, columns represent data points.
  mat tempDataset;
  data::Load("./Kaggle/data/train.csv", tempDataset, true);
  mat dataset = tempDataset.submat(0, 1,
      tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  // Split the dataset into training and validation sets.
  mat train, valid;
  data::Split(dataset, train, valid, RATIO);

  // The train and valid datasets contain both - the features as well as the
  // class labels. Split these into separate mats.
  arma::mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
  arma::mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);
  trainX /= 255.0;
  validX /= 255.0;

  // According to NegativeLogLikelihood output layer of NN, labels should
  // specify class of a data point and be in the interval from 1 to
  // number of classes (in this case from 1 to 10).
  // Create labels for training and validatiion datasets.
  arma::mat trainY = train.row(0) + 1;
  arma::mat validY = valid.row(0) + 1;

  cout << "Input Shape" << std::endl;
  cout << trainX.n_rows << " " << trainX.n_cols << endl;

  VGG19 vggnet(inputWidth, inputHeight, inputChannel, numClasses, false, "max", "mnist");
  Sequential<>* vgg19 = vggnet.CompileModel();

  VGGModel model;
  model.Add<IdentityLayer<> >();
  model.Add(vgg19);
  
  // /*Can be used if Top is not included in the VggNet.*/
  // size_t outputShape = vgg19.GetOutputShape();
  // model.Add<Linear<> >(outputShape, numClasses);
  
  model.Add<LogSoftMax<> >();

  // Set parameters of Stochastic Gradient Descent (SGD) optimizer.
  SGD<AdamUpdate> optimizer(
    // Step size of the optimizer.
    STEP_SIZE,
    // Batch size. Number of data points that are used in each iteration.
    BATCH_SIZE,
    // Max number of iterations.
    ITERATIONS_PER_CYCLE,
    // Tolerance, used as a stopping condition. Such a small value
    // means we almost never stop by this condition, and continue gradient
    // descent until the maximum number of iterations is reached.
    1e-8,
    // Shuffle. If optimizer should take random data points from the dataset at
    // each iteration.
    true,
    // Adam update policy.
    AdamUpdate(1e-8, 0.9, 0.999));

  cout << "Training ..." << endl;
  const clock_t begin_time = clock();

  for (int i = 0; i < CYCLES; i++)
  {
    // Train the CNN vgg19 If this is the first iteration, weights are
    // randomly initialized between -1 and 1. Otherwise, the values of weights
    // from the previous iteration are used.
    model.Train(trainX, trainY, optimizer);

    cout << "Epoch " << i << endl;
    // Don't reset optimizers parameters between cycles.
    optimizer.ResetPolicy() = false;

    std::cout << "Loss after cycle " << i << " -> " <<
        NLLLoss<VGGModel>(model, validX, validY, 50) << std::endl;
  }

  std::cout << "Time taken to train -> " << float(clock() - begin_time) /
      CLOCKS_PER_SEC << " seconds" << std::endl;

  // Vggnet SaveModel("vgg19.bin");
  if (saveModel)
  {
    data::Save("VGG19/saved_models/vgg19.bin", "VGG19", model);
    std::cout << "Model saved in VGG19/saved_models/" << std::endl;
  }
}
