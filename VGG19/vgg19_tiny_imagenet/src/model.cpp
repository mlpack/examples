/**
 * A model using VGG19 on tiny-imagent dataset for classification.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @file model.cpp
 * @author Mehul Kumar Nirala
 */

#include "dataloader.hpp"
#include <ensmallen.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include "../../VGG19.hpp"

using namespace mlpack;
using namespace mlpack::ann;

using namespace arma;
using namespace std;
using namespace ens;

// Convenience typedefs
typedef FFN<NegativeLogLikelihood<>, XavierInitialization> VGGModel;

// Calculates Log Likelihood Loss over batches.
template<typename NetworkType = FFN<NegativeLogLikelihood<>, XavierInitialization>,
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
  // Dataloader object for loading tiny imagenet dataset.
	Dataloader d(200);
	d.LoadTrainData("./imagenet/src/imagenet/tiny-imagenet-200/train");
	d.LoadValData("./imagenet/src/imagenet/tiny-imagenet-200/val");

	arma::Mat<unsigned char> X, valX;
	arma::Mat<size_t> y, valY;

  // Loading the first 10000 randonly shuffled images for training.
	d.LoadImageData(X, y, true, 10000);
  inplace_trans(y, "lowmem");

  // Loading the first 1000 randonly shuffled images for validation.
	d.LoadImageData(valX, valY, false, 1000);

	// Input parameters, the dataset contains images with shape 64x64x3.
  const size_t inputWidth = 64, inputHeight = 64, inputChannel = 3;

  // Number of iteration per cycle.
  constexpr int ITERATIONS_PER_CYCLE = 10;

  // Number of cycles.
  constexpr int CYCLES = 10;

  // Step size of the optimizer.
  constexpr double STEP_SIZE = 1.2e-3;

  // Number of data points in each iteration of SGD.
  constexpr int BATCH_SIZE = 16;

  // Save/ Load model.
  constexpr bool saveModel = true;
  constexpr bool loadModel = false;

	VGG19 vggnet(inputWidth, inputHeight, inputChannel, d.numClasses, true, "max", "mnist");
  Sequential<>* vgg19 = vggnet.CompileModel();
  VGGModel model;

  if (loadModel)
  {
    std::cout << "Loading model ..." << std::endl;
    data::Load("VGG19/saved_models/vgg19_imagenet.bin", "VGG19", model);
  }
  else
  {
    model.Add<IdentityLayer<> >();
    model.Add(vgg19);
    // /*Can be used if Top is not included in the VggNet.*/
    // size_t outputShape = vgg19.GetOutputShape();
    // model.Add<Linear<> >(outputShape, numClasses);
    model.Add<LogSoftMax<> >();
  }

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
  arma::mat testX = arma::conv_to<arma::mat>::from(valX);
  arma::mat testY = arma::conv_to<arma::mat>::from(valY);
  inplace_trans(testY, "lowmem");


  std::cout << "Training dataset shape " << X.n_rows << " x " << X.n_cols << std::endl;
  std::cout << "Training label shape " << y.n_rows << " x " << y.n_cols << std::endl;

  std::cout << "Validation dataset shape " << testX.n_rows << " x " 
      << testX.n_cols << std::endl;
  std::cout << "Validation label shape " << testY.n_rows << " x " 
      << testY.n_cols << std::endl;
  for (int i = 0; i < CYCLES; i++)
  {
    // Train the CNN vgg19 If this is the first iteration, weights are
    // randomly initialized between -1 and 1. Otherwise, the values of weights
    // from the previous iteration are used.
    model.Train(arma::conv_to<arma::mat>::from(X),
    						arma::conv_to<arma::mat>::from(y),
    						optimizer);

    cout << "Epoch " << i << endl;
    // Don't reset optimizers parameters between cycles.
    optimizer.ResetPolicy() = false;

    std::cout << "Loss after cycle " << i << " -> " 
        << NLLLoss<VGGModel>(model, testX, testY, 50) << std::endl;
  }

  std::cout << "Time taken to train -> " << float(clock() - begin_time) /
      CLOCKS_PER_SEC << " seconds" << std::endl;

  if (saveModel)
  {
    data::Save("VGG19/saved_models/vgg19_imagenet.bin", "VGG19", model);
    std::cout << "Model saved in VGG19/saved_models/" << std::endl;
  }
}