/**
 * @file mnist_gan.cpp
 * @author Shah Anwaar Khalid
 *
 * A Standard Generative Adverserial Network to generate
 * handwritten digits using the MNIST dataset
 *
 * mlpack is a free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have recieved a copy 
 * of the 3-clause BSD license along with mlpack.  If not, see
 * http:://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/gan/gan.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>

#include <ensmallen.hpp>

#if ((ENS_VERSION_MAJOR<2) || ((ENS_VERSION_MAJOR ==2) && (ENS_VERSION_MINOR <13)) )
#error "need ensmallen version 2.13.0 or later"
#endif

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace std::placeholders;

using namespace arma;

int main()
{
  // Hidden dimension size
  const size_t hiddenDim = 128;
  // Size of noise vector
  const size_t zDim = 64;
  // MNIST images are 28x28
  const size_t imDim = 784;
  // Number of samples to generate
  const size_t numSamples = 10;
  // No. of iterations to pretrain the Discriminator for
  const size_t discriminatorPreTrain = 300;
  // Size for minibacth
  const size_t batchSize= 128;
  // No. of steps to train Discriminator
  // before udating generator
  const size_t generatorUpdateStep = 1;
  // ratio of learning rate of Discriminator to the Generator
  const double multiplier = 1;

  // Parameters for Adam Optimizer
  const double stepSize = 0.0001;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps  = 1e-8;
  const size_t maxIterations = 0;
  const double tolerance = 1e-5; 
  const bool shuffle = true;

  std::cout << "Reading data ..."<< std::endl;
  

  // Let's load the dataset first
  // Labelled dataset that contains data for training is loaded from CSV file
  // Rows represent features, columns represent data points
  mat tempDataset;
  data::Load("./train.csv", tempDataset, true);

  // The original Kaggle datset CSV file has headings for each column,
  // so it's necessary to get rid of the first row. Since Armadillo stores
  // matrices in column major ordering, we need to get rid of the first column
  mat dataSet =
      tempDataset.submat(0, 1, tempDataset.n_rows -1, tempDataset.n_cols -1);

  // Let's split the data into features and labels
  mat trainY = dataSet.row(0) + 1;
  mat trainX = dataSet.submat(1, 0, dataSet.n_rows -1, dataSet.n_cols -1);
  trainX /= 255;

  // Create the Generator network
  FFN<SigmoidCrossEntropyError<>> Generator;
  /**
   * Model Architecture:
   *
   * Each block of Generator's neural network consists of
   * a Linear layer of dimensions ( input_size, output_size)
   * followed by a BatchNorm layer of dimensions (output_size)
   * followed by a ReLU Layer
   *
   * The final layer consists of a linear layer and a Sigmoid Layer
   */

  Generator.Add<Linear<>>(zDim, hiddenDim);
  Generator.Add<BatchNorm<>>(hiddenDim);
  Generator.Add<ReLULayer<>>();

  Generator.Add<Linear<>>(hiddenDim, hiddenDim * 2);
  Generator.Add<BatchNorm<>>(hiddenDim * 2);
  Generator.Add<ReLULayer<>>();

  Generator.Add<Linear<>>(hiddenDim * 2, hiddenDim * 4);
  Generator.Add<BatchNorm<>>(hiddenDim * 4);
  Generator.Add<ReLULayer<>>();

  Generator.Add<Linear<>>(hiddenDim * 4, hiddenDim * 8);
  Generator.Add<BatchNorm<>>(hiddenDim * 8);
  Generator.Add<ReLULayer<>>();

  Generator.Add<Linear<>>(hiddenDim * 8, imDim);
  Generator.Add<SigmoidLayer<>>();


  // Create the Discriminator network
  FFN<SigmoidCrossEntropyError<>> Discriminator;
  /**
  * Model Architecture:
  *
  * Each block of Discriminator's Neural Network consists of:
  * a Linear layer of dimensions (input_size, output_size)
  * followed by a LeakyReLU layer
  *
  * The final layer consists of a Linear layer 
  * followed by a Sigmoid Cross Entropy layer
  */

  Discriminator.Add<Linear<>>(imDim, hiddenDim * 4);
  Discriminator.Add<LeakyReLU<>>(0.2);

  Discriminator.Add<Linear<>>(hiddenDim * 4, hiddenDim * 2);
  Discriminator.Add<LeakyReLU<>>(0.2);

  Discriminator.Add<Linear<>>(hiddenDim * 2, hiddenDim);
  Discriminator.Add<LeakyReLU<>>(0.2);

  Discriminator.Add<Linear<>>(hiddenDim, 1);


  // Create GAN
  GaussianInitialization gaussian(0,0.1);
  std::function<double ()> noiseFunction = [](){return math::Random(-8,8) +
    math::RandNormal(0,1) * 0.01;};
  GAN<FFN<SigmoidCrossEntropyError<> >,
    GaussianInitialization,
    std::function<double()> >
    gan(Generator, Discriminator, gaussian, noiseFunction, zDim, batchSize,
        generatorUpdateStep, discriminatorPreTrain, multiplier);


  // Initialize the Optimizer
  ens::Adam optimizer(stepSize,      // Learning Rate
                      batchSize,     // Size of miniBatch
                      beta1,         // beta1 param for Adam
                      beta2,         // beta2 param for Adam
                      eps,           // epsilon param for Adam
                      maxIterations, // No. of epochs
                      tolerance,     // Tolerance of Adam
                      shuffle);      // Whether we want to shuffle the data  

  // Train the GAN
  gan.Train(trainX,
            optimizer,
            ens::PrintLoss(),
            ens::ProgressBar(),
            ens::EarlyStopAtMinLoss());
            // Stop the training using Early Stop at min loss

  // Let's generate samples now
  std::cout << "Generating Samples  ..." << std::endl;
  mat noise(zDim, batchSize);
  size_t dim = std::sqrt(trainX.n_rows);
  arma::mat generatedData(2 * dim, dim * numSamples);
  for (size_t i = 0; i < numSamples; i++)
  {
    arma::mat samples;
    noise.imbue( [&]() { return math::Random(0, 1); } );
    Generator.Forward(noise, samples);

    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;


    samples = trainX.col(math::RandInt(0, trainX.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(dim, i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
  }
  generatedData*= 255;
  generatedData.save("output.pgm", pgm_binary);

  return 0;

}

