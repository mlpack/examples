#include <mlpack/core.hpp>

#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/data/save.hpp>


#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
#include <mlpack/methods/ann/gan/gan.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

#include <ensmallen.hpp>

#include "gan_utils.hpp"

using namespace mlpack;
using namespace mlpack::ann;

int main()
{
  size_t discriminatorPreTrain = 5;
  size_t batchSize = 5;
  size_t noiseDim = 100;
  size_t generatorUpdateStep = 1;
  size_t numSamples = 10;
  double multiplier = 10;
  bool loadData = false;

  arma::mat trainData,inputData, validData;
  trainData.load("./mnist_first250_training_4s_and_9s.arm");

  if(loadData)
  {

    inputData.load("./mnist_first250_training_4s_and_9s.arm");

    // Removing the headers.
    inputData = inputData.submat(0, 1, inputData.n_rows - 1, inputData.n_cols - 1);
    inputData /= 255.0;

    // Removing the labels.
    inputData = inputData.submat(1, 0, inputData.n_rows - 1, inputData.n_cols - 1);

    inputData = (inputData - 0.5) * 2;

    data::Split(inputData, trainData, validData, 0.8);
  }

    arma::arma_rng::set_seed_random();

    // Define noise function.
    std::function<double ()> noiseFunction = [](){ return math::Random(-8, 8) +
    math::RandNormal(0, 1) * 0.01;};

    // Define generator.
    FFN<SigmoidCrossEntropyError<> > generator;

    // Define discriminator.
    FFN<SigmoidCrossEntropyError<> > discriminator;

    // Define GaussinaInitialization.
    GaussianInitialization gaussian(0,1);

    // Define GAN class.
    GAN<FFN<SigmoidCrossEntropyError<> >, GaussianInitialization,
    std::function<double()> > gan(generator, discriminator,
    gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
    discriminatorPreTrain, multiplier);

    // Load the saved model.
    data::Load("./saved_models/ganMnist_25epochs.bin", "ganMnist", gan);

    /*--------------Sampling-----------------------------------------*/

    std::cout << "Sampling...." << std::endl;

    // Noise matrix.
    arma::mat noise(noiseDim, batchSize);

    // Dimensions of the image.
    size_t dim = std::sqrt(trainData.n_rows);

    // Matrix to store the generated data.
    arma::mat generatedData(2 * dim, dim * numSamples);


    for (size_t i = 0; i < numSamples; ++i)
    {
    arma::mat samples;

    // Create random noise using noise function.
    noise.imbue( [&]() { return noiseFunction(); } );

    // Pass noise through generator and store output in samples.
    gan.Generator().Forward(noise, samples);

    // Reshape and Transpose the samples output.
    samples.reshape(dim, dim);
    samples = samples.t();

    // Store the output sample in a dimxdim grid in final output matrix.
    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;

    // Add the image from original train data to compare.
    samples = trainData.col(math::RandInt(0, trainData.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();
    generatedData.submat(dim,
        i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
    }
    // Save the output as csv.
    data::Save("./samples_csv_files/ouput_mnist_25_without_normalization.csv", generatedData, false, false);

    std::cout << "Output generated!" << std::endl;

}