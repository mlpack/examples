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
    // constexpr bool loadData = false;
    // constexpr size_t nofSamples = 10;
    // constexpr bool isBinary = false;
    // constexpr size_t batchSize = 5;
    // constexpr size_t noiseDim = 100;
    // constexpr size_t numSamples = 10;
    // // constexpr size_t latentSize =
  size_t dNumKernels = 32;
  size_t discriminatorPreTrain = 5;
  size_t batchSize = 5;
  size_t noiseDim = 100;
  size_t generatorUpdateStep = 1;
  size_t numSamples = 10;
  double stepSize = 0.0003;
  double eps = 1e-8;
  size_t numEpoches = 1;
  double tolerance = 1e-5;
  int datasetMaxCols = 10;
  bool shuffle = true;
  double multiplier = 10;
  bool loadData = false;

    arma::mat inputData, trainData, validData;
    trainData.load("./mnist_first250_training_4s_and_9s.arm");

    if(loadData)
    {

        inputData.load("./mnist_first250_training_4s_and_9s.arm");

        // Removing the headers
        inputData = inputData.submat(0, 1, inputData.n_rows - 1, inputData.n_cols - 1);
        inputData /= 255.0;

        // Removing the labels
        inputData = inputData.submat(1, 0, inputData.n_rows - 1, inputData.n_cols - 1);

        inputData = (inputData - 0.5) * 2;

        data::Split(inputData, trainData, validData, 0.8);
    }

    arma::arma_rng::set_seed_random();

    std::function<double ()> noiseFunction = [](){ return math::Random(-8, 8) +
    math::RandNormal(0, 1) * 0.01;};

    FFN<SigmoidCrossEntropyError<> > generator;

    FFN<SigmoidCrossEntropyError<> > discriminator;

    GaussianInitialization gaussian(0,1);

    GAN<FFN<SigmoidCrossEntropyError<> >, GaussianInitialization,
    std::function<double()> > gan(generator, discriminator,
    gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
    discriminatorPreTrain, multiplier);

    data::Load("./saved_models/ganMnist.bin", "ganMnist", gan);

    std::cout << "Sampling...." << std::endl;
    arma::mat noise(noiseDim, batchSize);
    size_t dim = std::sqrt(trainData.n_rows);
    arma::mat generatedData(2 * dim, dim * numSamples);


    for (size_t i = 0; i < numSamples; ++i)
    {
    arma::mat samples;
    noise.imbue( [&]() { return noiseFunction(); } );
    gan.Generator().Forward(noise, samples);
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;

    samples = trainData.col(math::RandInt(0, trainData.n_cols));
    samples.reshape(dim, dim);
    samples = samples.t();

    generatedData.submat(dim,
        i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
    }
    // arma::mat output;
    // GetSample(generatedData, output, false);
    data::Save("./samples_csv_files/ouput_mnist_2.csv", generatedData, false, false);

    std::cout << "Output generated!" << std::endl;

}