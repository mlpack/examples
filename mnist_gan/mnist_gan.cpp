#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
#include <mlpack/methods/ann/gan/gan.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::data;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace mlpack::regression;
using namespace std::placeholders;


int main()
{
    size_t dNumKernels = 32;
    size_t discriminatorPreTrain = 5;
    size_t batchSize = 5;
    size_t noiseDim = 100;
    size_t generatorUpdateStep = 1;
    size_t numSamples = 10;
    size_t cycles = 10;
    size_t numEpoches = 25;
    double stepSize = 0.0003;
    double trainRatio = 0.8;
    double eps = 1e-8;
    double tolerance = 1e-5;
    bool shuffle = true;
    double multiplier = 10;
    int datasetMaxCols = 10;

    std::cout << std::boolalpha
              << " batchSize = " << batchSize << std::endl
              << " generatorUpdateStep = " << generatorUpdateStep << std::endl
              << " noiseDim = " << noiseDim << std::endl
              << " numSamples = " << numSamples << std::endl
              << " stepSize = " << stepSize << std::endl
              << " numEpochs = " << numEpoches << std::endl
              << " shuffle = " << shuffle << std::endl;


    arma::mat mnistDataset;
    mnistDataset.load("./dataset/mnist_first250_training_4s_and_9s.arm");

    std::cout << arma::size(mnistDataset) << std::endl;

    mnistDataset = mnistDataset.cols(0, datasetMaxCols-1);
    size_t numIterations = mnistDataset.n_cols * numEpoches;
    numIterations /= batchSize;

    std::cout << "MnistDataset No. of rows: " << mnistDataset.n_rows << std::endl;

    /**
     * @brief Model Architecture:
     *
     * Discriminator:
     * 28x28x1-----------> conv (32 filters of size 5x5,
     *                     stride = 1, padding = 2)----------> 28x28x32
     * 28x28x32----------> ReLU -----------------------------> 28x28x32
     * 28x28x32----------> Mean pooling ---------------------> 14x14x32
     * 14x14x32----------> conv (64 filters of size 5x5,
     *                         stride = 1, padding = 2)------> 14x14x64
     * 14x14x64----------> ReLU -----------------------------> 14x14x64
     * 14x14x64----------> Mean pooling ---------------------> 7x7x64
     * 7x7x64------------> Linear Layer ---------------------> 1024
     * 1024--------------> ReLU -----------------------------> 1024
     * 1024 -------------> Linear ---------------------------> 1
     *
     *
     * Generator:
     * noiseDim---------> Linear ---------------------------> 3136
     * 3136 ------------> BatchNormalizaton ----------------> 3136
     * 3136 ------------> ReLu Layer -----------------------> 3136
     * 56x56x1 ---------> conv(1 filter of size 3x3,
     *                          stride = 2, padding = 1)----> 28x28x(noiseDim/2)
     * 28x28x(noiseDim/2)----> BatchNormalizaton -----------> 28x28x(noiseDim/2)
     * 28x28x(noiseDim/2)----> ReLu Layer-------------------> 28x28x(noiseDim/2)
     * 28x28x(noiseDim/2) ----> BilinearInterpolation ------> 56x56x(noiseDim/2)
     * 56x56x(noiseDim/2) -----> conv((noiseDim/2) filters
     *                               of size 3x3,stride = 2,
     *                                padding = 1)----------> 28x28x(noiseDim/4)
     * 28x28x(noiseDim/4) ----->BatchNormalization----------> 28x28x(noiseDim/4)
     * 28x28x(noiseDim/4) ------> ReLu Layer ---------------> 28x28x(noiseDim/4)
     * 28x28x(noiseDim/4) ------> BilinearInterpolation ----> 56x56x(noiseDim/4)
     * 56x56x(noiseDim/4) ------> conv((noiseDim/4) filters
     *                               of size 3x3, stride = 2,
     *                                   padding = 1)-------> 28x28x1
     * 28x28x1 ----------> tanh layer ----------------------> 28x28x1
     *
     *
     * Note: Output of a Convolution layer = [(W-K+2P)/S + 1]
     * where, W : Size of input volume
     *        K : Kernel size
     *        P : Padding
     *        S : Stride
     */


  // Creating the Discriminator network.
    FFN<SigmoidCrossEntropyError<> > discriminator;
    discriminator.Add<Convolution<> >(1, // Number of input activation maps
                                      dNumKernels, // Number of output activation maps
                                      5, // Filter width
                                      5, // Filter height
                                      1, // Stride along width
                                      1, // Stride along height
                                      2, // Padding width
                                      2, // Padding height
                                      28, // Input widht
                                      28); // Input height
    // Adding first ReLU
    discriminator.Add<ReLULayer<> >();
    // Adding mean pooling layer
    discriminator.Add<MeanPooling<> >(2, 2, 2, 2);
    // Adding second convolution layer
    discriminator.Add<Convolution<> >(dNumKernels, 2 * dNumKernels, 5, 5, 1, 1,
      2, 2, 14, 14);
    // Adding second ReLU
    discriminator.Add<ReLULayer<> >();
    // Adding second mean pooling layer
    discriminator.Add<MeanPooling<> >(2, 2, 2, 2);
    // Adding linear layer
    discriminator.Add<Linear<> >(7 * 7 * 2 * dNumKernels, 1024);
    // Adding third ReLU
    discriminator.Add<ReLULayer<> >();
    // Adding final layer
    discriminator.Add<Linear<> >(1024, 1);


    // Creating the Generator network
    FFN<SigmoidCrossEntropyError<> > generator;
    generator.Add<Linear<> >(noiseDim, 3136);
    generator.Add<BatchNorm<> >(3136);
    generator.Add<ReLULayer<> >();
    generator.Add<Convolution<> >(1, // Number of input activation maps
                                  noiseDim / 2, // Number of output activation maps
                                  3, // Filter width
                                  3, // Filter height
                                  2, // Stride along width
                                  2, // Stride along height
                                  1, // Padding width
                                  1, // Padding height
                                  56, // input width
                                  56); // input height
    // Adding first batch normalization layer
    generator.Add<BatchNorm<> >(39200);
    // Adding first ReLU
    generator.Add<ReLULayer<> >();
    // Adding a bilinear interpolation layer
    generator.Add<BilinearInterpolation<> >(28, 28, 56, 56, noiseDim / 2);
    // Adding second convolution layer
    generator.Add<Convolution<> >(noiseDim / 2, noiseDim / 4, 3, 3, 2, 2, 1, 1,
      56, 56);
    // Adding second batch normalization layer
    generator.Add<BatchNorm<> >(19600);
    // Adding second ReLU
    generator.Add<ReLULayer<> >();
    // Adding second bilinear interpolation layer
    generator.Add<BilinearInterpolation<> >(28, 28, 56, 56, noiseDim / 4);
    // Adding third convolution layer
    generator.Add<Convolution<> >(noiseDim / 4, 1, 3, 3, 2, 2, 1, 1, 56, 56);
    // Adding final tanh layer
    generator.Add<TanHLayer<> >();

    // Creating GAN.
    GaussianInitialization gaussian(0, 1);
    ens::Adam optimizer(stepSize, // Step size of optimizer.
                        batchSize, // Batch size.
                        0.9,       // Exponential decay rate for first moment estimates.
                        0.999,     // Exponential decay rate for weighted norm estimates.
                        eps,       // Value used to initialize the mean squared gradient parameter.
                        numIterations,  // iterPerCycle// Maximum number of iterations.
                        tolerance,    // Tolerance.
                        shuffle);     // Shuffle.
    std::function<double()> noiseFunction = [] () {
        return math::RandNormal(0, 1);};
    GAN<FFN<SigmoidCrossEntropyError<> >, GaussianInitialization,
      std::function<double()> > gan(generator, discriminator,
      gaussian, noiseFunction, noiseDim, batchSize, generatorUpdateStep,
      discriminatorPreTrain, multiplier);

    std::cout << "Training ... " << std::endl;

    const clock_t beginTime = clock();
    // Cycles for monitoring training progress.
    for( int i = 0; i < cycles; i++)
    {
        // Training the neural network. For first iteration, weights are random,
        // thus using current values as starting point.
        gan.Train(mnistDataset,  //trainDataset
                  optimizer,
                  ens::PrintLoss(),
                  ens::ProgressBar(),
                  ens::Report());

        optimizer.ResetPolicy() = false;
        std::cout << " Model Performance " <<
                  gan.Evaluate(gan.Parameters(), // Parameters of the network.
                               i, // Index of current input.
                               batchSize); // Batch size.
    }

    std::cout << " Time taken to train -> " << float(clock()-beginTime) / CLOCKS_PER_SEC << "seconds" << std::endl;

    // Let's save the model.
    data::Save("./saved_models/ganMnist_25epochs.bin", "ganMnist", gan);
    std::cout << "Model saved in mnist_gan/saved_models." << std::endl;
}