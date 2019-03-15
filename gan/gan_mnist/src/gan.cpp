/**
 * @file gan.cpp
 * @author Kwon Soonmok
 *
 * A Generative Adverserial Network(GAN) model to generate
 * MNIST.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/gan/gan.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

#include <gan/gan_utils.hpp>
#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::math;
using namespace mlpack::regression;
using namespace std::placeholders;

// Convenience typedefs
typedef GAN<FFN<CrossEntropyError<> >,
        GaussianInitialization,
        std::function<double()> > GanModel;

int main() {
    // Training data is randomly taken from the dataset in this ratio.
    constexpr double trainRatio = 0.8;
    // The noise Dimension
    constexpr int noiseDim = 100;
    // The batch size.
    constexpr int batchSize = 100;
    // The number of steps to update generator
    constexpr int generatorUpdateStep = 10;
    // The number of pretrain discriminator steps
    constexpr int discriminatorPretrain = 50;
    // The multipler
    constexpr double multiplier = 10;
    // The step size of the optimizer.
    constexpr double stepSize = 0.001;
    // The epsilon value of the optimizer.
    constexpr double eps = 1e-8;
    // The tolerance value of the optimizer.
    constexpr double tolerance = 1e-5;
    // The number of interations per cycle.
    constexpr int iterPerCycle = 56000;
    // Number of cycles.
    constexpr int cycles = 10;
    // Whether to shuffle data
    constexpr bool shuffle = true;
    // Whether to convert to binary MNIST.
    constexpr bool isBinary = true;
    // Whether to save model
    constexpr bool saveModel = true;

    std::cout << "Reading data ..." << std::endl;

    // Entire dataset(without labels) is loaded from a CSV file.
    // Each column represents a data point.
    arma::mat fullData;
    data::Load("../../data/mnist_full.csv", fullData, true, false);
    std::cout << "finished reading data" << std::endl;
    fullData /= 255.0;

    if (isBinary)
    {
        fullData = arma::conv_to<arma::mat>::from(arma::randu<arma::mat>
                                                          (fullData.n_rows, fullData.n_cols) <= fullData);
    }
    else
        fullData = (fullData - 0.5) * 2;

    arma::mat train, validation;
    data::Split(fullData, validation, train, trainRatio);

    // Loss is calculated on train_test data after each cycle.
    arma::mat train_test, dump;
    data::Split(train, dump, train_test, 0.045);
    std::cout << "starting construct model" << std::endl;
    // Create the Discriminator network
    FFN<CrossEntropyError<> > discriminator;
    discriminator.Add<Linear<> >(784, 128);
    discriminator.Add<ReLULayer<> >();
    discriminator.Add<Linear<> >(128, 1);
    discriminator.Add<SigmoidLayer<> >();
    // Create the Generator network
    FFN<CrossEntropyError<> > generator;
    generator.Add<Linear<> >(noiseDim, 128);
    generator.Add<ReLULayer<> >();
    generator.Add<Linear<> >(128, 784);
    generator.Add<ReLULayer<> >();

    std::cout << "Finished adding layers" << std::endl;
    // Create GAN
    GaussianInitialization gaussian(0, 0.1);
    std::function<double ()> noiseFunction = [](){
        return math::Random(-1.0, 1.0) + math::RandNormal(0, 1) * 0.01;
    };

    std::cout << "Building up Gan model" << std::endl;
    GanModel gan(train, generator, discriminator, gaussian, noiseFunction,
                    noiseDim, batchSize, generatorUpdateStep, discriminatorPretrain,
                    multiplier);

    std::cout << "Finished initialize model" << std::endl;
    ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, eps, iterPerCycle, tolerance, shuffle);
    std::cout << "Training..." << std::endl;
    std::cout << "Initial loss -> " <<
        TestLoss<GanModel>(gan, train_test, 50) << std::endl;

    const clock_t begin_time = clock();

    // Cycles for monitoring the progress.
    for (int i = 0; i < cycles; i++)
    {
        // Train neural network. If this is the first iteration, weights are
        // random, using current values as starting point otherwise.
        gan.Train(optimizer);

        // Don't reset optimizer's parameters between cycles.
        optimizer.ResetPolicy() = false;

        std::cout << "Loss after cycle " << i << " -> " <<
            TestLoss<GanModel>(gan, train_test, 50) << std::endl;
    }

    std::cout << "Time taken to train -> " << float(clock() - begin_time) /
                                              CLOCKS_PER_SEC << " seconds" << std::endl;
    FFN<CrossEntropyError<> > trainedGenerator = gan.Generator();
    if (saveModel)
    {
        data::Save("saved_models/gan.bin", "gan", trainedGenerator);
        std::cout << "Model saved in saved_models" << std::endl;
    }

}