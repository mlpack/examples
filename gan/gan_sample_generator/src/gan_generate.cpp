/**
 * @file gan_generator.cpp
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

// Convenience typedefs
typedef GAN<FFN<CrossEntropyError<> >,
        GaussianInitialization,
        std::function<double()> > GanModel;

int main() {
    // The number of samples to generate.
    constexpr size_t nofSamples = 20;
    // The noise Dimension
    constexpr int noiseDim = 100;
    // The batch Size
    constexpr int batchSize = 5;


    arma::mat fullData, train, validation;

    data::Load("../../data/mnist_full.csv", fullData, true, false);
    fullData /= 255.0;
    fullData = (fullData - 0.5) * 2;
    data::Split(fullData, validation, train, 0.8);
    arma::arma_rng::set_seed_random();

    // load gan model
    FFN<> generator;
    data::Load("gan/saved_models/gan.bin", "gan", generator);

    // Generate samples
    Log::Info << "Sampling..." << std::endl;
    arma::mat noise(noiseDim, batchSize);
    std::function<double ()> noiseFunction = [](){ return math::Random(-8, 8) +
                                                          math::RandNormal(0, 1) * 0.01;};
    size_t dim = std::sqrt(validation.n_rows);
    arma::mat generatedData(2 * dim, dim * nofSamples);

    for (size_t i = 0; i < nofSamples; i++)
    {
        arma::mat samples;
        noise.imbue( [&]() { return noiseFunction(); } );

        generator.Forward(noise, samples);
        samples.reshape(dim, dim);
        samples = samples.t();

        generatedData.submat(0, i * dim, dim - 1, i * dim + dim - 1) = samples;

        samples = validation.col(math::RandInt(0, validation.n_cols));
        samples.reshape(dim, dim);
        samples = samples.t();

        generatedData.submat(dim,
                             i * dim, 2 * dim - 1, i * dim + dim - 1) = samples;
    }


    std::cout << generatedData.n_cols << " / " << generatedData.n_rows << std::endl;
    data::Save("../../gan/samples_csv_files.csv", generatedData, false, false);
}