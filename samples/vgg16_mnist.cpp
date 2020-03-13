#include <ensmallen.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <ensmallen_bits/callbacks/callbacks.hpp>
#include <vgg/VGG16.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

int main()
{
    VGG16 vgg16(28, 28, 1, 10, true);
    Sequential<>* vgg16_net = vgg16.CompileModel();

    // Dataset is randomly split into validation
    // and training parts with following ratio.
    constexpr double RATIO = 0.1;

    // Number of iterations per cycle.
    constexpr int ITERATIONS_PER_CYCLE = 10000;

    // Number of cycles.
    constexpr int CYCLES = 40;

    // Step size of the optimizer.
    constexpr double STEP_SIZE = 1.2e-3;

    // Number of data points in each iteration of SGD.
    constexpr int BATCH_SIZE = 50;

    cout << "Reading data ..." << endl;

    // Labeled dataset that contains data for training is loaded from CSV file.
    // Rows represent features, columns represent data points.
    mat tempDataset;

    // The original file can be downloaded from
    // https://www.kaggle.com/c/digit-recognizer/data
    data::Load("../build/Kaggle/data/train.csv", tempDataset, true);

    // The original Kaggle dataset CSV file has headings for each column,
    // so it's necessary to get rid of the first row. In Armadillo representation,
    // this corresponds to the first column of our data matrix.
    mat dataset = tempDataset.submat(0, 1,
        tempDataset.n_rows - 1, tempDataset.n_cols - 1);

    // Split the dataset into training and validation sets.
    mat train, valid;
    data::Split(dataset, train, valid, RATIO);

    // The train and valid datasets contain both - the features as well as the
    // class labels. Split these into separate mats.
    const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
    const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);

    // Create labels for training and validatiion datasets.
    const mat trainY = train.row(0) + 1;
    const mat validY = valid.row(0) + 1;

    // Specify the NN model. NegativeLogLikelihood is the output layer that
    // is used for classification problem. RandomInitialization means that
    // initial weights are generated randomly in the interval from -1 to 1.
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;
    model.Add<IdentityLayer<>>();
    model.Add(vgg16_net);
    model.Add<LogSoftMax<>>();
    cout << "Training ..." << endl;

    // Set parameters of Stochastic Gradient Descent (SGD) optimizer.
    SGD<AdamUpdate> optimizer(
    // Step size of the optimizer.
    STEP_SIZE,
    // Batch size
    BATCH_SIZE,
    // Max number of iterations.
    ITERATIONS_PER_CYCLE,
    // Tolerance, used as stopping condtiion.
    1e-8,
    // Shuffle. If optimizer should take random data points from the dataset at
    // each iteration.
    true,
    // Adam update policy.
    AdamUpdate(1e-8, 0.9, 0.999));
    
    model.Train(trainX,
                trainY,
                optimizer,
                ens::PrintLoss(),
                ens::ProgressBar(),
                ens::EarlyStopAtMinLoss(1),
                ens::StoreBestCoordinates<arma::mat>());

    // Don't reset optimizers parameters between cycles.
    optimizer.ResetPolicy() = false;
    return 0;
}
