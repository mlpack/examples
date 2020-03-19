#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <ensmallen_bits/callbacks/callbacks.hpp>
#include <ensmallen.hpp>
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
    constexpr int MAX_ITERATIONS = 1000;

    // Step size of the optimizer.
    constexpr double STEP_SIZE = 1.0e-4;

    // Number of data points in each iteration.
    constexpr int BATCH_SIZE = 128;

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
    // is used for classification problem. 
    FFN<NegativeLogLikelihood<>, GlorotInitialization> model;
    model.Add<IdentityLayer<>>();
    model.Add(vgg16_net);
    model.Add<LogSoftMax<>>();
    cout << "Training ..." << endl;

    Adam optimizer(STEP_SIZE, BATCH_SIZE, 0.9, 0.999, 1e-8, MAX_ITERATIONS, 1e-5, true);
    
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
