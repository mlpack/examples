#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include "periodic_save.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

arma::Row<size_t> getLabels(arma::mat yPreds) 
{
    arma::Row<size_t> yLabels(yPreds.n_cols);
    for (arma::uword i = 0; i < yPreds.n_cols; ++i)
    {
        yLabels(i) = yPreds.col(i).index_max();
    }
    return yLabels;
}

int main() {

    // Hyperparameters for optimizer (Feel free to tweak these).
    const double RATIO = 0.1;
    constexpr int MAX_ITERATIONS = 200;
    constexpr double STEP_SIZE = 0.002;
    constexpr int BATCH_SIZE = 64;

    // Cifar 10 Dataset containing 3072 features (32 * 32) + labels
    // is loaded from CSV file.
    mat trainData;
    data::Load("./cifar-10_train.csv", trainData, true);

    // Header column is dropped.
    trainData.shed_col(0);

    // Split the dataset into training and validation sets.
    mat train, valid;
    data::Split(trainData, train, valid, RATIO);

    // Split the features and labels
    const mat trainX = train.submat(0, 0, train.n_rows - 2, train.n_cols - 1);
    const mat validX = valid.submat(0, 0, valid.n_rows - 2, valid.n_cols - 1);

    const mat trainY = train.row(train.n_rows - 1);
    const mat validY = valid.row(valid.n_rows - 1);

    // Number of iterations, should be equal to the No. of 
    // Datapoints seen times the MAX_ITERATIONS
    size_t numIterations = trainX.n_cols * MAX_ITERATIONS;

    // Create the Feed Forward Neural Network with Random weight
    // initalization and NegativeLogLikelihood Loss (NLLLoss).
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;

    // @model architecture.
    // 32 x 32 x 3 --- conv (6 feature maps of kernel size 5 x 5 with stride = 1) ---> 28 x 28 x 6
    // 28 x 28 x 6 ------------------------ Leaky ReLU ------------------------------> 28 x 28 x 6
    // 28 x 28 x 6 ------- max pooling (kernel size of 2 x 2 with stride = 2) -------> 14 x 14 x 6
    // 14 x 14 x 6 --- conv (16 feature maps of kernel size 5 x 5 and stride = 1) ---> 10 x 10 x 16
    // 10 x 10 x 16 ----------------------- Leaky ReLU ------------------------------> 10 x 10 x 16
    // 10 x 10 x 16 ------ max pooling (kernel size of 2 x 2 with stride = 2) -------> 5 x 5 x 16
    // 5 x 5 x 16  ------------------------- Linear ---------------------------------> 10

    model.Add<Convolution<>>(3, 6, 5, 5, 1, 1, 0, 0, 32, 32); 
    model.Add<LeakyReLU<>>(); 
    model.Add<MaxPooling<>>(2, 2, 2, 2, true);
    model.Add<Convolution<>>(6, 16, 5, 5, 1, 1, 0, 0, 14, 14);
    model.Add<LeakyReLU<>>();
    model.Add<MaxPooling<>>(2, 2, 2, 2, true);
    model.Add<Linear<>>(5*5*16, 120);
    model.Add<LeakyReLU<>>();
    model.Add<Linear<>>(120, 84);
    model.Add<LeakyReLU<>>();
    model.Add<Linear<>>(84, 10);
    model.Add<LogSoftMax<>>();

    cout << "Start training ..." << endl;

    ens::Adam optimizer(STEP_SIZE, BATCH_SIZE, 0.9, 0.999, 1e-8, numIterations, 1e-8, true);
    // ens::Adam optimizer(STEP_SIZE, BATCH_SIZE, numIterations, 1e-5, true, MomentumUpdate(0.9));
    model.Train(trainX,
            trainY,
            optimizer,
            ens::PrintLoss(),
            ens::ProgressBar(),
            ens::EarlyStopAtMinLoss(),
            ens::PeriodicSave<FFN<NegativeLogLikelihood<>, RandomInitialization>>(model, "./new_models/"));

    cout << "Starting evalutation on trainset ..." << endl;

    mat yPreds;

    model.Predict(trainX, yPreds);

    arma::Row<size_t> yLabels = getLabels(yPreds);

    double trainAccuracy = arma::accu(yLabels == trainY) / (double) trainY.n_elem * 100;

    cout << "Starting evalutation on validset ..." << endl;
    
    model.Predict(validX, yPreds);
    
    yLabels = getLabels(yPreds);

    double validAccuracy = arma::accu(yLabels == validY) / (double) validY.n_elem * 100;

    cout << "Accuracy: train = " << trainAccuracy << "%," 
         << "\t valid = " << validAccuracy <<"%" << endl;

    mlpack::data::Save("model.bin", "model", model, false);

    mat testData;
    mat testY;

    cout << "Starting Prediction on testset ..." << endl;

    data::Load("./cifar10_test.csv", testData, true);
    testData.shed_col(0);
    testY = testData.row(testData.n_rows - 1);
    testData.shed_row(testData.n_rows - 1);

    mat testPredProbs;
    model.Predict(testData, testPredProbs);

    arma::Row<size_t> testPreds = getLabels(testPredProbs);

    double testAccuracy = arma::accu(testPreds == testY) / (double) testY.n_elem * 100;

    cout << "Accuracy: test = " << testAccuracy << "%" << endl;

    return 0;
}
