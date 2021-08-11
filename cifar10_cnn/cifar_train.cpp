#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>

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

    const double RATIO = 0.2;
    constexpr int MAX_ITERATIONS = 0;
    constexpr double STEP_SIZE = 0.03;
    constexpr int BATCH_SIZE = 256;

    mat trainData;
    data::Load("./cifar-10_train.csv", trainData, true);

    trainData.shed_col(0);

    mat train, valid;
    data::Split(trainData, train, valid, RATIO);

    const mat trainX = train.submat(0, 0, train.n_rows - 2, train.n_cols - 1);
    const mat validX = valid.submat(0, 0, valid.n_rows - 2, valid.n_cols - 1);

    const mat trainY = train.row(train.n_rows - 1);
    const mat validY = valid.row(valid.n_rows - 1);

    FFN<NegativeLogLikelihood<>, RandomInitialization> model;

    model.Add<Convolution<>>(3, 32, 3, 3, 1, 1, 1, 1, 32, 32);
    model.Add<LeakyReLU<>>();
    model.Add<BatchNorm<>>(32);
    model.Add<Convolution<>>(32, 32, 3, 3, 1, 1, 1, 1, 32, 32);
    model.Add<LeakyReLU<>>();
    model.Add<BatchNorm<>>(32);
    model.Add<MaxPooling<>>(2, 2, 2, 2, true);
    model.Add<Dropout<>>(0.2);

    model.Add<Convolution<>>(32, 64, 3, 3, 1, 1, 1, 1, 16, 16);
    model.Add<LeakyReLU<>>();
    model.Add<BatchNorm<>>(64);
    model.Add<Convolution<>>(64, 64, 3, 3, 1, 1, 1, 1, 16, 16);
    model.Add<LeakyReLU<>>();
    model.Add<BatchNorm<>>(64);
    model.Add<MaxPooling<>>(2, 2, 2, 2, true);
    model.Add<Dropout<>>(0.3);

    model.Add<Convolution<>>(64, 128, 3, 3, 1, 1, 1, 1, 8, 8);
    model.Add<LeakyReLU<>>();
    model.Add<BatchNorm<>>(128);
    model.Add<Convolution<>>(128, 128, 3, 3, 1, 1, 1, 1, 8, 8);
    model.Add<LeakyReLU<>>();
    model.Add<BatchNorm<>>(128);
    model.Add<MaxPooling<>>(2, 2, 2, 2, true);
    model.Add<Dropout<>>(0.4);

    model.Add<Linear<>>(4*4*128, 10);
    model.Add<LogSoftMax<>>();

    cout << "Start training ..." << endl;

    ens::Adam optimizer(STEP_SIZE, BATCH_SIZE, 0.9, 0.999, 1e-6, MAX_ITERATIONS, 1e-8, true);
    model.Train(trainX,
            trainY,
            optimizer,
            ens::PrintLoss(),
            ens::ProgressBar(),
            ens::EarlyStopAtMinLoss(
                [&](const arma::mat&) 
                {
                    double validationLoss = model.Evaluate(validX, validY);
                    cout << "Validation Loss: " << validationLoss << "." << endl;
                    return validationLoss;
                }));

    mat yPreds;

    model.Predict(trainX, yPreds);

    arma::Row<size_t> yLabels = getLabels(yPreds);

    double trainAccuracy = arma::accu(yLabels == trainY) / (double) trainY.n_elem * 100;
    
    model.Predict(validX, yPreds);
    
    yLabels = getLabels(yPreds);

    double validAccuracy = arma::accu(yLabels == validY) / (double) validY.n_elem * 100;

    cout << "Accuracy: train = " << trainAccuracy << "%," 
         << "\t valid = " << validAccuracy <<"%" << endl;

    mlpack::data::Save("model.bin", "model", model, false);

    return 0;
}
