#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <dataloaders/dataloader.hpp>
#include <utils/utils.hpp>
#include <models/lenet/lenet.hpp>
#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;


int main()
{
    // Number of epochs.
    const int EPOCHS = 5;

    // Step size of the optimizer.
    const double STEP_SIZE = 1.2e-3;

    // Number of data points in each iteration of SGD.
    const int BATCH_SIZE = 50;

    // Ratio for train-validation split.
    const double RATIO = 0.2;

    DataLoader<arma::mat, arma::mat> dataloader("mnist", RATIO);

    // NegativeLogLikelihood is the output layer that
    // is used for classification problem. RandomInitialization means that
    // initial weights are generated randomly in the interval from -1 to 1.
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;
    LeNet module1(1, 28, 28, 10);
    Sequential<>* layer = module1.GetModel();
    model.Add(layer);

    cout << "Training." << endl;

    // Set parameters of Stochastic Gradient Descent (SGD) optimizer.
    SGD<AdamUpdate> optimizer(
        // Step size of the optimizer.
        STEP_SIZE,
        // Batch size. Number of data points that are used in each iteration.
        BATCH_SIZE,
        // Max number of iterations.
        EPOCHS * dataloader.TrainY().n_cols,
        // Tolerance, used as a stopping condition. Such a small value
        // means we almost never stop by this condition, and continue gradient
        // descent until the maximum number of iterations is reached.
        1e-8,
        // Shuffle. If optimizer should take random data points from the dataset at
        // each iteration.
        true,
        // Adam update policy.
        AdamUpdate(1e-8, 0.9, 0.999));

    model.Train(dataloader.TrainX(),
                dataloader.TrainY(),
                optimizer,
                ens::PrintLoss(),
                ens::ProgressBar());

    cout << "Training Complete." <<endl;

    mat predOut;
    mat valX = dataloader.ValidationX();
    model.Predict(valX, predOut);
    // Calculating accuracy on validating data points.
    Row<size_t> predLabels = GetLabels(predOut);
    mat valY = dataloader.ValidationY();
    double validAccuracy = accuracy(predLabels, valY);

    cout << "Validation Accuracy: "<< validAccuracy << endl;

    cout << "Predicting ..." << endl;
    // Matrix to store the predictions on test dataset.
    mat testPredOut;
    // Get predictions on test data points.
    model.Predict(dataloader.TestX(), testPredOut);
    // Generate labels for the test dataset.
    Row<size_t> testPred = GetLabels(testPredOut);
    cout << "Saving predicted labels to results.csv." << endl;

    return 0;
}
