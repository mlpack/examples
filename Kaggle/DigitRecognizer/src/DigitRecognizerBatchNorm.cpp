/**
 * An example of using Feed Forward Neural Network (FFN) for
 * solving Digit Recognizer problem from Kaggle website.
 *
 * The full description of a problem as well as datasets for training
 * and testing are available here https://www.kaggle.com/c/digit-recognizer using BatchNorm
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @author Eugene Freyman
 * @author Manthan-R-Sheth
 */

#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/adam/adam_update.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <Kaggle/kaggle_utils.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

using namespace arma;
using namespace std;

int main()
{
    // Dataset is randomly split into training
    // and validation parts with following ratio.
    constexpr double RATIO = 0.9;
    // The number of neurons in the first layer.
    constexpr int H1 = 100;
    // The number of neurons in the second layer.
    constexpr int H2 = 100;

    // The solution is done in several approaches (CYCLES), so each approach
    // uses previous results as starting point and have a different optimizer
    // options (here the step size is different).

    // Number of iteration per cycle.
    constexpr int ITERATIONS_PER_CYCLE = 10000;

    // Number of cycles.
    constexpr int CYCLES = 20;

    // Step size of an optimizer.
    constexpr double STEP_SIZE = 5e-4;

    // Number of data points in each iteration of SGD
    // Power of 2 is better for data parallelism
    constexpr int BATCH_SIZE = 64;

    cout << "Reading data ..." << endl;

    // Labeled dataset that contains data for training is loaded from CSV file,
    // rows represent features, columns represent data points.
    mat tempDataset;
    // The original file could be download from
    // https://www.kaggle.com/c/digit-recognizer/data
    data::Load("train.csv", tempDataset, true);

    // Originally on Kaggle dataset CSV file has header, so it's necessary to
    // get rid of the this row, in Armadillo representation it's the first column.
    mat dataset = tempDataset.submat(0, 1,
                                     tempDataset.n_rows - 1, tempDataset.n_cols - 1);

    // Splitting the dataset on training and validation parts.
    mat train, valid;
    data::Split(dataset, train, valid, RATIO);

    // Getting training and validating dataset with features only.
    const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
    const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);

    // According to NegativeLogLikelihood output layer of NN, labels should
    // specify class of a data point and be in the interval from 1 to
    // number of classes (in this case from 1 to 10).

    // Creating labels for training and validating dataset.
    const mat trainY = train.row(0) + 1;
    const mat validY = valid.row(0) + 1;

    // Specifying the NN model. NegativeLogLikelihood is the output layer that
    // is used for classification problem. RandomInitialization means that
    // initial weights in neurons are generated randomly in the interval
    // from -1 to 1.
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;
    // This is intermediate layer that is needed for connection between input
    // data and sigmoid layer. Parameters specify the number of input features
    // and number of neurons in the next layer.
    model.Add<Linear<> >(trainX.n_rows, H1);
    // The first sigmoid layer.
    model.Add<SigmoidLayer<> >();
    // BatchNorm layer applied after sigmoid activation as it gives better results practically.
    model.Add<BatchNorm<> >(H1);
    // Intermediate layer between sigmoid layers.
    model.Add<Linear<> >(H1, H2);
    // The second sigmoid layer.
    model.Add<SigmoidLayer<> >();
    //Second BatchNorm layer
    model.Add<BatchNorm<> >(H2);
    // Dropout layer for regularization. First parameter is the probability of
    // setting a specific value to 0.
    // model.Add<Dropout<> >(0.3, true);
    // Intermediate layer.
    model.Add<Linear<> >(H2, 10);
    // LogSoftMax layer is used together with NegativeLogLikelihood for mapping
    // output values to log of probabilities of being a specific class.
    model.Add<LogSoftMax<> >();

    cout << "Training ..." << endl;

    // Setting parameters Stochastic Gradient Descent (SGD) optimizer.
    SGD<AdamUpdate> optimizer(
            // Step size of the optimizer.
            STEP_SIZE,
            // Batch size. Number of data points that are used in each iteration.
            BATCH_SIZE,
            // Max number of iterations
            ITERATIONS_PER_CYCLE,
            // Tolerance, used as a stopping condition. This small number
            // means we never stop by this condition and continue to optimize
            // up to reaching maximum of iterations.
            1e-8,
            // Shuffle. If optimizer should take random data points from the dataset at
            // each iteration.
            true,
            // Adam update policy.
            AdamUpdate(1e-8, 0.9, 0.999));

    // Cycles for monitoring the process of a solution.
    for (int i = 0; i <= CYCLES; i++) {

        // Train neural network. If this is the first iteration, weights are
        // random, using current values as starting point otherwise.
        model.Train(trainX, trainY, optimizer);

        // Don't reset optimizer's parameters between cycles.
        optimizer.ResetPolicy() = false;

        mat predOut;
        // Getting predictions on training data points.
        model.Predict(trainX, predOut);
        // Calculating accuracy on training data points.
        Row<size_t> predLabels = getLabels(predOut);
        double trainAccuracy = accuracy(predLabels, trainY);
        // Getting predictions on validating data points.
        model.Predict(validX, predOut);
        // Calculating accuracy on validating data points.
        predLabels = getLabels(predOut);
        double validAccuracy = accuracy(predLabels, validY);

        cout << i << " - accuracy: train = "<< trainAccuracy << "%," <<
             " valid = "<< validAccuracy << "%" <<  endl;
    }

    cout << "Predicting ..." << endl;

    // Loading test dataset (the one whose predicted labels
    // should be sent to Kaggle website).
    // As before, it's necessary to get rid of header.

    // The original file could be download from
    // https://www.kaggle.com/c/digit-recognizer/data
    data::Load("test.csv", tempDataset, true);
    mat testX = tempDataset.submat(0, 1,
                                   tempDataset.n_rows - 1, tempDataset.n_cols - 1);

    mat testPredOut;
    // Getting predictions on test data points .
    model.Predict(testX, testPredOut);
    // Generating labels for the test dataset.
    Row<size_t> testPred = getLabels(testPredOut);
    cout << "Saving predicted labels to \"results.csv\" ..." << endl;

    // Saving results into Kaggle compatibe CSV file.
    save("results.csv", "ImageId,Label", testPred);
    cout << "Results were saved to \"results.csv\" and could be uploaded to "
         << "https://www.kaggle.com/c/digit-recognizer/submissions for a competition"
         << endl;
    cout << "Finished" << endl;
}