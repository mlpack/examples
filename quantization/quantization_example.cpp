#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

#if ((ENS_VERSION_MAJOR < 2) || \
    ((ENS_VERSION_MAJOR == 2) && (ENS_VERSION_MINOR < 13)))
  #error "need ensmallen version 2.13.0 or later"
#endif

using namespace mlpack;
using namespace std;

CEREAL_REGISTER_MLPACK_LAYERS(arma::fmat);

arma::Row<size_t> getLabels(arma::fmat predOut)
{
  arma::Row<size_t> predLabels(predOut.n_cols);
  for (arma::uword i = 0; i < predOut.n_cols; ++i)
  {
    predLabels(i) = predOut.col(i).index_max();
  }
  return predLabels;
}

int main()
{
  constexpr double RATIO = 0.1;
  constexpr int H1 = 200;
  constexpr int H2 = 100;
  const double STEP_SIZE = 5e-3;
  const size_t BATCH_SIZE = 64;
  const int EPOCHS = 50;

  arma::fmat dataset;
  data::Load("../../../data/mnist_train.csv", dataset, true);

  arma::fmat headerLessDataset =
      dataset.submat(0, 1, dataset.n_rows - 1, dataset.n_cols - 1);

  arma::fmat train, valid;
  data::Split(headerLessDataset, train, valid, RATIO);

  const arma::fmat trainX =
      train.submat(1, 0, train.n_rows - 1, train.n_cols - 1) / 255.0;
  const arma::fmat validX =
      valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1) / 255.0;

  const arma::fmat trainY = train.row(0);
  const arma::fmat validY = valid.row(0);

  FFN<NegativeLogLikelihoodType<arma::fmat>, GlorotInitialization, arma::fmat> model;
  model.Add<LinearType<arma::fmat>>(H1);
  model.Add<ReLUType<arma::fmat>>();
  model.Add<LinearType<arma::fmat>>(H2);
  model.Add<ReLUType<arma::fmat>>();
  model.Add<DropoutType<arma::fmat>>(0.2);
  model.Add<LinearType<arma::fmat>>(10);
  model.Add<LogSoftMaxType<arma::fmat>>();

  cout << "Start training ..." << endl;

  ens::Adam optimizer(
      STEP_SIZE,
      BATCH_SIZE,
      0.9,
      0.999,
      1e-8,
      EPOCHS * trainX.n_cols,
      1e-8,
      true);

  ens::StoreBestCoordinates<arma::fmat> bestCoordinates;

  model.Train(trainX,
              trainY,
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              ens::EarlyStopAtMinLossType<arma::fmat>(
                  [&](const arma::fmat& /* param */)
                  {
                    double validationLoss = model.Evaluate(validX, validY);
                    cout << "Validation loss: " << validationLoss << "." << endl;
                    return validationLoss;
                  }),
              bestCoordinates);

  model.Parameters() = bestCoordinates.BestCoordinates();

  arma::fmat predOut;
  model.Predict(trainX, predOut);
  arma::Row<size_t> predLabels = getLabels(predOut);
  double trainAccuracy =
      arma::accu(predLabels == trainY) / (double) trainY.n_elem * 100;
  model.Predict(validX, predOut);
  predLabels = getLabels(predOut);
  double validAccuracy =
      arma::accu(predLabels == validY) / (double) validY.n_elem * 100;

  cout << "Accuracy: train = " << trainAccuracy << "%,"
       << "\t valid = " << validAccuracy << "%" << endl;

  // Quantize the model parameters
  FFN<NegativeLogLikelihoodType<arma::imat>, GlorotInitialization, arma::imat> quantizedModel = model.Quantize<arma::imat>();

  cout << "Quantized model parameters:\n" << quantizedModel.Parameters() << endl;
  cout << "Minimum value in quantized parameters: " << quantizedModel.Parameters().min() << endl;
  cout << "Maximum value in quantized parameters: " << quantizedModel.Parameters().max() << endl;

  data::Save("model.bin", "model", model, false);

  data::Load("../data/mnist_test.csv", dataset, true);
  arma::fmat testY = dataset.row(dataset.n_rows - 1);
  dataset.shed_row(dataset.n_rows - 1);

  cout << "Predicting on test set..." << endl;
  arma::fmat testPredOut;
  model.Predict(dataset, testPredOut);
  arma::Row<size_t> testPred = getLabels(testPredOut);

  double testAccuracy = arma::accu(testPred == testY) / (double) testY.n_elem * 100;
  cout << "Accuracy: test = " << testAccuracy << "%" << endl;

  cout << "Saving predicted labels to \"results.csv\" ..." << endl;
  testPred.save("results.csv", arma::csv_ascii);

  cout << "Neural network model is saved to \"model.bin\"" << endl;
  cout << "Finished" << endl;
}

