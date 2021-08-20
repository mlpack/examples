#include <mlpack/core.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer_names.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp> 
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

std::queue<std::string> batchNormRunningMean;
std::queue<std::string> batchNormRunningVar;

template <
    typename OutputLayer = mlpack::ann::SigmoidCrossEntropyError<>,
    typename InitializationRule = mlpack::ann::RandomInitialization>
void LoadWeights(mlpack::ann::FFN<OutputLayer, InitializationRule>& model,
                 std::string modelConfigPath)
{
  std::cout << "Loading Weights\n";
  size_t currentOffset = 0;
  boost::property_tree::ptree xmlFile;
  boost::property_tree::read_xml(modelConfigPath, xmlFile);
  boost::property_tree::ptree modelConfig = xmlFile.get_child("model");

  model.Parameters().fill(0);
  BOOST_FOREACH (boost::property_tree::ptree::value_type const &layer, modelConfig)
  {
    std::string progressBar(81, '-');
    size_t filled = std::ceil(currentOffset * 80.0 / model.Parameters().n_elem);
    progressBar[0] = '[';
    std::fill(progressBar.begin() + 1, progressBar.begin() + filled + 1, '=');
    std::cout << progressBar << "] " << filled * 100.0 / 80.0 << "%\r";
    std::cout.flush();

    // Load Weights.
    if (layer.second.get_child("has_weights").data() != "0")
    {
      arma::mat weights;
      mlpack::data::Load("./" + layer.second.get_child("weight_csv").data(), weights);
      model.Parameters()(arma::span(currentOffset, currentOffset + weights.n_elem - 1),
                         arma::span()) = weights.t();
      currentOffset += weights.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("weight_offset").data());
    }

    // Load Biases.
    if (layer.second.get_child("has_bias").data() != "0")
    {
      arma::mat bias;
      mlpack::data::Load("./" + layer.second.get_child("bias_csv").data(), bias);
      model.Parameters()(arma::span(currentOffset, currentOffset + bias.n_elem - 1),
                         arma::span()) = bias.t();
      currentOffset += bias.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("bias_offset").data());
    }

    if (layer.second.get_child("has_running_mean").data() != "0")
    {
      batchNormRunningMean.push("./" + layer.second.get_child("running_mean_csv").data());
    }

    if (layer.second.get_child("has_running_var").data() != "0")
    {
      batchNormRunningVar.push("./" + layer.second.get_child("running_var_csv").data());
    }
  }
  std::cout << std::endl;
  std::cout << "Loaded Weights\n";
}

void LoadBNMats(arma::mat &runningMean, arma::mat &runningVar)
{
  runningMean.clear();
  if (!batchNormRunningMean.empty())
  {
    mlpack::data::Load(batchNormRunningMean.front(), runningMean);
    batchNormRunningMean.pop();
  }
  else
    std::cout << "This should never happen!\n";

  runningVar.clear();
  if (!batchNormRunningVar.empty())
  {
    mlpack::data::Load(batchNormRunningVar.front(), runningVar);
    batchNormRunningVar.pop();
  }
  else
    std::cout << "This should never happen!\n";
}

template <
    typename OutputLayer = mlpack::ann::SigmoidCrossEntropyError<>,
    typename InitializationRule = mlpack::ann::RandomInitialization>
void HardCodedRunningMeanAndVariance(
    mlpack::ann::FFN<OutputLayer, InitializationRule>& model)
{
  arma::mat runningMean, runningVar;
  // vector<size_t> indices = {1, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23};
  // vector<size_t> indices = {2, 6, 10, 13, 16, 20, 23, 26, 30, 33, 36, 39, 42};
  vector<size_t> indices = {2, 6, 10, 13, 16, 20, 23, 26, 30, 33, 36, 39, 42, 46, 49, 52, 55, 58};
  for (size_t idx : indices)
  {
    // Is this necessary in my case ?
    LoadBNMats(runningMean, runningVar);
    std::cout << "Loading RunningMean and Variance for " << idx << std::endl;
    boost::get<BatchNorm<>*>(model.Model()[idx])->TrainingMean() = runningMean.t();
    boost::get<BatchNorm<>*>(model.Model()[idx])->TrainingVariance() = runningVar.t();
    // boost::get<BatchNorm<>*>(model.Model()[idx])->Deterministic() = true;
  }

  std::cout << batchNormRunningMean.size() << std::endl;
}

void imageTest()
{
  /**
   * 
   * 
   *   cout << "test on images" << std::endl;
  mlpack::data::ImageInfo imageInfo(224, 224, 3, 100);

  input.clear();
  DataLoader<> dataloader;
  dataloader.LoadImageDatasetFromDirectory("./../../../../imagenette_new/", 224, 224, 3, true, 0.2, false);
  dataloader.TrainFeatures() = dataloader.TrainFeatures() / 255.0;


  input = dataloader.TrainFeatures();
  for (int i = 0; i < input.n_cols; i++)
  {
    std::cout << arma::accu(input.col(i)) << " ----> " << dataloader.TrainLabels().col(i) << std::endl;
    output.clear();
    model.Predict(input.col(i), output);
    sum = arma::accu(output);
    std::cout << std::setprecision(10) << sum << " --> " << output.col(0).index_max() <<
      "  " << output.col(0).max() << std::endl;
  }

   * 
   * 
   */
}

int main()
{
  FFN<mlpack::ann::SigmoidCrossEntropyError<>> model;
  
  // layer_1
  model.Add<Convolution<>>(1, 64, 3, 3, 1, 1, 1, 1, 28, 28);
  model.Add<LeakyReLU<>>(0.1);
  model.Add<MaxPooling<>>(2, 2, 2, 2);
  
  // layer_2
  model.Add<Convolution<>>(64, 32, 3, 3, 1, 1, 1, 1, 14, 14);
  model.Add<LeakyReLU<>>(0.1);
  model.Add<MaxPooling<>>(2, 2, 2, 2);
  
  // layer_3
  model.Add<Convolution<>>(32, 16, 3, 3, 1, 1, 1, 1, 7, 7);
  model.Add<LeakyReLU<>>(0.1);
  model.Add<MaxPooling<>>(2, 2, 2, 2);
  
  // layer_4
  model.Add<Convolution<>>(16, 8, 3, 3, 1, 1, 1, 1, 3, 3);
  model.Add<LeakyReLU<>>(0.1);
  model.Add<MaxPooling<>>(2, 2, 2, 2);

  // layer_5
  model.Add<TransposedConvolution<>>(8, 8, 3, 3, 2, 2, 0, 0, 1, 1, 3, 3);
  model.Add<LeakyReLU<>>(0.1);
  
  // layer_6
  model.Add<TransposedConvolution<>>(8, 16, 3, 3, 2, 2, 0, 0, 3, 3, 7, 7);
  model.Add<LeakyReLU<>>(0.1);
  
  // layer_7
  model.Add<TransposedConvolution<>>(16, 32, 2, 2, 2, 2, 0, 0, 7, 7, 14, 14);
  model.Add<LeakyReLU<>>(0.1);
  
  // layer_8
  model.Add<TransposedConvolution<>>(32, 64, 2, 2, 2, 2, 0, 0, 14, 14, 28, 28);
  model.Add<LeakyReLU<>>(0.1);

  // classifier
  model.Add<Convolution<>>(64, 1, 3, 3, 1, 1, 1, 1, 28, 28);
  //model.Add<Sigmoid<>>(); //NEW_Q: Can I use Sigmoid(), I don't see a Sigmoid Implementation in ann namespace, my python model using sigmoid in the output layer ?


  model.ResetParameters();
  model.Parameters().fill(0.0);
  std::cout << model.Parameters().n_elem << std::endl;
  LoadWeights<>(model, "./cfg/denoiseNet.xml");
  std::cout << model.Parameters().n_elem << std::endl;
  mlpack::data::Save("ae.xml", "model", model, false);
  // Can I comment the below line ?
  // HardCodedRunningMeanAndVariance<>(model);
  /*
  arma::mat input(28 * 28* 3, 1), output;
  mlpack::data::Load("../rand_tensor.csv", input);

  model.Predict(input, output);
  double sum = arma::accu(output);
  std::cout << std::setprecision(10) << sum << " --> " << output.col(0).index_max() << std::endl;

  //input.clear();
  // mlpack::data::Load("./../../../../imagenette_new/n03394916/ILSVRC2012_val_00000957.jpg",input, imageInfo);
  mlpack::data::Load("./../../../../imagenette_image.csv", input);
  std::cout << input.n_cols << std::endl;
  if (input.n_cols > 80)
  {
    input = input.t();
    cout << "New cols : " << input.n_cols << std::endl;
  }

  for (int i = 0; i < input.n_cols; i++)
  {
    output.clear();
    model.Predict(input.col(i), output);
    sum = arma::accu(output);
    std::cout << std::setprecision(10) << sum << " --> " << output.col(0).index_max() <<
      "  " << output.col(0).max() << std::endl;
  }
  */

  return 0;
}
