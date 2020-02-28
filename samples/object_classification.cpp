#include <ensmallen.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <Kaggle/kaggle_utils.hpp>
#include <ensmallen_bits/callbacks/callbacks.hpp>
#include<models/alexnet.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;


int main()
{
  AlexNet alexnet(1, 28, 28, 10, true);
  Sequential<>* layer = alexnet.CompileModel();
  return 0;
}