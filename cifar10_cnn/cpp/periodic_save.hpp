/**
 * @file utils.hpp
 * @author Kartik Dutt
 *
 * Definition of Periodic Save utility functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef PERIODIC_SAVE_HPP
#define PERIODIC_SAVE_HPP

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

/**
 * Saves model being trained periodically.
 *
 * @tparam ANNType Type of model which will be used for evaluating metric.
 */
template<typename AnnType>
class PeriodicSave
{
 public:
  /**
   * Constructor for PeriodicSave class.
   *
   * @param network Network type which will be saved periodically.
   * @param filePath Base path / folder where weights will be saved.
   * @param modelPrefix Weights will be stored as
   *      modelPrefix_epoch_loss.bin.
   * @param period Period after which the model will be saved.
   * @param silent Boolean to determine whether or not to print saving
   *      of model.
   * @param output Outputstream where output will be directed.
   */
  PeriodicSave(AnnType& network,
               const std::string filePath = "./",
               const std::string modelPrefix = "model",
               const size_t period = 1,
               const bool silent = false,
               std::ostream& output = arma::get_cout_stream()) :
               network(network),
               filePath(filePath),
               modelPrefix(modelPrefix),
               period(period),
               silent(silent),
               output(output)
  {
    // Nothing to do here.
  }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double objective)
  {
    if (epoch % period == 0)
    {
      std::string objectiveString = std::to_string(objective);
      std::replace(objectiveString.begin(), objectiveString.end(), '.', '_');
      std::string modelName = modelPrefix + "_" + std::to_string(epoch) + "_" +
          objectiveString;
      mlpack::data::Save(filePath + modelName + ".bin", modelPrefix, network);
      if (!silent)
        output << "Model saved as " << modelName << std::endl;
    }

    return false;
  }

 private:
  // Reference to the model which will be used for evaluated using the metric.
  AnnType& network;

  // Locally held string that depicts path for saving the model.
  std::string filePath;

  // Locally held string that depicts the prefix name of model being trained.
  std::string modelPrefix;

  // Period to save the model.
  size_t period;

  // Locally held boolean to determine whether to print success / failure output
  // when model is saved.
  bool silent;

  // The output stream that all data is to be sent to; example: std::cout.
  std::ostream& output;
};

#endif
