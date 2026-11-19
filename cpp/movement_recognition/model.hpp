/**
 * @file model.hpp
 * @author Omar Shrit
 *
 * The trained movement-recognition model: the network, the feature scaler, and
 * the class names, bundled so they serialize into a single `.bin` file.
 */
#ifndef MOVEMENT_RECOGNITION_MODEL_HPP
#define MOVEMENT_RECOGNITION_MODEL_HPP

#include <mlpack.hpp>

struct MovementModel
{
  mlpack::FFN<mlpack::NegativeLogLikelihoodType<arma::fmat>,
              mlpack::GlorotInitialization,
              arma::fmat> net;
  mlpack::data::StandardScaler<arma::fmat> scaler;
  std::vector<std::string> classes;

  // Classify one feature column and return the class name.  The features are
  // standardized with the scaler fitted in train before predicting.
  std::string Classify(const arma::fmat& feat)
  {
    arma::fmat scaled;
    scaler.Transform(feat, scaled);

    arma::fmat scores;
    net.Predict(scaled, scores);
    const size_t idx = scores.col(0).index_max();
    return idx < classes.size() ? classes[idx] : "?";
  }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar(CEREAL_NVP(net));
    ar(CEREAL_NVP(scaler));
    ar(CEREAL_NVP(classes));
  }
};

#endif
