/**
 * @file environment.hpp
 * @author Marcus Edel
 *
 * Definition of miscellaneous environment routines.
 */
#ifndef GYM_ENVIRONMENT_HPP
#define GYM_ENVIRONMENT_HPP

#include <string>
#include <armadillo>

#include "client.hpp"
#include "parser.hpp"
#include "space.hpp"
#include "monitor.hpp"

namespace gym {

/*
 * Definition of the Environment class.
 */
class Environment
{
 public:
  //! Locally stored observation space instance.
  Space observation_space;

  //! Locally stored action space instance.
  Space action_space;

  //! Locally stored monitor instance.
  Monitor monitor;

  //! Locally stored reward value.
  double reward;

  //! Locally stored info value.
  std::string info;

  //! Locally stored done value.
  bool done;

  //! Locally-stored observation object.
  arma::mat observation;

  //! Locally-stored instance identifier.
  std::string instance;

  /**
   * Instantiate the Environment object.
   */
  Environment();

  /*
   * Instantiate the Environment object using the specified parameter.
   *
   * @param host The host name used for the connection.
   * @param port The port used for the connection.
   */
  Environment(const std::string& host,
              const std::string& port);

  /*
   * Instantiate the Environment object using the specified parameter.
   *
   * @param host The host name used for the connection.
   * @param port The port used for the connection.
   * @param environment Name of the environments used to train/evaluate
   *        the model.
   */
  Environment(const std::string& host,
              const std::string& port,
              const std::string& environment);

  /*
   * Instantiate the environment object using the specified environment name.
   *
   * @param environment Name of the environments used to train/evaluate
   *        the model.
   */
  void make(const std::string& environment);

  /*
   * Renders the environment.
   */
  void render();

  /*
   * Close the environment.
   */
  void close();

  /*
   * Resets the state of the environment and returns an initial observation.
   */
  const arma::mat& reset();

  /*
   * Run one timestep of the environment's dynamics using the specified action.
   *
   * @param action The action performed at the timestep.
   */
  void step(const arma::mat& action);

  /*
   * Sets the seed for this env's random number generator.
   *
   * @param s The seed used for the random number generator.
   */
  void seed(const size_t s);

  /*
   * Sets the compression level in range [0, 9] where 0 means no compression.
   *
   * @param compression The compression level.
   */
  void compression(const size_t compression);

  /*
   * Get the environment url.
   */
  std::string url();

 private:
  //! Get the observation space information.
  void observationSpace();

  //! The the action space information.
  void actionSpace();

  //! Locally-stored client object.
  Client client;

  //! Locally-stored parser object.
  Parser parser;

  //! Locally-stored current render value.
  bool renderValue;
};

} // namespace gym

// Include implementation.
#include "environment_impl.hpp"

#endif
