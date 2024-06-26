/**
 * @file space.hpp
 * @author Marcus Edel
 *
 * Defenition of miscellaneous space routines.
 */

#ifndef GYM_SPACE_HPP
#define GYM_SPACE_HPP

#include <iostream>
#include <armadillo>

#include "client.hpp"

namespace gym {

class Parser;

/**
 * Definition of the space class.
 */
class Space
{
 public:
  /**
   * Instantiate the Environment object.
   */
  Space();

  /**
   * Destroy the space object.
   */
  ~Space();

  /**
   * Set the client which is connected with the server.
   *
   * @param c The client object which is connected with the server.
   */
  void client(Client& c);

  /*
   * Return a sample action.
   */
  const arma::mat& sample();

  //! Space type defenition.
  enum SpaceType
  {
    DISCRETE,
    BOX,
    MULTIDISCRETE,
  } type;

  //! Observation and action space information.
  std::vector<int> boxShape;
  std::vector<float> boxHigh;
  std::vector<float> boxLow;
  int n;

private:
  //! Locally-stored parser pointer.
  Parser* parser;

  //! Locally-stored client pointer.
  Client* clientPtr;

  //! Locally-stored action sample.
  arma::mat actionSample;
};

} // namespace gym

// Include implementation.
#include "space_impl.hpp"

#endif