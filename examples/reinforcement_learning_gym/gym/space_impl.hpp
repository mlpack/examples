/**
 * @file space_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of miscellaneous space routines.
 */
#ifndef GYM_SPACE_IMPL_HPP
#define GYM_SPACE_IMPL_HPP

// In case it hasn't been included yet.
#include "space.hpp"
#include "parser.hpp"
#include "messages.hpp"

namespace gym {

inline Space::Space() : parser(new Parser())
{
  // Nothing to do here.
}

inline Space::~Space()
{
  delete parser;
}

inline void Space::client(Client& c)
{
  clientPtr = &c;
}

inline const arma::mat& Space::sample()
{
  clientPtr->send(messages::EnvironmentActionSpaceSample());

  std::string json;
  clientPtr->receive(json);

  parser->parse(json);
  parser->actionSample(this, actionSample);

  return actionSample;
}

} // namespace gym

#endif
