/**
 * @file parser.hpp
 * @author Marcus Edel
 *
 * Definition of miscellaneous parser routines.
 */
#ifndef GYM_PARSER_HPP
#define GYM_PARSER_HPP

#include <string>
#include <armadillo>

#include "pjson/pjson.h"

namespace gym {


class Space;

/**
 * Definition of a parser class that is used to parse the reponses.
 */
class Parser
{
 public:
  /**
   * Create the Parser object.
   */
  Parser();

  /**
   * Create the Parser object using the specified json string and create the
   * tree to extract the attributes.
   *
   * @param data The data encoded as json string.
   */
  Parser(const std::string& data);

  /*
   * Deconstructor to delete the datastream.
   */
  ~Parser();

  /**
   * Parse the specified json string and create a tree to extract the
   * attributes.
   *
   * @param data The data encoded as json string.
   */
  void parse(const std::string& data);

  /**
   * Parse the observation data.
   *
   * @param space The space information class.
   * @param observation The parsed observation.
   */
  void observation(const Space* space, arma::mat& observation);

  /**
   * Parse the space data.
   *
   * @param space The space information class.
   */
  void space(Space* space);

  /**
   * Parse the action sample.
   *
   * @param space The space information class.
   * @param sample The parsed sample.
   */
  void actionSample(const Space* space, arma::mat& sample);

  /**
   * Parse the info data.
   *
   * @param reward The reward information.
   * @param done The information whether task succeed or not.
   */
  void info(double& reward, bool& done, std::string& info);

  /**
   * Parse the environment data.
   *
   * @param instance The instance identifier.
   */
  void environment(std::string& instance);

  /**
   * Parse the url data.
   *
   * @param url The url.
   */
  void url(std::string& url);
 private:
  //! Store results of the given json string in the row'th of the given
  //! matrix v.
  void vec(const pjson::value_variant_vec_t& vector, arma::mat& v);

  // void vec(pjson::value_variant_vec_t&, arma::mat& v);

  //! Store results of the given json string in the row'th of the given
  //! matrix v.
  void vec(const pjson::value_variant_vec_t& vector, std::vector<float>& v);

  //! Store results of the given json string in the row'th of the given
  //! matrix v.
  void vec(const pjson::value_variant_vec_t& vector, std::vector<int>& v);

  //! Locally-stored document to parse the json string.
  pjson::document doc;

  //! Locally-stored data stream.
  char* dataStream;
};

} // namespace gym


// Include implementation.
#include "parser_impl.hpp"

#endif
