/**
 * @file parser_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of miscellaneous parser routines.
 */
#ifndef GYM_PARSER_IMPL_HPP
#define GYM_PARSER_IMPL_HPP

// In case it hasn't been included yet.
#include "parser.hpp"
#include "space.hpp"

namespace gym {

inline Parser::Parser() : dataStream(0)
{
  // Nothing to do here.
}

inline Parser::~Parser()
{
  if ((dataStream != NULL) && (dataStream[0] == '\0'))
    delete[] dataStream;
}

inline Parser::Parser(const std::string& data)
{
  parse(data);
}

inline void Parser::parse(const std::string& data)
{
  if ((dataStream != NULL) && (dataStream[0] == '\0'))
    delete[] dataStream;

  dataStream = new char[data.size() + 1];
  std::copy(data.begin(), data.end(), dataStream);
  dataStream[data.size()] = '\0';

  doc.deserialize_in_place(dataStream);
}

inline void Parser::actionSample(const Space* space, arma::mat& sample)
{
  if (space->type == Space::DISCRETE)
  {
    if (sample.is_empty())
    {
      sample = arma::mat(1, 1);
    }

    sample(0) = doc.get_object()[0].get_value().as_double();
  }
}

inline void Parser::info(double& reward, bool& done, std::string& info)
{
  const pjson::value_variant* doneValue = doc.find_value_variant("done");
  if (doneValue != NULL)
    done = doneValue->as_bool();

  const pjson::value_variant* rewardValue = doc.find_value_variant("reward");
  if (rewardValue != NULL)
    reward = rewardValue->as_double();
}

inline void Parser::environment(std::string& instance)
{
  pjson::key_value_vec_t& obj = doc.get_object();
  for (size_t i = 0u; i < obj.size(); ++i)
  {
    if (std::strncmp("instance", obj[i].get_key().m_p, 8) == 0)
      instance = obj[i].get_value().get_string_ptr();
  }
}

inline void Parser::observation(const Space* space, arma::mat& observation)
{
  const pjson::value_variant* value = doc.find_value_variant("observation");
  if (space->boxShape.size() == 1)
  {
    observation = arma::mat(space->boxShape[0], 1);
    vec(value->get_array(), observation);
  }
  else if (space->boxShape.size() == 2)
  {
    observation = arma::mat(space->boxShape[1], space->boxShape[0]);

    size_t elem = 0;
    const pjson::value_variant_vec_t& array1 = value->get_array();
    for (size_t i = 0; i < array1.size(); i++)
    {
      const pjson::value_variant_vec_t& array2 = array1[i].get_array();
      for (size_t j = 0; j < array2.size(); j++)
      {
        observation(elem++) = array2[j].as_double();
      }
    }

    observation = observation.t();
  }
  else if (space->boxShape.size() == 3)
  {
    arma::cube temp(space->boxShape[1], space->boxShape[0], space->boxShape[2]);
    observation = arma::mat(space->boxShape[0] * space->boxShape[1],
        space->boxShape[2]);

    size_t elem = 0;
    const pjson::value_variant_vec_t& array1 = value->get_array();
    for (size_t i = 0; i < array1.size(); i++)
    {
      const pjson::value_variant_vec_t& array2 = array1[i].get_array();
      for (size_t j = 0; j < array2.size(); j++)
      {
        size_t z = 0;
        const pjson::value_variant_vec_t& array3 = array2[j].get_array();
        for (size_t k = 0; k < array3.size(); k++, elem++, z++)
        {
          elem = elem % observation.n_rows;
          temp.slice(z)(elem) = array3[k].as_double();
        }
      }
    }

    for (size_t i = 0; i < space->boxShape[2]; ++i)
    {
      arma::mat slice = arma::trans(temp.slice(i));
      observation.col(i) = arma::vectorise(slice);
    }
  }
}

inline void Parser::space(Space* space)
{
  const pjson::key_value_vec_t& obj = doc.find_value_variant(
      "info")->get_object();
  for (size_t i = 0u; i < obj.size(); ++i)
  {
    if (std::strncmp("name", obj[i].get_key().m_p, 4) == 0)
    {
      if (std::strncmp("MultiDiscrete",
          obj[i].get_value().get_string_ptr(), 13) == 0)
      {
        space->type = Space::MULTIDISCRETE;
      }
      else if (std::strncmp("Discrete",
          obj[i].get_value().get_string_ptr(), 8) == 0)
      {
        space->type = Space::DISCRETE;
      }
      else if (std::strncmp("Box",
          obj[i].get_value().get_string_ptr(), 3) == 0)
      {
        space->type = Space::BOX;
      }
    }
    else if (std::strncmp("n", obj[i].get_key().m_p, 1) == 0)
    {
      space->n = obj[i].get_value().as_int64();
    }
    else if (std::strncmp("high", obj[i].get_key().m_p, 4) == 0)
    {
      vec(obj[i].get_value().get_array(), space->boxHigh);
    }
    else if (std::strncmp("low", obj[i].get_key().m_p, 3) == 0)
    {
      vec(obj[i].get_value().get_array(), space->boxLow);
    }
    else if (std::strncmp("shape", obj[i].get_key().m_p, 5) == 0)
    {
      vec(obj[i].get_value().get_array(), space->boxShape);
    }
  }
}

inline void Parser::vec(const pjson::value_variant_vec_t& vector, arma::mat& v)
{
  size_t idx = 0;
  for (pjson::uint i = 0; i < vector.size(); ++i)
    v(idx++) = vector[i].as_double();
}

inline void Parser::vec(
    const pjson::value_variant_vec_t& vector, std::vector<float>& v)
{
  for (pjson::uint i = 0; i < vector.size(); ++i)
    v.push_back(vector[i].as_float());
}

inline void Parser::vec(
    const pjson::value_variant_vec_t& vector, std::vector<int>& v)
{
  for (pjson::uint i = 0; i < vector.size(); ++i)
    v.push_back(vector[i].as_int64());
}

inline void Parser::url(std::string& url)
{
  pjson::key_value_vec_t& obj = doc.get_object();
  for (size_t i = 0u; i < obj.size(); ++i)
  {
    if (std::strncmp("url", obj[i].get_key().m_p, 3) == 0)
      url = obj[i].get_value().get_string_ptr();
  }
}

} // namespace gym

#endif
