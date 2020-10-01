/**
 * @file monitor.hpp
 * @author Marcus Edel
 *
 * Definition of miscellaneous monitor routines.
 */
#ifndef GYM_MONITOR_HPP
#define GYM_MONITOR_HPP

#include "client.hpp"

namespace gym {


/*
 * Definition of the monitor class.
 */
class Monitor
{
 public:
  /**
   * Create the Parser object.
   */
  Monitor();

  /**
   * Set the client which is connected with the server.
   *
   * @param c The client object which is connected with the server.
   */
  void client(Client& c);

  /*
   * Start the monitor using the specified paramter.
   *
   * @param directory The directory where to record stats.
   * @param force Clear out existing training data from this directory.
   * @param resume Retain the training data already in this directory
   *        (merge with existing files).
   */
  void start(const std::string& directory, const bool force, const bool resume);

  /*
   * Close the monitor.
   */
  void close();
private:
  //! Locally-stored client pointer used to communicate with the connected
  //! server.
  Client* clientPtr;
};
} // namespace gym

// Include implementation.
#include "monitor_impl.hpp"

#endif