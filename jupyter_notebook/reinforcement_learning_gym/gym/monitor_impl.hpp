/**
 * @file monitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of miscellaneous monitor routines.
 */
#ifndef GYM_MONITOR_IMPL_HPP
#define GYM_MONITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "monitor.hpp"
#include "messages.hpp"

namespace gym {

inline Monitor::Monitor()
{
  // Nothing to do here.
}

inline void Monitor::client(Client& c)
{
  clientPtr = &c;
}

inline void Monitor::start(
    const std::string& directory, const bool force, const bool resume)
{
  clientPtr->send(messages::MonitorStart(directory, force, resume));
}

inline void Monitor::close()
{
  clientPtr->send(messages::MonitorClose());
}

} // namespace gym

#endif
