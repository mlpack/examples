/**
 * @file driver/l3gd20h.hpp
 * @author Omar Shrit
 *
 * Driver for the L3GD20H 3-axis MEMS gyroscope, one of the chips on the GY-89
 * breakout.  Exception-free: every method that touches the bus returns a bool
 * (false on I2C error, with errno set by the failing syscall).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MOVEMENT_RECOGNITION_DRIVER_L3GD20H_HPP
#define MOVEMENT_RECOGNITION_DRIVER_L3GD20H_HPP

#include <cstdint>

#include "i2c_bus.hpp"

//! Configuration + read driver for the L3GD20H gyroscope (output in dps).
class L3GD20H
{
 public:
  static constexpr uint8_t kDefaultAddress = 0x6B;    //!< SDO/SA0 high
  static constexpr uint8_t kAlternateAddress = 0x6A;  //!< SDO/SA0 low
  static constexpr uint8_t kWhoAmIValue = 0xD7;

  enum class Scale { DPS_245, DPS_500, DPS_2000 };

  L3GD20H(const I2CBus& bus,
          uint8_t address = kDefaultAddress,
          Scale scale = Scale::DPS_245);

  /**
   * Read WHO_AM_I, configure for continuous output, and report identity.
   *
   * @param identityMatched set to whether WHO_AM_I matched the L3GD20H value.
   * @param whoAmI set to the raw WHO_AM_I byte that was read.
   * @return true if all I2C transactions succeeded (the chip is on the bus);
   *     false on an I2C error.
   */
  bool Begin(bool& identityMatched, uint8_t& whoAmI) const;

  //! Read angular rate of all three axes (dps).  False on I2C error.
  bool Read(float& gx, float& gy, float& gz) const;

 private:
  const I2CBus* bus;
  uint8_t address;
  Scale scale;
  float sensitivity;  //!< dps per LSB for the configured full scale.
};

#endif
