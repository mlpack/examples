/**
 * @file driver/lsm303d.hpp
 * @author Omar Shrit
 *
 * Driver for the LSM303D (3-axis accelerometer + 3-axis magnetometer) on the
 * GY-89.  Exception-free: bus methods return bool (false on I2C error).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MOVEMENT_RECOGNITION_DRIVER_LSM303D_HPP
#define MOVEMENT_RECOGNITION_DRIVER_LSM303D_HPP

#include <cstdint>

#include "i2c_bus.hpp"

//! Configuration + read driver for the LSM303D (accel in g, mag in gauss).
class LSM303D
{
 public:
  static constexpr uint8_t kDefaultAddress = 0x1D;    //!< SDO/SA0 high
  static constexpr uint8_t kAlternateAddress = 0x1E;  //!< SDO/SA0 low
  static constexpr uint8_t kWhoAmIValue = 0x49;

  enum class AccelScale { G_2, G_4, G_6, G_8, G_16 };
  enum class MagScale { GAUSS_2, GAUSS_4, GAUSS_8, GAUSS_12 };

  LSM303D(const I2CBus& bus,
          uint8_t address = kDefaultAddress,
          AccelScale accelScale = AccelScale::G_2,
          MagScale magScale = MagScale::GAUSS_4);

  /**
   * Read WHO_AM_I, configure both sensors for continuous 100 Hz output, and
   * report identity.  Returns true if all I2C transactions succeeded.
   */
  bool Begin(bool& identityMatched, uint8_t& whoAmI) const;

  //! Read acceleration of all three axes (g).  False on I2C error.
  bool ReadAccel(float& ax, float& ay, float& az) const;

  //! Read magnetic field of all three axes (gauss).  False on I2C error.
  bool ReadMag(float& mx, float& my, float& mz) const;

 private:
  const I2CBus* bus;
  uint8_t address;
  AccelScale accelScale;
  MagScale magScale;
  float accelSensitivity;  //!< g per LSB.
  float magSensitivity;    //!< gauss per LSB.
};

#endif
