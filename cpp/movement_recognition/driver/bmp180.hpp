/**
 * @file driver/bmp180.hpp
 * @author Omar Shrit
 *
 * Driver for the Bosch BMP180 digital barometric pressure + temperature sensor,
 * the third chip on the GY-89 breakout (a 10-DOF board).  Exception-free: bus
 * methods return bool (false on I2C error).  Begin() reads the factory
 * calibration coefficients from the sensor's EEPROM and Read() applies the
 * datasheet compensation to produce temperature (deg C) and pressure (Pa).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MOVEMENT_RECOGNITION_DRIVER_BMP180_HPP
#define MOVEMENT_RECOGNITION_DRIVER_BMP180_HPP

#include <cstdint>

#include "i2c_bus.hpp"

//! Configuration + read driver for the BMP180 (temperature in degC, pressure
//! in Pa).
class BMP180
{
 public:
  static constexpr uint8_t kAddress = 0x77;   //!< fixed I2C address
  static constexpr uint8_t kChipId = 0x55;    //!< value of the ID register

  //! Pressure oversampling setting (more samples = less noise, more time).
  enum class Oversampling { ULTRA_LOW = 0, STANDARD = 1, HIGH = 2,
                            ULTRA_HIGH = 3 };

  BMP180(const I2CBus& bus, Oversampling oversampling = Oversampling::STANDARD);

  /**
   * Verify the chip ID and read the factory calibration coefficients.
   *
   * @param identityMatched set to whether the ID register matched kChipId.
   * @param chipId set to the raw ID byte that was read.
   * @return true if all I2C transactions succeeded; false on an I2C error.
   */
  bool Begin(bool& identityMatched, uint8_t& chipId);

  /**
   * Take a temperature + pressure measurement (this blocks for a few ms while
   * the sensor converts).  Returns false on I2C error.
   *
   * @param temperatureC compensated temperature, degrees Celsius.
   * @param pressurePa compensated pressure, pascals.
   */
  bool Read(float& temperatureC, float& pressurePa) const;

 private:
  bool Read16(uint8_t reg, uint16_t& out) const;
  bool ReadRawTemperature(int32_t& ut) const;
  bool ReadRawPressure(int32_t& up) const;

  const I2CBus* bus;
  int oss;  //!< oversampling value 0..3

  // Factory calibration coefficients (datasheet section 3.4).
  int16_t ac1, ac2, ac3;
  uint16_t ac4, ac5, ac6;
  int16_t b1, b2, mb, mc, md;
};

#endif
