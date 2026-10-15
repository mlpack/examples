/**
 * @file driver/sensor_board.hpp
 * @author Omar Shrit
 *
 * The whole GY-89 board behind a single object.  Rather than juggling an I2C
 * bus, the 9-DOF IMU, and the BMP180 barometer separately, an application
 * creates one SensorBoard, brings it up once with Begin(), and then Read()s a
 * full sample.  Only the sensors named in the Sensors selection are powered up
 * and read; the others are left untouched.  Exception-free, like the drivers it
 * composes: Begin()/Read() return bool.
 *
 * This class is only thin composition glue, so it is header-only: the method
 * bodies are defined inline below the class rather than in a separate .cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MOVEMENT_RECOGNITION_DRIVER_SENSOR_BOARD_HPP
#define MOVEMENT_RECOGNITION_DRIVER_SENSOR_BOARD_HPP

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

#include "bmp180.hpp"
#include "i2c_bus.hpp"
#include "imu.hpp"
#include "imu_sample.hpp"

//! Which sensors to record.  accel/gyro/mag come from the IMU; baro is the
//! BMP180.
struct Sensors
{
  bool accel = false, gyro = false, mag = false, baro = false;

  bool AnyMotion() const
  {
    return accel || gyro || mag;
  }

  //! Number of scalar channels the selection produces, in the canonical order
  //! accel(3), gyro(3), mag(3), baro(2) -- the order used for CSV columns and
  //! feature vectors alike.
  size_t Channels() const
  {
    return 3 * accel + 3 * gyro + 3 * mag + 2 * baro;
  }
};

//! Parse "all" or a comma list like "accel,gyro,mag,baro" into a Sensors.
//! Returns false if the spec is empty or contains an unknown name.
inline bool ParseSensors(const std::string& spec, Sensors& out)
{
  if (spec == "all")
  {
    out = { true, true, true, true };
    return true;
  }

  std::string token;
  std::stringstream ss(spec);

  while (std::getline(ss, token, ','))
  {
    if (token == "accel")
      out.accel = true;
    else if (token == "gyro")
      out.gyro = true;
    else if (token == "mag")
      out.mag = true;
    else if (token == "baro")
      out.baro = true;
    else
      return false;
  }

  return out.accel || out.gyro || out.mag || out.baro;
}

//! One reading of the whole board: the 9-DOF motion sample plus the barometer.
//! Only the fields for the selected sensors are meaningful.
struct Reading
{
  ImuSample motion;         //!< accelerometer / gyroscope / magnetometer
  float pressurePa = 0.f;
  float tempC = 0.f;
};

/**
 * The GY-89 board as a single object: open the bus, bring up the selected
 * sensors, then read whole samples.
 */
class SensorBoard
{
 public:
  SensorBoard(const std::string& devicePath, const Sensors& selected)
      : device(devicePath),
        sensors(selected),
        bus(device.c_str()),   // members init in declaration order: device, then bus
        imu(bus),
        baro(bus)
  {
  }

  //! Which sensors this board was told to record (for the CSV columns).
  const Sensors& Selected() const { return sensors; }

  /**
   * Open the I2C bus and bring up every selected sensor.  A chip that fails to
   * identify is only a warning, so a partly-populated board can still be
   * recorded; the one hard error is the bus failing to open.
   *
   * @return false only if the I2C bus could not be opened.
   */
  bool Begin();

  //! Read one sample of all selected sensors.  False on an I2C error.
  bool Read(Reading& out) const;

 private:
  std::string device;
  Sensors sensors;
  I2CBus bus;
  IMU imu;
  BMP180 baro;
};

inline bool SensorBoard::Begin()
{
  if (!bus.IsOpen())
  {
    std::fprintf(stderr, "error: cannot open I2C device '%s': %s\n",
                 device.c_str(), std::strerror(errno));
    return false;
  }

  if (sensors.AnyMotion())
  {
    if (imu.Begin())
      std::printf("IMU detected (sensor identities OK).\n");
    else
      std::printf("Proceeding despite the IMU warning(s) above.\n");
  }

  if (sensors.baro)
  {
    bool baroId = false;
    uint8_t baroChip = 0;
    if (baro.Begin(baroId, baroChip) && baroId)
      std::printf("BMP180 barometer detected.\n");
    else
      std::fprintf(stderr, "warning: BMP180 not detected (chip id 0x%02X); "
                   "pressure/temperature columns may be invalid.\n", baroChip);
  }

  return true;
}

inline bool SensorBoard::Read(Reading& out) const
{
  if (sensors.AnyMotion() && !imu.Sample(out.motion))
    return false;
  if (sensors.baro && !baro.Read(out.tempC, out.pressurePa))
    return false;
  return true;
}

#endif
