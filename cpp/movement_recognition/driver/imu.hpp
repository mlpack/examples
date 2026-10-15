/**
 * @file driver/imu.hpp
 * @author Omar Shrit
 *
 * A 9-DOF IMU built from the two motion chips on the GY-89 board: the L3GD20H
 * gyroscope and the LSM303D accelerometer/magnetometer.  It is named IMU rather
 * than after the board so that swapping in a different sensor only means
 * changing the two member chips below.  (The board's BMP180 barometer has its
 * own driver, bmp180.hpp.)  Exception-free: Begin()/Sample() return bool.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MOVEMENT_RECOGNITION_DRIVER_IMU_HPP
#define MOVEMENT_RECOGNITION_DRIVER_IMU_HPP

#include <chrono>
#include <cstdint>
#include <functional>

#include "i2c_bus.hpp"
#include "imu_sample.hpp"
#include "l3gd20h.hpp"
#include "lsm303d.hpp"

/**
 * Composite 9-DOF IMU.  Construct it, call Begin(), then Sample() or
 * RunSamplingLoop().
 */
class IMU
{
 public:
  IMU(const I2CBus& bus,
      uint8_t gyroAddress = L3GD20H::kDefaultAddress,
      uint8_t accelMagAddress = LSM303D::kDefaultAddress);

  /**
   * Configure both chips and start the timestamp clock.  Prints a warning to
   * stderr for any chip that does not respond or whose WHO_AM_I mismatches.
   *
   * @return true only if both sensors responded AND reported their expected
   *     identity.  (Sampling may still be attempted if false.)
   */
  bool Begin();

  //! Read one synchronized, timestamped 9-DOF sample.  False on I2C error.
  bool Sample(ImuSample& out) const;

  /**
   * Fixed-rate sampling loop.  Calls `onSample` for each reading; stops when it
   * returns false.  Uses an absolute (sleep_until) schedule so callback jitter
   * does not drift the average rate.
   *
   * @return true if stopped by the callback; false if a sensor read failed.
   */
  bool RunSamplingLoop(double rateHz,
                       const std::function<bool(const ImuSample&)>& onSample)
      const;

  const L3GD20H& Gyro() const { return gyro; }
  const LSM303D& AccelMag() const { return accelMag; }

 private:
  L3GD20H gyro;
  LSM303D accelMag;
  std::chrono::steady_clock::time_point epoch;
};

#endif
