/**
 * @file driver/imu_sample.hpp
 * @author Omar Shrit
 *
 * One synchronized 9-DOF reading from the GY-89.  Kept in the driver directory
 * so the sensor code is self-contained and can be built and tested on its own,
 * independent of the rest of the example.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MOVEMENT_RECOGNITION_DRIVER_IMU_SAMPLE_HPP
#define MOVEMENT_RECOGNITION_DRIVER_IMU_SAMPLE_HPP

#include <cstdint>

/**
 * Acceleration is in g, angular rate in degrees/second, magnetic field in
 * gauss.  The timestamp is monotonic microseconds since the device was started
 * (GY89::Begin()), which is what the collection app records so the sampling
 * frequency can be recovered later from the data alone.
 */
struct ImuSample
{
  uint64_t timestampMicros = 0;
  float ax = 0.f, ay = 0.f, az = 0.f;  // accelerometer, g
  float gx = 0.f, gy = 0.f, gz = 0.f;  // gyroscope, dps
  float mx = 0.f, my = 0.f, mz = 0.f;  // magnetometer, gauss
};

#endif
