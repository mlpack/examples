/**
 * @file driver/imu.cpp
 * @author Omar Shrit
 *
 * Implementation of the composite 9-DOF IMU driver (exception-free).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "imu.hpp"

#include <cstdio>

IMU::IMU(const I2CBus& bus, uint8_t gyroAddress, uint8_t accelMagAddress) :
    gyro(bus, gyroAddress),
    accelMag(bus, accelMagAddress),
    epoch(std::chrono::steady_clock::now())
{
}

bool IMU::Begin()
{
  bool gyroId = false, amId = false;
  uint8_t gyroWho = 0, amWho = 0;

  const bool gyroIo = gyro.Begin(gyroId, gyroWho);
  if (!gyroIo)
    std::fprintf(stderr, "IMU: L3GD20H not responding (I2C error) -- check "
                 "wiring, the bus/pinmux, and the gyro address.\n");
  else if (!gyroId)
    std::fprintf(stderr, "IMU: L3GD20H WHO_AM_I mismatch (got 0x%02X, expected "
                 "0x%02X).\n", gyroWho, L3GD20H::kWhoAmIValue);

  const bool amIo = accelMag.Begin(amId, amWho);
  if (!amIo)
    std::fprintf(stderr, "IMU: LSM303D not responding (I2C error) -- check "
                 "wiring, the bus/pinmux, and the accel/mag address.\n");
  else if (!amId)
    std::fprintf(stderr, "IMU: LSM303D WHO_AM_I mismatch (got 0x%02X, expected "
                 "0x%02X).\n", amWho, LSM303D::kWhoAmIValue);

  epoch = std::chrono::steady_clock::now();
  return gyroIo && amIo && gyroId && amId;
}

bool IMU::Sample(ImuSample& out) const
{
  const auto now = std::chrono::steady_clock::now();
  out.timestampMicros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(now - epoch)
          .count());

  if (!accelMag.ReadAccel(out.ax, out.ay, out.az))
    return false;
  if (!gyro.Read(out.gx, out.gy, out.gz))
    return false;
  if (!accelMag.ReadMag(out.mx, out.my, out.mz))
    return false;

  return true;
}
