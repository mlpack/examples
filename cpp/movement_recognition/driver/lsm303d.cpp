/**
 * @file driver/lsm303d.cpp
 * @author Omar Shrit
 *
 * Implementation of the LSM303D accelerometer + magnetometer driver
 * (exception-free).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "lsm303d.hpp"

namespace {

// LSM303D register map (subset used here).
constexpr uint8_t REG_WHO_AM_I   = 0x0F;
constexpr uint8_t REG_CTRL1      = 0x20;  // Accel ODR + axis enable.
constexpr uint8_t REG_CTRL2      = 0x21;  // Accel full-scale range.
constexpr uint8_t REG_CTRL5      = 0x24;  // Mag resolution + ODR.
constexpr uint8_t REG_CTRL6      = 0x25;  // Mag full-scale range.
constexpr uint8_t REG_CTRL7      = 0x26;  // Mag mode.
constexpr uint8_t REG_OUT_X_L_M  = 0x08;  // First of six magnetometer bytes.
constexpr uint8_t REG_OUT_X_L_A  = 0x28;  // First of six accelerometer bytes.

constexpr float ACC_SENS_2  = 0.000061f;
constexpr float ACC_SENS_4  = 0.000122f;
constexpr float ACC_SENS_6  = 0.000183f;
constexpr float ACC_SENS_8  = 0.000244f;
constexpr float ACC_SENS_16 = 0.000732f;

constexpr float MAG_SENS_2  = 0.000080f;
constexpr float MAG_SENS_4  = 0.000160f;
constexpr float MAG_SENS_8  = 0.000320f;
constexpr float MAG_SENS_12 = 0.000479f;

float AccelSensitivityFor(LSM303D::AccelScale scale)
{
  switch (scale)
  {
    case LSM303D::AccelScale::G_4:  return ACC_SENS_4;
    case LSM303D::AccelScale::G_6:  return ACC_SENS_6;
    case LSM303D::AccelScale::G_8:  return ACC_SENS_8;
    case LSM303D::AccelScale::G_16: return ACC_SENS_16;
    case LSM303D::AccelScale::G_2:
    default:                        return ACC_SENS_2;
  }
}

float MagSensitivityFor(LSM303D::MagScale scale)
{
  switch (scale)
  {
    case LSM303D::MagScale::GAUSS_2:  return MAG_SENS_2;
    case LSM303D::MagScale::GAUSS_8:  return MAG_SENS_8;
    case LSM303D::MagScale::GAUSS_12: return MAG_SENS_12;
    case LSM303D::MagScale::GAUSS_4:
    default:                          return MAG_SENS_4;
  }
}

uint8_t Ctrl2AccelScaleBits(LSM303D::AccelScale scale)
{
  switch (scale)
  {
    case LSM303D::AccelScale::G_4:  return 0x08;  // 001 << 3
    case LSM303D::AccelScale::G_6:  return 0x10;  // 010 << 3
    case LSM303D::AccelScale::G_8:  return 0x18;  // 011 << 3
    case LSM303D::AccelScale::G_16: return 0x20;  // 100 << 3
    case LSM303D::AccelScale::G_2:
    default:                        return 0x00;  // 000 << 3
  }
}

uint8_t Ctrl6MagScaleBits(LSM303D::MagScale scale)
{
  switch (scale)
  {
    case LSM303D::MagScale::GAUSS_2:  return 0x00;  // 00 << 5
    case LSM303D::MagScale::GAUSS_8:  return 0x40;  // 10 << 5
    case LSM303D::MagScale::GAUSS_12: return 0x60;  // 11 << 5
    case LSM303D::MagScale::GAUSS_4:
    default:                          return 0x20;  // 01 << 5
  }
}

int16_t ToInt16(uint8_t low, uint8_t high)
{
  return static_cast<int16_t>((static_cast<uint16_t>(high) << 8) | low);
}

}  // namespace

LSM303D::LSM303D(const I2CBus& bus,
                 uint8_t address,
                 AccelScale accelScale,
                 MagScale magScale) :
    bus(&bus), address(address), accelScale(accelScale), magScale(magScale),
    accelSensitivity(AccelSensitivityFor(accelScale)),
    magSensitivity(MagSensitivityFor(magScale))
{
}

bool LSM303D::Begin(bool& identityMatched, uint8_t& whoAmI) const
{
  identityMatched = false;
  whoAmI = 0;
  if (!bus->ReadRegister(address, REG_WHO_AM_I, whoAmI))
    return false;
  identityMatched = (whoAmI == kWhoAmIValue);

  // CTRL1: AODR=0110 (100 Hz), Z/Y/X enabled (low three bits) -> 0x67.
  if (!bus->WriteRegister(address, REG_CTRL1, 0x67))
    return false;
  if (!bus->WriteRegister(address, REG_CTRL2, Ctrl2AccelScaleBits(accelScale)))
    return false;
  // CTRL5: M_RES=11 (high res) + M_ODR=100 (100 Hz) -> 0x74.
  if (!bus->WriteRegister(address, REG_CTRL5, 0x74))
    return false;
  if (!bus->WriteRegister(address, REG_CTRL6, Ctrl6MagScaleBits(magScale)))
    return false;
  // CTRL7: MD=00 -> continuous-conversion magnetometer mode.
  if (!bus->WriteRegister(address, REG_CTRL7, 0x00))
    return false;
  return true;
}

bool LSM303D::ReadAccel(float& ax, float& ay, float& az) const
{
  uint8_t raw[6];
  if (!bus->ReadRegisters(address, REG_OUT_X_L_A, raw, 6))
    return false;
  ax = ToInt16(raw[0], raw[1]) * accelSensitivity;
  ay = ToInt16(raw[2], raw[3]) * accelSensitivity;
  az = ToInt16(raw[4], raw[5]) * accelSensitivity;
  return true;
}

bool LSM303D::ReadMag(float& mx, float& my, float& mz) const
{
  uint8_t raw[6];
  if (!bus->ReadRegisters(address, REG_OUT_X_L_M, raw, 6))
    return false;
  mx = ToInt16(raw[0], raw[1]) * magSensitivity;
  my = ToInt16(raw[2], raw[3]) * magSensitivity;
  mz = ToInt16(raw[4], raw[5]) * magSensitivity;
  return true;
}
