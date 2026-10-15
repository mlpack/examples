/**
 * @file driver/l3gd20h.cpp
 * @author Omar Shrit
 *
 * Implementation of the L3GD20H gyroscope driver (exception-free).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "l3gd20h.hpp"

namespace {

// L3GD20H register map (subset used here).
constexpr uint8_t REG_WHO_AM_I = 0x0F;
constexpr uint8_t REG_CTRL1    = 0x20;  // ODR, bandwidth, power, axis enable.
constexpr uint8_t REG_CTRL4    = 0x23;  // Block data update, full-scale range.
constexpr uint8_t REG_OUT_X_L  = 0x28;  // First of six output bytes (X/Y/Z).

// Sensitivities (datasheet Table 3), dps per LSB.
constexpr float SENS_245  = 0.00875f;
constexpr float SENS_500  = 0.01750f;
constexpr float SENS_2000 = 0.07000f;

float SensitivityFor(L3GD20H::Scale scale)
{
  switch (scale)
  {
    case L3GD20H::Scale::DPS_500:  return SENS_500;
    case L3GD20H::Scale::DPS_2000: return SENS_2000;
    case L3GD20H::Scale::DPS_245:
    default:                       return SENS_245;
  }
}

uint8_t Ctrl4ScaleBits(L3GD20H::Scale scale)
{
  switch (scale)
  {
    case L3GD20H::Scale::DPS_500:  return 0x10;  // 01 << 4
    case L3GD20H::Scale::DPS_2000: return 0x20;  // 10 << 4
    case L3GD20H::Scale::DPS_245:
    default:                       return 0x00;  // 00 << 4
  }
}

int16_t ToInt16(uint8_t low, uint8_t high)
{
  return static_cast<int16_t>((static_cast<uint16_t>(high) << 8) | low);
}

}  // namespace

L3GD20H::L3GD20H(const I2CBus& bus, uint8_t address, Scale scale) :
    bus(&bus), address(address), scale(scale),
    sensitivity(SensitivityFor(scale))
{
}

bool L3GD20H::Begin(bool& identityMatched, uint8_t& whoAmI) const
{
  identityMatched = false;
  whoAmI = 0;
  if (!bus->ReadRegister(address, REG_WHO_AM_I, whoAmI))
    return false;
  identityMatched = (whoAmI == kWhoAmIValue);

  // CTRL1: ODR ~100 Hz, normal mode (PD=1), Z/Y/X enabled (low nibble 0xF).
  if (!bus->WriteRegister(address, REG_CTRL1, 0x0F))
    return false;
  // CTRL4: block data update on (bit 7) + selected full-scale range.
  if (!bus->WriteRegister(address, REG_CTRL4,
                          static_cast<uint8_t>(0x80 | Ctrl4ScaleBits(scale))))
    return false;
  return true;
}

bool L3GD20H::Read(float& gx, float& gy, float& gz) const
{
  uint8_t raw[6];
  if (!bus->ReadRegisters(address, REG_OUT_X_L, raw, 6))
    return false;
  gx = ToInt16(raw[0], raw[1]) * sensitivity;
  gy = ToInt16(raw[2], raw[3]) * sensitivity;
  gz = ToInt16(raw[4], raw[5]) * sensitivity;
  return true;
}
