/**
 * @file driver/bmp180.cpp
 * @author Omar Shrit
 *
 * Implementation of the BMP180 barometer driver (exception-free).  The
 * compensation arithmetic follows the integer algorithm in the BMP180
 * datasheet (Bosch, section 3.5).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "bmp180.hpp"

#include <chrono>
#include <thread>

namespace {

// BMP180 register map.
constexpr uint8_t REG_CHIP_ID   = 0xD0;  // reads 0x55
constexpr uint8_t REG_CAL_AC1   = 0xAA;  // first of 11 16-bit coefficients
constexpr uint8_t REG_CONTROL   = 0xF4;
constexpr uint8_t REG_OUT_MSB   = 0xF6;
constexpr uint8_t CMD_READ_TEMP = 0x2E;
constexpr uint8_t CMD_READ_PRES = 0x34;  // OR'd with (oss << 6)

// Maximum conversion time per oversampling setting, milliseconds (datasheet).
constexpr int CONVERSION_MS[4] = { 5, 8, 14, 26 };

void SleepMs(int ms)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

}  // namespace

BMP180::BMP180(const I2CBus& bus, Oversampling oversampling) :
    bus(&bus), oss(static_cast<int>(oversampling)),
    ac1(0), ac2(0), ac3(0), ac4(0), ac5(0), ac6(0),
    b1(0), b2(0), mb(0), mc(0), md(0)
{
}

bool BMP180::Read16(uint8_t reg, uint16_t& out) const
{
  // The BMP180 does not use the auto-increment convention of the other GY-89
  // chips, so read the two bytes as separate register accesses.
  uint8_t msb, lsb;
  if (!bus->ReadRegister(kAddress, reg, msb))
    return false;
  if (!bus->ReadRegister(kAddress, static_cast<uint8_t>(reg + 1), lsb))
    return false;
  out = static_cast<uint16_t>((static_cast<uint16_t>(msb) << 8) | lsb);
  return true;
}

bool BMP180::Begin(bool& identityMatched, uint8_t& chipId)
{
  identityMatched = false;
  chipId = 0;
  if (!bus->ReadRegister(kAddress, REG_CHIP_ID, chipId))
    return false;
  identityMatched = (chipId == kChipId);

  // Read the 11 calibration coefficients (0xAA..0xBF, big-endian 16-bit).
  uint16_t raw[11];
  for (int i = 0; i < 11; ++i)
  {
    if (!Read16(static_cast<uint8_t>(REG_CAL_AC1 + 2 * i), raw[i]))
      return false;
  }
  ac1 = static_cast<int16_t>(raw[0]);
  ac2 = static_cast<int16_t>(raw[1]);
  ac3 = static_cast<int16_t>(raw[2]);
  ac4 = raw[3];
  ac5 = raw[4];
  ac6 = raw[5];
  b1  = static_cast<int16_t>(raw[6]);
  b2  = static_cast<int16_t>(raw[7]);
  mb  = static_cast<int16_t>(raw[8]);
  mc  = static_cast<int16_t>(raw[9]);
  md  = static_cast<int16_t>(raw[10]);
  return true;
}

bool BMP180::ReadRawTemperature(int32_t& ut) const
{
  if (!bus->WriteRegister(kAddress, REG_CONTROL, CMD_READ_TEMP))
    return false;
  SleepMs(5);
  uint16_t value;
  if (!Read16(REG_OUT_MSB, value))
    return false;
  ut = static_cast<int32_t>(value);
  return true;
}

bool BMP180::ReadRawPressure(int32_t& up) const
{
  if (!bus->WriteRegister(kAddress, REG_CONTROL,
                          static_cast<uint8_t>(CMD_READ_PRES | (oss << 6))))
    return false;
  SleepMs(CONVERSION_MS[oss]);

  uint8_t msb, lsb, xlsb;
  if (!bus->ReadRegister(kAddress, REG_OUT_MSB, msb))
    return false;
  if (!bus->ReadRegister(kAddress, REG_OUT_MSB + 1, lsb))
    return false;
  if (!bus->ReadRegister(kAddress, REG_OUT_MSB + 2, xlsb))
    return false;

  up = ((static_cast<int32_t>(msb) << 16) |
        (static_cast<int32_t>(lsb) << 8) | xlsb) >> (8 - oss);
  return true;
}

bool BMP180::Read(float& temperatureC, float& pressurePa) const
{
  int32_t ut, up;
  if (!ReadRawTemperature(ut))
    return false;
  if (!ReadRawPressure(up))
    return false;

  // Temperature (datasheet integer algorithm).
  int32_t x1 = ((ut - static_cast<int32_t>(ac6)) * static_cast<int32_t>(ac5))
               >> 15;
  int32_t x2 = (static_cast<int32_t>(mc) << 11) / (x1 + static_cast<int32_t>(md));
  const int32_t b5 = x1 + x2;
  temperatureC = ((b5 + 8) >> 4) / 10.0f;

  // Pressure.
  const int32_t b6 = b5 - 4000;
  x1 = (static_cast<int32_t>(b2) * ((b6 * b6) >> 12)) >> 11;
  x2 = (static_cast<int32_t>(ac2) * b6) >> 11;
  int32_t x3 = x1 + x2;
  const int32_t b3 =
      (((static_cast<int32_t>(ac1) * 4 + x3) << oss) + 2) / 4;
  x1 = (static_cast<int32_t>(ac3) * b6) >> 13;
  x2 = (static_cast<int32_t>(b1) * ((b6 * b6) >> 12)) >> 16;
  x3 = ((x1 + x2) + 2) >> 2;
  const uint32_t b4 =
      (static_cast<uint32_t>(ac4) * static_cast<uint32_t>(x3 + 32768)) >> 15;
  const uint32_t b7 =
      (static_cast<uint32_t>(up) - b3) * static_cast<uint32_t>(50000 >> oss);

  int32_t p;
  if (b7 < 0x80000000UL)
    p = static_cast<int32_t>((b7 * 2) / b4);
  else
    p = static_cast<int32_t>((b7 / b4) * 2);

  x1 = (p >> 8) * (p >> 8);
  x1 = (x1 * 3038) >> 16;
  x2 = (-7357 * p) >> 16;
  p = p + ((x1 + x2 + 3791) >> 4);

  pressurePa = static_cast<float>(p);
  return true;
}
