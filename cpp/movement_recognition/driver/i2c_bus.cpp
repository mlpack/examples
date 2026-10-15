/**
 * @file driver/i2c_bus.cpp
 * @author Omar Shrit
 *
 * Implementation of the exception-free Linux i2c-dev wrapper.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "i2c_bus.hpp"

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

I2CBus::I2CBus() : fd(-1), currentAddress(-1) {}

I2CBus::I2CBus(const char* device) : fd(-1), currentAddress(-1)
{
  Open(device);
}

I2CBus::~I2CBus()
{
  if (fd >= 0)
    ::close(fd);
}

I2CBus::I2CBus(I2CBus&& other) noexcept :
    fd(other.fd), currentAddress(other.currentAddress)
{
  other.fd = -1;
  other.currentAddress = -1;
}

I2CBus& I2CBus::operator=(I2CBus&& other) noexcept
{
  if (this != &other)
  {
    if (fd >= 0)
      ::close(fd);
    fd = other.fd;
    currentAddress = other.currentAddress;
    other.fd = -1;
    other.currentAddress = -1;
  }
  return *this;
}

bool I2CBus::Open(const char* device)
{
  if (fd >= 0)
    ::close(fd);
  currentAddress = -1;
  fd = ::open(device, O_RDWR);
  return fd >= 0;
}

bool I2CBus::SelectDevice(uint8_t address) const
{
  if (currentAddress == static_cast<int>(address))
    return true;
  if (::ioctl(fd, I2C_SLAVE, address) < 0)
    return false;
  currentAddress = address;
  return true;
}

bool I2CBus::ReadRegister(uint8_t address, uint8_t reg, uint8_t& value) const
{
  if (fd < 0 || !SelectDevice(address))
    return false;
  if (::write(fd, &reg, 1) != 1)
    return false;
  return ::read(fd, &value, 1) == 1;
}

bool I2CBus::ReadRegisters(uint8_t address,
                           uint8_t reg,
                           uint8_t* buffer,
                           std::size_t count) const
{
  if (fd < 0 || !SelectDevice(address))
    return false;

  // Setting the high bit of the sub-address requests auto-increment, which is
  // how the L3GD20H and LSM303D expose their X/Y/Z output as a single burst.
  const uint8_t autoIncrement = static_cast<uint8_t>(reg | 0x80);
  if (::write(fd, &autoIncrement, 1) != 1)
    return false;

  const ssize_t got = ::read(fd, buffer, count);
  return got >= 0 && static_cast<std::size_t>(got) == count;
}

bool I2CBus::WriteRegister(uint8_t address, uint8_t reg, uint8_t value) const
{
  if (fd < 0 || !SelectDevice(address))
    return false;
  const uint8_t payload[2] = { reg, value };
  return ::write(fd, payload, 2) == 2;
}
