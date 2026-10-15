/**
 * @file driver/i2c_bus.hpp
 * @author Omar Shrit
 *
 * A small, exception-free wrapper around a Linux I2C bus (/dev/i2c-N) using the
 * userspace i2c-dev interface (ioctl(I2C_SLAVE) + read()/write()).  It knows
 * nothing about any particular sensor; the GY-89 chip drivers are built on top
 * of it.  Errors are reported by return value (no exceptions) and errno is left
 * set by the failing syscall, which keeps the binaries small.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MOVEMENT_RECOGNITION_DRIVER_I2C_BUS_HPP
#define MOVEMENT_RECOGNITION_DRIVER_I2C_BUS_HPP

#include <cstddef>
#include <cstdint>

/**
 * Owns a file descriptor to an I2C bus.  Construct, then Open() (or use the
 * device constructor and check IsOpen()).  All accessors take an explicit 7-bit
 * device address so one bus object can talk to every GY-89 chip.
 */
class I2CBus
{
 public:
  //! Construct without opening; call Open() afterwards.
  I2CBus();

  //! Construct and open `device` (e.g. "/dev/i2c-0"); check IsOpen().
  explicit I2CBus(const char* device);

  ~I2CBus();

  // Owns a file descriptor: non-copyable, movable.
  I2CBus(const I2CBus&) = delete;
  I2CBus& operator=(const I2CBus&) = delete;
  I2CBus(I2CBus&& other) noexcept;
  I2CBus& operator=(I2CBus&& other) noexcept;

  //! Open the bus device.  Returns false (and sets errno) on failure.
  bool Open(const char* device);

  //! Whether a bus is currently open.
  bool IsOpen() const { return fd >= 0; }

  //! Read a single 8-bit register into `value`.  Returns false on error.
  bool ReadRegister(uint8_t address, uint8_t reg, uint8_t& value) const;

  /**
   * Read `count` consecutive registers starting at `reg` into `buffer`, setting
   * the auto-increment bit (sub-address MSB) the L3GD20H and LSM303D require for
   * multi-byte bursts.  Returns false on error.  (The BMP180 does not use this
   * convention; it is read one register at a time with ReadRegister.)
   */
  bool ReadRegisters(uint8_t address,
                     uint8_t reg,
                     uint8_t* buffer,
                     std::size_t count) const;

  //! Write `value` to register `reg`.  Returns false on error.
  bool WriteRegister(uint8_t address, uint8_t reg, uint8_t value) const;

 private:
  //! Point the bus at `address` via ioctl(I2C_SLAVE); cached.  False on error.
  bool SelectDevice(uint8_t address) const;

  int fd;
  mutable int currentAddress;
};

#endif
