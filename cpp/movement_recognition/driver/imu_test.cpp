/**
 * @file driver/imu_test.cpp
 * @author Omar Shrit
 *
 * Standalone, self-contained test tool for the GY-89 sensors.  It depends only
 * on the drivers in this directory (no mlpack, Armadillo, CLI11 or iostream, and
 * no C++ exceptions), so it builds into a tiny binary you can copy to the device
 * to confirm wiring and check the IMU and barometer.
 *
 *   ./imu_test [i2c-device] [rate-hz]                  stream IMU + read baro
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "bmp180.hpp"
#include "imu.hpp"

namespace {

// Stream live IMU readings, after reporting the BMP180 barometer once.
int RunStream(const char* device, double rateHz)
{
  I2CBus bus(device);
  if (!bus.IsOpen())
  {
    std::fprintf(stderr, "Error: cannot open I2C device '%s'.\n", device);
    return 1;
  }

  IMU imu(bus);
  if (imu.Begin())
    std::printf("GY-89 IMU detected (L3GD20H + LSM303D identities OK).\n");
  else
    std::printf("Proceeding despite the IMU warning(s) above.\n");

  // BMP180 barometer (also on the GY-89 board).
  BMP180 baro(bus);
  bool baroId = false;
  uint8_t baroChip = 0;
  if (baro.Begin(baroId, baroChip) && baroId)
  {
    float tempC = 0.f, pressurePa = 0.f;
    if (baro.Read(tempC, pressurePa))
      std::printf("BMP180 barometer: %.1f C, %.2f hPa\n",
                  tempC, pressurePa / 100.0f);
    else
      std::printf("BMP180 detected but its measurement read failed.\n");
  }
  else
  {
    std::printf("BMP180 not detected (chip id 0x%02X, expected 0x%02X).\n",
                baroChip, BMP180::expectedChipId);
  }

  std::printf("\nSampling IMU at %.0f Hz on %s. Press Ctrl-C to stop.\n",
              rateHz, device);
  std::printf("%10s | %-23s | %-23s | %-23s\n", "t[s]",
              "accel[g] x y z", "gyro[dps] x y z", "mag[gauss] x y z");

  using namespace std::chrono;
  const auto period =
      duration_cast<steady_clock::duration>(duration<double>(1.0 / rateHz));
  ImuSample s;
  while (true)
  {
    if (!imu.Sample(s))
    {
      std::fprintf(stderr, "\nError: I2C read failed during sampling.\n");
      return 1;
    }
    std::printf("\r%10.3f | %7.3f %7.3f %7.3f | %7.1f %7.1f %7.1f | "
                "%7.3f %7.3f %7.3f",
                s.timestampMicros / 1e6,
                s.ax, s.ay, s.az,
                s.gx, s.gy, s.gz,
                s.mx, s.my, s.mz);
    std::fflush(stdout);
    std::this_thread::sleep_for(period);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  const char* device = (argc > 1) ? argv[1] : "/dev/i2c-0";
  const double rateHz = (argc > 2) ? std::atof(argv[2]) : 50.0;
  return RunStream(device, rateHz);
}
