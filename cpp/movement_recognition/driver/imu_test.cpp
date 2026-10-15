/**
 * @file driver/imu_test.cpp
 * @author Omar Shrit
 *
 * Standalone, self-contained test tool for the GY-89 sensors.  It depends only
 * on the drivers in this directory (no mlpack, Armadillo, CLI11 or iostream, and
 * no C++ exceptions), so it builds into a tiny binary you can copy to the device
 * to confirm wiring, check the IMU and barometer, and -- with --calibrate --
 * produce a magnetometer calibration file before collecting any data.
 *
 *   ./imu_test [i2c-device] [rate-hz]                  stream IMU + read baro
 *   ./imu_test --calibrate mag.cal [i2c-device] [secs] magnetometer calibration
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#include "bmp180.hpp"
#include "imu.hpp"
#include "mag_calibration.hpp"

namespace {

std::atomic<bool> g_stop{false};

void HandleSigint(int) { g_stop = true; }

// Sleep helper that keeps an absolute schedule (no rate drift).
template<typename TimePoint>
void SleepUntil(TimePoint& next, std::chrono::steady_clock::duration period)
{
  next += period;
  std::this_thread::sleep_until(next);
}

// Magnetometer calibration: rotate the board through all orientations, track the
// per-axis min/max of the RAW magnetometer, and derive a hard-iron + soft-iron
// correction.  This is the routine that used to be the separate `calibrate` app.
int RunCalibration(const char* outFile, const char* device, double seconds)
{
  I2CBus bus(device);
  if (!bus.IsOpen())
  {
    std::fprintf(stderr, "Error: cannot open I2C device '%s'.\n", device);
    return 1;
  }

  IMU imu(bus);
  imu.Begin();  // configure the chips; raw mag is read via AccelMag() below

  std::printf("Magnetometer calibration: slowly rotate the board through ALL\n"
              "orientations (figure-eights, turning it over) for %.0f s. "
              "Ctrl-C to stop early.\n\n", seconds);

  float minVals[3] = { std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max() };
  float maxVals[3] = { std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest() };

  using namespace std::chrono;
  const auto period = duration_cast<steady_clock::duration>(duration<double>(1.0 / 50.0));
  const auto start = steady_clock::now();
  const auto deadline =
      start + duration_cast<steady_clock::duration>(duration<double>(seconds));

  auto next = start;
  unsigned long count = 0;
  while (!g_stop.load() && steady_clock::now() < deadline)
  {
    float mx, my, mz;
    if (!imu.AccelMag().ReadMag(mx, my, mz))
    {
      std::fprintf(stderr, "\nError: I2C read failed during calibration.\n");
      return 1;
    }
    const float raw[3] = { mx, my, mz };
    for (int i = 0; i < 3; ++i)
    {
      if (raw[i] < minVals[i]) minVals[i] = raw[i];
      if (raw[i] > maxVals[i]) maxVals[i] = raw[i];
    }
    if (++count % 25 == 0)
    {
      std::printf("\r  %lu samples; x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]   ",
                  count, minVals[0], maxVals[0], minVals[1], maxVals[1],
                  minVals[2], maxVals[2]);
      std::fflush(stdout);
    }
    SleepUntil(next, period);
  }
  std::printf("\n");

  if (count < 2)
  {
    std::fprintf(stderr, "Error: not enough samples collected.\n");
    return 1;
  }

  const MagCalibration cal = MagCalibration::FromMinMax(minVals, maxVals);
  std::printf("Offsets (gauss): %.5f %.5f %.5f\n",
              cal.offset[0], cal.offset[1], cal.offset[2]);
  std::printf("Scales:          %.5f %.5f %.5f\n",
              cal.scale[0], cal.scale[1], cal.scale[2]);

  if (!cal.Save(outFile))
  {
    std::fprintf(stderr, "Error: cannot write calibration to '%s'.\n", outFile);
    return 1;
  }
  std::printf("Saved calibration to %s. Use it with: collect --mag-cal %s\n",
              outFile, outFile);
  return 0;
}

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
                baroChip, BMP180::kChipId);
  }

  std::printf("\nSampling IMU at %.0f Hz on %s. Press Ctrl-C to stop.\n",
              rateHz, device);
  std::printf("%10s | %-23s | %-23s | %-23s\n", "t[s]",
              "accel[g] x y z", "gyro[dps] x y z", "mag[gauss] x y z");

  const bool ok = imu.RunSamplingLoop(rateHz, [](const ImuSample& s) {
    std::printf("\r%10.3f | %7.3f %7.3f %7.3f | %7.1f %7.1f %7.1f | "
                "%7.3f %7.3f %7.3f",
                s.timestampMicros / 1e6,
                s.ax, s.ay, s.az,
                s.gx, s.gy, s.gz,
                s.mx, s.my, s.mz);
    std::fflush(stdout);
    return !g_stop.load();
  });

  if (!ok)
  {
    std::fprintf(stderr, "\nError: I2C read failed during sampling.\n");
    return 1;
  }
  std::printf("\nStopped.\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  std::signal(SIGINT, HandleSigint);

  if (argc > 1 && !std::strcmp(argv[1], "--calibrate"))
  {
    if (argc < 3)
    {
      std::fprintf(stderr,
                   "Usage: %s --calibrate <output-file> [i2c-device] [seconds]\n",
                   argv[0]);
      return 2;
    }
    const char* outFile = argv[2];
    const char* device = (argc > 3) ? argv[3] : "/dev/i2c-0";
    const double seconds = (argc > 4) ? std::atof(argv[4]) : 20.0;
    return RunCalibration(outFile, device, seconds);
  }

  const char* device = (argc > 1) ? argv[1] : "/dev/i2c-0";
  const double rateHz = (argc > 2) ? std::atof(argv[2]) : 50.0;
  return RunStream(device, rateHz);
}
