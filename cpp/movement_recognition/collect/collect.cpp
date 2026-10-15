/**
 * @file collect.cpp
 * @author Omar Shrit
 *
 * Small data-collection app that records GY-89 sensor data to a CSV file.  It
 * depends only on the drivers in ../driver and argument parsing is hand-rolled,
 * so the binary stays tiny.
 *
 * - choose which sensors to record with the positional `[sensors]` argument (a
 *   comma list like `accel,gyro,mag,baro`, or `all`);
 * - timestamp every row with the Unix clock in microseconds;
 * - name the output file `<label>_<date>.csv`, so the label is the file name
 *   (there is no label column).
 *
 * Recording runs for a fixed duration (the required duration-sec argument) and
 * then stops on its own.
 *
 * Examples:
 *   collect walking accel data /dev/i2c-0 100 30   # accel into data/ for 30 s
 *   collect stairs_up accel,gyro data /dev/i2c-0 100 30
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "../driver/sensor_board.hpp"

namespace {

uint64_t UnixMicros()
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000ull + ts.tv_nsec / 1000;
}

/*
 * Save collected data set in the following format:
 * "<dir>/<label>_<YYYYmmdd-HHMMSS>.csv".
 */ 
std::string MakeFilename(const std::string& dir, const std::string& label)
{
  const time_t now = std::time(nullptr);
  struct tm tmv;
  localtime_r(&now, &tmv);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tmv);
  return dir + "/" + label + "_" + stamp + ".csv";
}

void WriteHeader(std::ostream& f, const Sensors& s)
{
  f << "timestamp_unix_us";

  if (s.accel)
    f << ",ax,ay,az";
  if (s.gyro)
    f << ",gx,gy,gz";
  if (s.mag)
    f << ",mx,my,mz";
  if (s.baro)
    f << ",pressure_pa,temp_c";

  f << "\n";
}

void WriteRow(std::ostream& f, const Sensors& s, uint64_t ts, const Reading& r)
{
  const ImuSample& m = r.motion;
  f << ts;
  if (s.accel)
    f << "," << m.ax << "," << m.ay << "," << m.az;
  if (s.gyro)
    f << "," << m.gx << "," << m.gy << "," << m.gz;
  if (s.mag)
    f << "," << m.mx << "," << m.my << "," << m.mz;
  if (s.baro)
    f << "," << r.pressurePa << "," << r.tempC;

  f << "\n";
}

// Sample the board at `rateHz`, writing one CSV row per sample until the
// duration elapses.
int Record(const SensorBoard& board, std::ostream& csv,
           double rateHz, double durationSec)
{
  using namespace std::chrono;
  const steady_clock::duration period =
      duration_cast<steady_clock::duration>(duration<double>(1.0 / rateHz));
  const uint64_t durationMicros =
      static_cast<uint64_t>(durationSec * 1e6);

  // Report progress to the terminal every this many samples.
  constexpr uint64_t kProgressEvery = 50;

  uint64_t count = 0, firstTs = 0, lastTs = 0;
  steady_clock::time_point nextTick = steady_clock::now();

  while (true)
  {
    const uint64_t ts = UnixMicros();

    Reading reading;

    if (!board.Read(reading))
    {
      std::cerr << "\nerror: I2C read failed during sampling: "
                << std::strerror(errno) << "\n";
      return 1;
    }

    WriteRow(csv, board.Selected(), ts, reading);

    if (count == 0)
      firstTs = ts;

    lastTs = ts;

    if (++count % kProgressEvery == 0)
      std::cout << "\r  " << count << " samples..." << std::flush;

    if (ts - firstTs >= durationMicros)
      break;

    nextTick += period;
    std::this_thread::sleep_until(nextTick);
  }

  const double elapsed = (count > 1) ? (lastTs - firstTs) / 1e6 : 0.0;
  const double effHz = (elapsed > 0.0) ? (count - 1) / elapsed : 0.0;
  std::cout << "\rWrote " << count << " samples in "
            << std::fixed << std::setprecision(2) << elapsed << " s (effective "
            << std::setprecision(1) << effHz << " Hz).\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <label> [sensors] [out-dir] [device]"
                 " [rate-hz] <duration-sec>\n"
                 "  duration-sec is required and must be > 0\n";
    return 1;
  }

  const std::string label      = argv[1];
  const std::string sensorSpec = argc > 2 ? argv[2] : "all";
  const std::string outDir     = argc > 3 ? argv[3] : ".";
  const std::string device     = argc > 4 ? argv[4] : "/dev/i2c-0";
  const double rateHz          = argc > 5 ? std::stod(argv[5]) : 100.0;
  const double durationSec     = argc > 6 ? std::stod(argv[6]) : 0.0;

  if (rateHz <= 0.0)
  {
    std::cerr << "error: rate-hz must be > 0\n";
    return 2;
  }

  if (durationSec <= 0.0)
  {
    std::cerr << "error: duration-sec is required and must be > 0\n";
    return 2;
  }

  Sensors sensors;

  if (!ParseSensors(sensorSpec, sensors))
  {
    std::cerr << "error: sensors must be 'all' or a comma list of "
                 "accel,gyro,mag,baro\n";
    return 2;
  }

  // The whole GY-89 board behind one object: open the bus and bring up only the
  // selected sensors.
  SensorBoard board(device, sensors);

  if (!board.Begin())
    return 1;

  // Open one dated file named after the label.
  const std::string path = MakeFilename(outDir, label);

  std::ofstream csv(path);

  if (!csv)
  {
    std::cerr << "error: cannot open '" << path << "' for writing: "
              << std::strerror(errno) << "\n";
    return 1;
  }

  // Six significant digits per value (see WriteRow); set once for the file.
  csv << std::setprecision(6);

  WriteHeader(csv, sensors);
  std::cout << "Recording '" << label << "' (" << sensorSpec << ") at "
            << std::fixed << std::setprecision(0) << rateHz << " Hz to " << path
            << " for " << durationSec << " s.\n";

  const int rc = Record(board, csv, rateHz, durationSec);

  csv.close();

  return rc;
}
