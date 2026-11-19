/**
 * @file collect.cpp
 * @author Omar Shrit
 *
 * Small data-collection app that records GY-89 sensor data to a CSV file.  It
 * depends only on the drivers in driver/ and argument parsing is hand-rolled,
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

#include "driver/sensor_board.hpp"

using namespace std;

// Sample the board at `rateHz`, writing one CSV row per sample until the
// duration elapses.
int Record(const SensorBoard& board, ostream& csv,
           double rateHz, double durationSec)
{
  using namespace chrono;
  const Sensors& s = board.Selected();
  const steady_clock::duration period =
      duration_cast<steady_clock::duration>(duration<double>(1.0 / rateHz));
  const uint64_t durationMicros =
      static_cast<uint64_t>(durationSec * 1e6);

  // Report progress to the terminal every this many samples.
  constexpr uint64_t progressEvery = 50;

  uint64_t count = 0, firstTs = 0, lastTs = 0;
  steady_clock::time_point nextTick = steady_clock::now();

  while (true)
  {
    // Unix time in microseconds.
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    const uint64_t ts =
        static_cast<uint64_t>(now.tv_sec) * 1000000ull + now.tv_nsec / 1000;

    Reading reading;

    if (!board.Read(reading))
    {
      cerr << "\nerror: I2C read failed during sampling: "
                << strerror(errno) << "\n";
      return 1;
    }

    // Write one CSV row.
    const ImuSample& m = reading.motion;
    csv << ts;
    if (s.accel)
      csv << "," << m.ax << "," << m.ay << "," << m.az;
    if (s.gyro)
      csv << "," << m.gx << "," << m.gy << "," << m.gz;
    if (s.mag)
      csv << "," << m.mx << "," << m.my << "," << m.mz;
    if (s.baro)
      csv << "," << reading.pressurePa << "," << reading.tempC;

    csv << "\n";

    if (count == 0)
      firstTs = ts;

    lastTs = ts;

    if (++count % progressEvery == 0)
      cout << "\r  " << count << " samples..." << flush;

    if (ts - firstTs >= durationMicros)
      break;

    nextTick += period;
    this_thread::sleep_until(nextTick);
  }

  const double elapsed = (count > 1) ? (lastTs - firstTs) / 1e6 : 0.0;
  const double effHz = (elapsed > 0.0) ? (count - 1) / elapsed : 0.0;
  cout << "\rWrote " << count << " samples in "
            << fixed << setprecision(2) << elapsed << " s (effective "
            << setprecision(1) << effHz << " Hz).\n";
  return 0;
}

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    cerr << "Usage: " << argv[0] << " <label> [sensors] [out-dir] [device]"
                 " [rate-hz] <duration-sec>\n"
                 "  duration-sec is required and must be > 0\n";
    return 1;
  }

  const string label      = argv[1];
  const string sensorSpec = argc > 2 ? argv[2] : "all";
  const string outDir     = argc > 3 ? argv[3] : ".";
  const string device     = argc > 4 ? argv[4] : "/dev/i2c-0";
  const double rateHz          = argc > 5 ? stod(argv[5]) : 100.0;
  const double durationSec     = argc > 6 ? stod(argv[6]) : 0.0;

  if (rateHz <= 0.0)
  {
    cerr << "error: rate-hz must be > 0\n";
    return 2;
  }

  if (durationSec <= 0.0)
  {
    cerr << "error: duration-sec is required and must be > 0\n";
    return 2;
  }

  Sensors sensors;

  if (!ParseSensors(sensorSpec, sensors))
  {
    cerr << "error: sensors must be 'all' or a comma list of "
                 "accel,gyro,mag,baro\n";
    return 2;
  }

  // The whole GY-89 board behind one object: open the bus and bring up only the
  // selected sensors.
  SensorBoard board(device, sensors);

  if (!board.Begin())
    return 1;

  // Open one dated file named after the label, in the following format:
  // "<dir>/<label>_<YYYYmmdd-HHMMSS>.csv".
  const time_t now = time(nullptr);
  struct tm tmv;
  localtime_r(&now, &tmv);
  char stamp[32];
  strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tmv);
  const string path = outDir + "/" + label + "_" + stamp + ".csv";

  ofstream csv(path);

  if (!csv)
  {
    cerr << "error: cannot open '" << path << "' for writing: "
              << strerror(errno) << "\n";
    return 1;
  }

  // Six significant digits per value; set once for the file.
  csv << setprecision(6);

  csv << "timestamp_unix_us";
  if (sensors.accel)
    csv << ",ax,ay,az";
  if (sensors.gyro)
    csv << ",gx,gy,gz";
  if (sensors.mag)
    csv << ",mx,my,mz";
  if (sensors.baro)
    csv << ",pressure_pa,temp_c";
  csv << "\n";

  cout << "Recording '" << label << "' (" << sensorSpec << ") at "
            << fixed << setprecision(0) << rateHz << " Hz to " << path
            << " for " << durationSec << " s.\n";

  return Record(board, csv, rateHz, durationSec);
}
