/**
 * @file infer/infer.cpp
 * @author Omar Shrit
 *
 * Live movement-recognition inference on the device.  It reads the GY-89 IMU
 * directly, slides a window over the stream, runs the SAME per-channel FFT used
 * for training (one arma::fft call) to get features, and classifies each window
 * with the f32 neural network trained by `train`.
 *
 * `train` writes the network, feature scaler, and class names together into a
 * single model.bin in a directory.  Point infer at that directory.
 *
 *   infer accel /dev/i2c-0 model                   # prints predictions to stdout
 *
 * Everything runs in f32 to stay light on the 64 MB device.  The feature
 * extraction here is deliberately identical to train/train.cpp -- keep the two
 * in sync if you change one.  The window size and step are hardcoded constants
 * (see main()) and MUST match the ones in train.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <atomic>
#include <chrono>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Enable mlpack's neural-network serialization so we can load the trained
// network with data::Load (see the mnist_simple_f32 example).
#define MLPACK_ENABLE_ANN_SERIALIZATION
#define MLPACK_ENABLE_ANN_SERIALIZATION_FMAT
#include <mlpack.hpp>

#include "../model.hpp"
#include "../driver/sensor_board.hpp"

using namespace mlpack;


namespace {

std::atomic<bool> g_stop{false};
void HandleSigint(int) { g_stop = true; }

// Read the data from the sensors in the same order that has been used in
// collect.cpp, if the order changes there, then this one needs to be adapted.
arma::fvec SampleToColumn(const Sensors& s, const Reading& r)
{
  const ImuSample& m = r.motion;
  arma::fvec col(s.Channels());
  size_t i = 0;

  if (s.accel)
  {
    col[i++] = m.ax;
    col[i++] = m.ay;
    col[i++] = m.az;
  }

  if (s.gyro)
  {
    col[i++] = m.gx;
    col[i++] = m.gy;
    col[i++] = m.gz;
  }

  if (s.mag)
  {
    col[i++] = m.mx;
    col[i++] = m.my;
    col[i++] = m.mz;
  }

  if (s.baro)
  {
    col[i++] = r.pressurePa;
    col[i++] = r.tempC;
  }

  return col;
}

// ===========================================================================
// Turning one window into a feature vector (must match train's pipeline)
// ===========================================================================
//
// infer feeds each live window through the SAME transform train used, so the
// features line up with the model.  Here `win` is the current sliding window
// (the newest `window` samples the sensor loop has buffered), shape
// (channels x window).  The example below uses accel only (3 channels: ax, ay,
// az) with window = 256 (the training default); the code works for any
// channel count and window length.  arma::fft transforms each COLUMN
// independently, so the trick is to put each channel in a column:
//
//   1. the window                       2. transpose: win.t()
//      (channels x window)                  (window x channels), channel = column
//                                                  ax     ay     az
//          s0  s1 ... s255            t0   [  ax0    ay0    az0  ]
//    ax  [ ax0 ax1... ax255]          t1   [  ax1    ay1    az1  ]
//    ay  [ ay0 ay1... ay255]   ──►    ...  [  ...    ...    ...  ]
//    az  [ az0 az1... az255]          t255 [  ax255  ay255  az255]
//
//   3. arma::fft runs DOWN each column       4. one-sided half + power:
//      (one 1-D FFT per channel, batched          square(abs(rows(0,numBins-1)))
//       in a single call, no cross-mixing)         numBins = 256/2+1 = 129
//           FFT(ax) FFT(ay) FFT(az)                   pow(ax) pow(ay) pow(az)
//     bin0  [ AX0    AY0    AZ0  ]              bin0   [  .       .       .  ]
//     bin1  [ AX1    AY1    AZ1  ]      ──►      ...   (129 rows kept)
//     ...   [ ...    ...    ...  ]              bin128 [  .       .       .  ]
//     bin255[ AX255  AY255  AZ255]
//
//   5. vectorise() flattens the power matrix column by column, then the raw
//      per-channel time-domain stats (mean, stddev, median) are appended:
//
//      [ ax bin0..bin128 | ay bin0..bin128 | az bin0..bin128 | mean | std | median ]
//        └── 129 ──────┘ └── 129 ──────┘ └── 129 ──────┘ └ 3 ─┘└3 ─┘└─ 3 ──┘
//        └──────── FFT power: channels*(window/2+1) = 387 ────┘└ stats: 3*channels=9┘
//
//      feature length = channels*(window/2+1) + 3*channels = 387 + 9 = 396.
//
// See train.cpp for the full pipeline (recording -> sliding windows -> here).
// ===========================================================================

// A (channels x window) window -> one feature vector: the one-sided FFT power
// spectrum of each channel, followed by per-channel mean, stddev, and median.
// MUST match train's WindowToFeatures.
arma::fvec WindowToFeatures(const arma::fmat& win)
{
  const size_t numBins = win.n_cols / 2 + 1;

  const arma::cx_fmat spectrum = arma::fft(win.t());
  const arma::fmat power = arma::square(arma::abs(spectrum.rows(0, numBins - 1)));

  return arma::join_cols(arma::vectorise(power),
                         arma::mean(win, 1),
                         arma::stddev(win, 0, 1),
                         arma::median(win, 1));
}

// Classify one feature column -> class name (arg-max of the network output).
// Apply the SAME standardization train fitted (the scaler works in f32)
// before predicting.
std::string Classify(Network& nn, data::StandardScaler<arma::fmat>& scaler,
                     const std::vector<std::string>& classes,
                     const arma::fmat& feat)
{
  // The scaler works directly on f32 matrices, so no double conversion is
  // needed.
  arma::fmat scaled;
  scaler.Transform(feat, scaled);

  arma::fmat scores;
  nn.Predict(scaled, scores);
  const size_t idx = scores.col(0).index_max();
  return idx < classes.size() ? classes[idx] : "?";
}

// --- the live classification loop -------------------------------------------

// Sample the board at `rateHz`, and every time `window` samples have been
// buffered, turn them into features, classify, print the prediction, and slide
// the window forward by `step`.  Runs until Ctrl-C.
int RunInference(const SensorBoard& board, Network& nn,
                 data::StandardScaler<arma::fmat>& scaler,
                 const std::vector<std::string>& classes,
                 size_t window, size_t step, double rateHz)
{
  using namespace std::chrono;
  const Sensors& sensors = board.Selected();
  const steady_clock::duration period =
      duration_cast<steady_clock::duration>(duration<double>(1.0 / rateHz));
  std::deque<arma::fvec> buf;
  steady_clock::time_point nextTick = steady_clock::now();

  while (!g_stop.load())
  {
    Reading reading;
    if (!board.Read(reading))
    {
      std::cerr << "error: I2C read failed during sampling\n";
      return 1;
    }
    buf.push_back(SampleToColumn(sensors, reading));

    if (buf.size() >= window)
    {
      // Pack the last `window` samples into a (channels x window) matrix.
      arma::fmat win(sensors.Channels(), window);
      for (size_t j = 0; j < window; ++j)
        win.col(j) = buf[j];

      const std::string pred =
          Classify(nn, scaler, classes, WindowToFeatures(win));
      // std::endl flushes, so each prediction appears live.
      std::cout << "predict: " << pred << std::endl;

      // Slide the window forward by `step` samples.
      for (size_t j = 0; j < step && !buf.empty(); ++j)
        buf.pop_front();
    }

    nextTick += period;
    std::this_thread::sleep_until(nextTick);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  // Positional arguments: the sensors, the I2C device, and the directory train
  // wrote model.bin to.  The sample rate is 100 Hz.  The window and step are
  // hardcoded constants below and must match the ones in train.cpp.
  if (argc != 4)
  {
    std::cerr << "Usage: " << argv[0]
              << " <sensors> <device> <model-dir>\n"
                 "  e.g. " << argv[0] << " accel /dev/i2c-0 model\n";
    return 1;
  }

  const std::string sensorSpec = argv[1];
  const std::string device     = argv[2];
  const std::string modelDir   = argv[3];
  const double rateHz = 100.0;
  // Window size and step: must be identical to the constants in train.cpp.
  const size_t window = 256;
  const size_t step   = 128;

  Sensors sensors;
  if (!ParseSensors(sensorSpec, sensors))
  {
    std::cerr << "error: sensors must be 'all' or a comma list of "
                 "accel,gyro,mag,baro\n";
    return 2;
  }

  // The model lives in one file inside the directory.
  namespace fs = std::filesystem;
  const std::string modelFile = (fs::path(modelDir) / "model.bin").string();

  // Load the network, scaler, and class names in one shot.
  MovementModel model;
  data::Load(modelFile, model, data::BIN + data::Fatal);

  // Quick dimension check before running the inference loop, Our target here
  // is to verify that the channels from the sensors and the features numbers
  // after FFT matches the expected input by the neural network.
  const size_t featDim =
      sensors.Channels() * (window / 2 + 1) + sensors.Channels() * 3;
  const size_t expected =
      model.net.InputDimensions().empty() ? 0 : model.net.InputDimensions()[0];
  if (expected != featDim)
  {
    std::cerr << "error: '" << modelDir << "' expects " << expected
              << " features but the chosen sensors/window produce " << featDim
              << " -- they must match training.\n";
    return 1;
  }
  std::cerr << "loaded model from '" << modelDir << "' (window " << window
            << ", " << model.classes.size() << " classes)\n";

  // The whole GY-89 board behind one object: open the bus and bring up only the
  // sensors the model was trained on.
  SensorBoard board(device, sensors);
  if (!board.Begin())
    return 1;

  std::signal(SIGINT, HandleSigint);

  return RunInference(board, model.net, model.scaler, model.classes,
                      window, step, rateHz);
}
