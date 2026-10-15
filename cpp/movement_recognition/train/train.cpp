/**
 * @file train/train.cpp
 * @author Omar Shrit
 *
 * Train a movement-recognition neural network, on the device, from the CSV
 * files written by the collect tool.  Each <label>_<date>.csv is loaded, cut
 * into fixed-length windows, turned into FFT power-spectrum features, and used
 * to train a small f32 feed-forward network.
 *
 * The window size and step are hardcoded constants (kWindow / kStep) at the top
 * of main(); edit them in the source if your movements are slower or faster.
 *
 *   train DIR model            # data dir, output directory "model"
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <algorithm>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <vector>

#define MLPACK_ENABLE_ANN_SERIALIZATION
#define MLPACK_ENABLE_ANN_SERIALIZATION_FMAT

#include <mlpack.hpp>

#include "../model.hpp"

using namespace mlpack;


namespace fs = std::filesystem;

constexpr size_t kHidden = 64;

namespace {

// The label is the file name without the trailing "_<date>".
std::string LabelOf(const fs::path& p)
{
  const std::string stem = p.stem().string();
  const size_t u = stem.rfind('_');

  if (u == std::string::npos)
    return stem;

  return stem.substr(0, u);
}

bool LoadRecording(const fs::path& path, size_t window, arma::fmat& raw)
{
  // mlpack loads text column-major, so each CSV column becomes a row.
  data::TextOptions opts;
  opts.HasHeaders() = true;
  opts.Fatal() = true;

  std::cerr << "  load " << path.filename().string() << " ... ";

  data::Load(path.string(), raw, opts);

  std::cerr << "loaded " << raw.n_rows << " channels(+ts) x " << raw.n_cols
            << " samples";

  if (raw.n_rows < 2)
  {
    std::cerr << " -- < 2 rows, skipped\n";
    return false;
  }

  if (raw.n_cols < window)
  {
    std::cerr << " -- fewer than the window size (" << window
              << ") samples, skipped\n";
    return false;
  }

  // Drop the timestamp row; the remaining rows are the sensor channels.
  raw.shed_row(0);

  return true;
}

// ===========================================================================
// The feature pipeline, end to end
// ===========================================================================
//
// A set of steps from a raw recording on disk to the feature matrix the
// network trains on.  The example below uses accel only (3 channels: ax, ay,
// az) with window = 256 and step = 128 (50% overlap), the training defaults;
// the code itself works for any channel count and window length.
//
//
// STAGE A -- the raw recording, as `collect` wrote it and `LoadRecording`
//            hands it back (after dropping the timestamp row).
//            shape = (channels x samples): rows are channels, columns are the
//            consecutive samples over the whole recording.
//
//                 sample0  sample1  sample2  ...            sampleN-1
//         ax  →  [  ax0      ax1      ax2    ...              axN-1  ]  row 0
//         ay  →  [  ay0      ay1      ay2    ...              ayN-1  ]  row 1
//         az  →  [  az0      az1      az2    ...              azN-1  ]  row 2
//
//
// STAGE B -- ExtractWindows slides a window of `window` samples along the
//            columns, advancing by `step` each time.  With step < window the
//            windows OVERLAP, so one recording yields many training windows:
//
//         |<-------- window=256 ------->|
//         [============ w0 =============]                         (start s=0)
//                        |<-------- window ------->|
//                        [======= w1 ==============]              (start s=128)
//                                        [======= w2 ==... ]      (start s=256)
//         └ step=128 ┘
//
//            Each window w_i is the sub-matrix raw.cols(s, s+window-1),
//            shape (channels x window) = (3 x 256).  Every window becomes one
//            feature column via WindowToFeatures (STAGE C).
//
//
// STAGE C -- WindowToFeatures turns one (channels x window) window into a
//            single feature column.  arma::fft transforms each COLUMN
//            independently, so the trick is to put each channel in a column:
//
//   C1. the window                     C2. transpose: win.t()
//       (channels x window)                (window x channels), channel = column
//                                                  ax     ay     az
//          s0  s1 ... s255            t0   [  ax0    ay0    az0  ]
//    ax  [ ax0 ax1... ax255]          t1   [  ax1    ay1    az1  ]
//    ay  [ ay0 ay1... ay255]   ──►    ...  [  ...    ...    ...  ]
//    az  [ az0 az1... az255]          t255 [  ax255  ay255  az255]
//
//   C3. arma::fft runs DOWN each column      C4. one-sided half + power:
//       (one 1-D FFT per channel, batched         square(abs(rows(0,numBins-1)))
//        in a single call, no cross-mixing)        numBins = 256/2+1 = 129
//           FFT(ax) FFT(ay) FFT(az)                   pow(ax) pow(ay) pow(az)
//     bin0  [ AX0    AY0    AZ0  ]              bin0   [  .       .       .  ]
//     bin1  [ AX1    AY1    AZ1  ]      ──►      ...   (129 rows kept)
//     ...   [ ...    ...    ...  ]              bin128 [  .       .       .  ]
//     bin255[ AX255  AY255  AZ255]
//
//   C5. vectorise() flattens the power matrix column by column, then the raw
//       per-channel time-domain stats (mean, stddev, median) are appended.
//       The FFT bins capture periodicity (walking cadence); the stats capture
//       posture and intensity (the mean encodes tilt for a static pose).
//
//       [ ax bin0..bin128 | ay bin0..bin128 | az bin0..bin128 | mean | std | median ]
//         └── 129 ──────┘ └── 129 ──────┘ └── 129 ──────┘ └ 3 ─┘└3 ─┘└─ 3 ──┘
//         └──────── FFT power: channels*(window/2+1) = 387 ────┘└ stats: 3*channels=9┘
//
//            feature length = channels*(window/2+1) + 3*channels = 387 + 9 = 396.
//
//
// STAGE D -- main() stacks every window's feature column into the matrix X
//            (one column per window) with its class label in the row y, then
//            standardizes, splits, and trains the network on it.
//
//                   w0    w1    w2   ...  wM-1
//          feat0  [  .     .     .   ...    .  ]
//          feat1  [  .     .     .   ...    .  ]   X: (feat x numWindows)
//          ...    [ ...   ...   ...  ...   ... ]      = (396 x M)
//        feat395  [  .     .     .   ...    .  ]
//              y  (  c0    c1    c2   ...  cM-1 )   class index per window
//
// ===========================================================================

// Applying Fast Fourier Transform to extract frequency domain, to allow to
// capture the periodicity. In addition to this we are adding time domain
// information using statistical methods (mean, stddev, median) to capture the
// intensity of the movements.
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

// Cut `raw` (channels x samples) into overlapping windows and turn each into one
// feature column, appending the columns and their class to `cols`/`labels`.
// Consecutive windows start `step` samples apart, so with step < window they
// overlap.  Overlap is a cheap way to get more training windows out of the same
// recording (a 50% overlap roughly doubles them), which measurably improves
// accuracy; it also matches how `infer` slides its window at run time.  No
// window taper is applied -- for these short, low-frequency movement windows a
// rectangular window classifies better than a Hamming-tapered one.
void ExtractWindows(const arma::fmat& raw, size_t window, size_t step,
                    size_t classIdx, std::vector<arma::fvec>& cols,
                    std::vector<size_t>& labels)
{
  size_t numWindows = 0;

  for (size_t s = 0; s + window <= raw.n_cols; s += step)
  {
    cols.push_back(WindowToFeatures(raw.cols(s, s + window - 1)));
    labels.push_back(classIdx);
    ++numWindows;
  }

  const size_t numFeatures = cols.empty() ? 0 : cols.back().n_elem;
  std::cerr << " -> " << numWindows << " windows of " << numFeatures
            << " features\n";
}

double Accuracy(const arma::Row<size_t>& pred, const arma::Row<size_t>& truth)
{
  if (truth.n_elem == 0)
    return 0.0;

  return (double) arma::accu(pred == truth) / truth.n_elem;
}

// Train the network on the training split, report accuracy on the held-out test
// split, and fill `net`.  `patience` is the early-stopping patience.
void TrainNN(const arma::fmat& trainData, const arma::Row<size_t>& trainLabels,
             const arma::fmat& testData, const arma::Row<size_t>& testLabels,
             size_t numClasses, size_t patience, Network& net)
{
  net.Add<Linear<arma::fmat>>(kHidden);
  net.Add<ReLU<arma::fmat>>();
  net.Add<Linear<arma::fmat>>(numClasses);
  net.Add<LogSoftMax<arma::fmat>>();

  const arma::fmat responses = arma::conv_to<arma::fmat>::from(trainLabels);
  const arma::fmat testResponses = arma::conv_to<arma::fmat>::from(testLabels);

  ens::Adam optimizer(
      1e-2,    // step size (learning rate)
      32,      // batch size
      0.9,     // exp. decay for the 1st moment estimate
      0.999,   // exp. decay for the 2nd moment estimate
      1e-8,    // epsilon for numerical stability
      0,       // max iterations: 0 = no limit
      1e-8,    // tolerance
      true);   // shuffle between epochs

  double bestValLoss = std::numeric_limits<double>::infinity();
  arma::fmat bestParams;
  ens::EarlyStopAtMinLossType<arma::fmat> earlyStop(
      [&](const arma::fmat& /* param */)
      {
        const double valLoss = net.Evaluate(testData, testResponses);
        if (valLoss < bestValLoss)
        {
          bestValLoss = valLoss;
          bestParams = net.Parameters();
        }
        return valLoss;
      },
      patience);

  net.Train(trainData, responses, optimizer,
            ens::PrintLoss(),
            ens::ProgressBar(),
            earlyStop);

  if (!bestParams.is_empty())
    net.Parameters() = bestParams;

  arma::fmat scores;
  net.Predict(testData, scores);

  arma::Row<size_t> pred(scores.n_cols);
  for (size_t i = 0; i < scores.n_cols; ++i)
  {
    pred[i] = scores.col(i).index_max();
  }
  std::cout << "neural net test accuracy: " << Accuracy(pred, testLabels)
            << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
  // Window size and step are hardcoded for this example.  These movements can
  // be executed in 2 ~ 3 seconds at 100 Hz. The step is a 50% overlap (window / 2).
  // Edit these two constants here (and the matching ones in infer.cpp) if your movements are
  // slower or faster.
  const size_t window = 256;
  const size_t step   = 128;

  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0]
              << " <data-dir> [out-dir] [patience] [test-split]\n"
                 "  out-dir holds the model file model.bin (default: model)\n";
    return 1;
  }

  const std::string dataDir = argv[1];
  // Output directory for the model; the network, scaler, and class names are
  // all serialized together into model.bin.
  const std::string outDir  = argc > 2 ? argv[2] : "model";
  const size_t patience     = argc > 3 ? std::stoul(argv[3]) : 10;
  const double testSplit    = argc > 4 ? std::stod(argv[4]) : 0.2;

  if (!fs::is_directory(dataDir))
  {
    std::cerr << "error: '" << dataDir << "' is not a directory\n";
    return 1;
  }

  // Create the output directory, then build the three fixed file paths in it.
  std::error_code ec;
  fs::create_directories(outDir, ec);
  if (ec)
  {
    std::cerr << "error: cannot create output directory '" << outDir
              << "': " << ec.message() << "\n";
    return 1;
  }
  const std::string modelFile = (fs::path(outDir) / "model.bin").string();

  // Collect the CSV files in the data directory.
  std::vector<fs::path> files;
  for (const fs::directory_entry& entry : fs::directory_iterator(dataDir))
  {
    if (entry.path().extension() == ".csv")
      files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());

  std::cerr << "data dir '" << dataDir << "': " << files.size()
            << " .csv file(s), window=" << window << ", step=" << step << "\n";
  if (files.empty())
  {
    std::cerr << "error: no .csv files in '" << dataDir << "'\n";
    return 1;
  }

  // Map each label to a class index, then turn every file into feature columns.
  std::map<std::string, size_t> classOf;
  std::vector<std::string> classNames;
  std::vector<arma::fvec> cols;
  std::vector<size_t> labels;

  for (size_t f = 0; f < files.size(); ++f)
  {
    const fs::path& path = files[f];
    const std::string label = LabelOf(path);

    if (classOf.count(label) == 0)
    {
      classOf[label] = classNames.size();
      classNames.push_back(label);
    }
    const size_t classIdx = classOf[label];

    std::cerr << "[" << label << "]\n";

    arma::fmat raw;
    if (LoadRecording(path, window, raw))
      ExtractWindows(raw, window, step, classIdx, cols, labels);
  }

  // Every file must yield the same feature length (same sensors + window).
  const size_t feat = cols.empty() ? 0 : cols[0].n_elem;
  for (size_t i = 0; i < cols.size(); ++i)
  {
    if (cols[i].n_elem != feat)
    {
      std::cerr << "error: window " << i << " has " << cols[i].n_elem
                << " features but the first has " << feat
                << " -- were the files recorded with different --sensors?\n";
      return 1;
    }
  }

  std::cerr << "totals: " << cols.size() << " windows, " << feat
            << " features, " << classNames.size() << " classes (";
  for (size_t i = 0; i < classNames.size(); ++i)
  {
    std::cerr << (i ? ", " : "") << classNames[i];
  }
  std::cerr << ")\n";

  if (cols.size() < 2 || classNames.size() < 2)
  {
    std::cerr << "error: need >=2 windows and >=2 labels. Collect more data, "
                 "or lower the window size (currently " << window
              << ") in the source so short recordings yield windows.\n";
    return 1;
  }

  // Assemble the feature matrix X (one column per window) and the label row y.
  arma::fmat X(feat, cols.size());
  arma::Row<size_t> y(cols.size());
  for (size_t i = 0; i < cols.size(); ++i)
  {
    X.col(i) = cols[i];
    y[i] = labels[i];
  }
  std::cout << X.n_cols << " windows, " << X.n_rows << " features, "
            << classNames.size() << " classes.\n";

  arma::fmat trainData, testData;
  arma::Row<size_t> trainLabels, testLabels;
  data::Split(X, y, trainData, testData, trainLabels, testLabels, testSplit);
  std::cerr << "split: " << trainData.n_cols << " train, " << testData.n_cols
            << " test\n";

  // Since we have features from time domain and frequency domain, it is
  // better to standardize all of features to have similar scale.  The scaler
  // works directly on f32 matrices, so no double conversion is needed.
  MovementModel model;
  model.classes = classNames;
  model.scaler.Fit(trainData);
  model.scaler.Transform(trainData, trainData);
  model.scaler.Transform(testData, testData);

  TrainNN(trainData, trainLabels, testData, testLabels, classNames.size(),
            patience, model.net);

  // Serialize the network, scaler, and class names together into model.bin.
  data::Save(modelFile, model, data::BIN);

  std::cout << "saved model to '" << modelFile << "'\n";
  return 0;
}
