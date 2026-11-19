/**
 * @file utils.hpp
 * @author Omar Shrit
 *
 * Feature extraction shared by `train` and `infer`, so both always compute the
 * same features.
 */
#ifndef MOVEMENT_RECOGNITION_UTILS_HPP
#define MOVEMENT_RECOGNITION_UTILS_HPP

#include <mlpack.hpp>

// Applying Fast Fourier Transform to extract frequency domain, to allow to
// capture the periodicity. In addition to this we are adding time domain
// information using statistical methods (mean, stddev, median) to capture the
// intensity of the movements.
inline arma::fvec WindowToFeatures(const arma::fmat& win)
{
  const size_t numBins = win.n_cols / 2 + 1;

  const arma::cx_fmat spectrum = arma::fft(win.t());
  const arma::fmat power = arma::square(arma::abs(spectrum.rows(0, numBins - 1)));

  return arma::join_cols(arma::vectorise(power),
                         arma::mean(win, 1),
                         arma::stddev(win, 0, 1),
                         arma::median(win, 1));
}

#endif
