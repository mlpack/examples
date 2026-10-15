/**
 * @file driver/mag_calibration.hpp
 * @author Omar Shrit
 *
 * Magnetometer calibration: a per-axis hard-iron offset plus a simple soft-iron
 * scale.  Raw magnetometer readings are distorted by nearby ferromagnetic
 * material and the device's own electronics; uncorrected, the field vector
 * traces an off-centre, stretched ellipsoid as the sensor is rotated.  Hard-iron
 * correction re-centres it and the scale equalizes the axes.  Lives in the
 * driver directory so the sensor code is self-contained for testing/adjustment.
 *
 * Small and free of heavy dependencies (just C stdio), so it is header-only:
 * the method bodies are defined inline below the struct.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MOVEMENT_RECOGNITION_DRIVER_MAG_CALIBRATION_HPP
#define MOVEMENT_RECOGNITION_DRIVER_MAG_CALIBRATION_HPP

#include <cstdio>
#include <string>

/**
 * Correction applied as: corrected_i = (raw_i - offset_i) * scale_i.  The
 * default (zero offset, unit scale) is the identity transform.
 */
struct MagCalibration
{
  float offset[3] = {0.f, 0.f, 0.f};  //!< hard-iron offsets (gauss)
  float scale[3]  = {1.f, 1.f, 1.f};  //!< soft-iron per-axis scale

  //! Apply the correction in place.
  void Apply(float& mx, float& my, float& mz) const;

  /**
   * Compute calibration from the per-axis min/max observed while rotating the
   * sensor through all orientations.  Offsets are the range midpoints; scales
   * normalize each axis to the average half-range.
   */
  static MagCalibration FromMinMax(const float minVals[3],
                                   const float maxVals[3]);

  //! Save as a small, human-readable text file.  Returns false on I/O error.
  bool Save(const std::string& path) const;

  //! Load a calibration previously written by Save().  Returns false on error.
  bool Load(const std::string& path);
};

inline void MagCalibration::Apply(float& mx, float& my, float& mz) const
{
  mx = (mx - offset[0]) * scale[0];
  my = (my - offset[1]) * scale[1];
  mz = (mz - offset[2]) * scale[2];
}

inline MagCalibration MagCalibration::FromMinMax(const float minVals[3],
                                                 const float maxVals[3])
{
  MagCalibration cal;

  float halfRange[3];
  float sumHalfRange = 0.f;
  for (int i = 0; i < 3; ++i)
  {
    cal.offset[i] = 0.5f * (maxVals[i] + minVals[i]);
    halfRange[i] = 0.5f * (maxVals[i] - minVals[i]);
    sumHalfRange += halfRange[i];
  }

  const float avgHalfRange = sumHalfRange / 3.f;
  for (int i = 0; i < 3; ++i)
    cal.scale[i] = (halfRange[i] > 1e-9f) ? (avgHalfRange / halfRange[i]) : 1.f;

  return cal;
}

inline bool MagCalibration::Save(const std::string& path) const
{
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f)
    return false;

  std::fprintf(f, "# magnetometer calibration\n"
                  "# offset_x offset_y offset_z scale_x scale_y scale_z\n");
  std::fprintf(f, "%g %g %g %g %g %g\n",
               offset[0], offset[1], offset[2], scale[0], scale[1], scale[2]);
  std::fclose(f);
  return true;
}

inline bool MagCalibration::Load(const std::string& path)
{
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f)
    return false;

  char line[256];
  while (std::fgets(line, sizeof(line), f))
  {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\0')
      continue;
    const int n = std::sscanf(line, "%f %f %f %f %f %f",
                              &offset[0], &offset[1], &offset[2],
                              &scale[0], &scale[1], &scale[2]);
    std::fclose(f);
    return n == 6;
  }
  std::fclose(f);
  return false;
}

#endif
