# Movement-recognition FFT feature-pipeline figure

Reproduces the feature pipeline in `train/train.cpp` (`WindowToFeatures`) as a
single figure, drawn from the **real** CSV recordings the `collect` tool writes
on the Milk-V Duo.  It is the movement-recognition counterpart of mlpack's
MFE/MFCC pipeline figures, but for *this* pipeline: overlapping sliding windows
(no Hamming taper) and a one-sided FFT power spectrum per channel.

The figure has three stacked stages:

| Stage | What's plotted                                                        |
|-------|-----------------------------------------------------------------------|
| (a) raw recording   | one movement's three accelerometer axes over the first few windows |
| (b) sliding windows | the overlapping windows `train` cuts the signal into (step = window/2, no taper) |
| (c) FFT power       | each movement's mean `az` power spectrum, DC removed, cadence band, dominant peak **arrowed** |

## Files

| File                  | Purpose                                                          |
|-----------------------|------------------------------------------------------------------|
| `generate_figures.py` | Python port of `WindowToFeatures`; emits the combined figure + raw arrays |
| `Makefile`            | One-shot build (`make figures DATA=<csv dir>`)                   |

## Quick start

```bash
# 1. python deps (numpy + matplotlib)
python3 -m pip install --user numpy matplotlib

# 2. build, pointing DATA at your collect recordings
make figures DATA=../../data
```

That produces:

```
figures/
  movement_fft_pipeline.png    # the combined figure (also .svg)
  movement_fft_pipeline.svg
  data.npz                     # freqs + per-movement az spectra, for cross-checking
```

## Customizing

```bash
# a different FFT window, spectrum channel, or cadence-band width for panel (c)
python3 generate_figures.py --data ../../data --window 128 --channel ax --fview 8
```

The dataset shipped with this example is accelerometer-only (`ax, ay, az`); if
you collect gyro/magnetometer channels too, pass e.g. `--channel gz`.

## The point of the figure

Panel **(c)** is the payoff: it draws every movement's mean power spectrum on
the vertical axis (`az`, which carries the strongest signal) and marks each
movement's dominant frequency with a small arrow -- so you can read straight off
the FFT which spectrum belongs to which movement: squat (slow, sub-Hz), walking
and the two stair directions (~1.5-2.5 Hz cadence), and a flat, peakless
*sitting*.  Only `az` is shown because the horizontal axes (`ax`/`ay`) carry a
much weaker signal (~10x lower peak power) whose movement peaks bunch into the
same one or two low-frequency bins -- drawing them would suggest a separation
that is not there.

Two choices make that panel legible, and both are worth understanding:

* **The DC bin (bin 0) is dropped from the figure.**  On the accelerometer
  that bin is the constant pull of gravity: orders of magnitude larger than any
  movement and essentially the same for all of them, so it would otherwise
  swamp the cadence peaks.  Note this is a *visualization* choice only -- the
  DC/gravity term is kept in the features the model trains on, because posture
  and device tilt help distinguish the movements (see the experiment below).
* **The window is 256 samples (`--window 256`, also `train`'s default).**
  Human motion lives below a few Hz, and the FFT's frequency resolution is
  `sample_rate / window`.  At 100 Hz a window of 64 gives ~1.6 Hz per bin --
  too coarse to separate walking from stairs, which all land in the first bin.
  A window of 256 gives ~0.4 Hz per bin, enough to tell the movements apart by
  cadence.  The same window happens to give the best accuracy too (see below),
  so the figure and the model agree; pass a different `--window` to explore.

The sample rate is estimated from the recordings' own timestamps, the FFT
length is the `--window` size, and the feature dimension is
`channels * (window/2 + 1) + channels * 3` -- exactly what `train` and `infer`
compute (e.g. 3 accelerometer channels with `--window 256` gives
`3 * 129 + 3 * 3 = 396`).

## What actually improves accuracy: overlap, window size, gravity

Three natural ideas -- overlapping windows, a longer window, and removing the
gravity/DC term -- were each measured on this dataset with an MLP (one 64-unit
hidden layer + standardizing scaler, the same shape as `train`).  The test set
matters here: because overlapping windows share raw samples, a random split
would leak neighbouring windows across the train/test boundary and flatter the
numbers.  These use a **temporal split** instead -- the first 70% of each
recording trains, the tail tests, and no window straddles the cut -- so overlap
only augments *training*:

| window | step         | overlap | train windows | accuracy |
|-------:|--------------|--------:|--------------:|---------:|
| 64     | 64           |     0%  | 264 | 0.811 |
| 64     | 32           |    50%  | 525 | 0.865 |
| 64     | 16           |    75%  | 1044 | 0.865 |
| 128    | 128          |     0%  | 131 | 0.830 |
| 128    | 64           |    50%  | 255 | 0.849 |
| 256    | 256          |     0%  | 64  | 0.760 |
| 256    | 128          |    50%  | 122 | **0.920** |

Takeaways:

* **Overlapping windows help, and it is free.**  Going from no overlap to 50%
  overlap lifts accuracy by extracting more training windows from the same
  recordings, with no new data.  The effect is largest for long windows, which
  are otherwise data-starved: at window 256 a non-overlapping split yields only
  64 training windows (0.760), while a 50% overlap nearly doubles that to 122
  and reaches 0.920.  Past 50% overlap there is no further gain.  This is why
  `train` defaults to a 50% overlap (`step = window / 2`) and why panel (b)
  draws the windows overlapping.
* **A longer window helps *once it has enough windows to train on.***  The best
  configuration here is window 256 at 50% overlap (0.920) -- better than any
  window-64 setting.  A longer window gives the FFT finer frequency resolution
  (which is also why the figure uses `--window 256`); the catch is that longer
  windows produce fewer of them, so the overlap above is what makes the large
  window pay off rather than starve.  Shorter windows still have their place --
  they give lower latency on the device.
* **Removing the DC/gravity term *hurts* accuracy** (a separate run:
  0.865 -> 0.766 at window 64).  Gravity encodes how the board is tilted, which
  helps tell e.g. sitting from walking apart.  So gravity is noise for the
  *picture* but signal for the *model* -- hence it is dropped only from panel
  (c), not from the features.

(A Hamming taper -- the window `mlpack::SlidingWindow` applies for MFE/MFCC --
was also tried and *hurt* here, ~0.865 -> 0.766, because it attenuates the
edges of these already-short low-frequency windows.  The pipeline therefore uses
a plain rectangular window.)

## Cross-checking against the C++ features

`figures/data.npz` holds the frequency axis and each movement's mean power
spectrum.  To compare against mlpack's output, dump the matching window from
C++ (e.g. `power.save("power.txt", arma::raw_ascii)`) and diff.
