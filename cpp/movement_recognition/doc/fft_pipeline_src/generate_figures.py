#!/usr/bin/env python3
"""
Per-stage figures for the movement-recognition FFT feature pipeline, drawn from
the *real* CSV recordings the `collect` tool writes on the Milk-V Duo.

This is the direct counterpart to mlpack's MFE/MFCC pipeline figures, but for
the pipeline in `train/train.cpp` (WindowToFeatures): the recordings are cut
into overlapping sliding windows (50% overlap by default, no Hamming taper --
unlike MFCC), each channel gets a one-sided FFT power spectrum, and per-channel
time-domain statistics (mean, standard deviation, median) are appended.
Everything here mirrors that C++ code so the figure shows exactly what the
network is trained on.

Emits ONE combined figure (PNG @ 300 dpi + SVG) into ./figures with three
stacked stages, minimal chrome, each labelled in the top-left corner:
    movement_fft_pipeline  (a) raw recording -> (b) overlapping sliding windows
                           -> (c) per-movement az FFT power, DC removed, arrowed

Usage:
    python3 generate_figures.py --data ../../data             # examples data dir
    python3 generate_figures.py --data /path/to/csvs --window 128 --channel az
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

mpl.rcParams.update({
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "standard",
    "savefig.pad_inches": 0,
    "font.family":        "DejaVu Sans",
    "axes.linewidth":     0.8,
})

# A stable, colour-blind-friendly palette; classes are assigned in sorted order.
CLASS_COLORS = [
    "#1f4e8c", "#c1440e", "#1f6f3a", "#7a3ea0",
    "#c47a00", "#0e7c7b", "#8c1f4e", "#4e4e4e",
]


# --------------------------------------------------------------------------
# The pipeline -- a direct port of train/train.cpp's WindowToFeatures.
# --------------------------------------------------------------------------

def window_power(win):
    """One-sided FFT power spectrum of a (channels x window) window.

    Mirrors: spectrum = fft(win.t()); power = square(abs(spectrum.rows(0,
    numBins-1))).  arma::fft is column-wise on win.t(), i.e. one FFT per
    channel along time.  Returns (channels x numBins)."""
    num_bins = win.shape[1] // 2 + 1
    spectrum = np.fft.fft(win, axis=1)             # per channel, along time
    return np.abs(spectrum[:, :num_bins]) ** 2


def window_features(win):
    """The full feature vector for one window: flattened power spectrum, then
    per-channel mean, standard deviation (ddof=1, matching arma stddev's
    default N-1 norm), and median.  Matches join_cols(vectorise(power),
    mean, stddev, median) in train.cpp."""
    power = window_power(win)         # (channels x num_bins)
    return np.concatenate([
        # Row-major flatten of (channels x num_bins) gives [ch0 bins, ch1 bins,
        # ...], the same order as arma's column-major vectorise of the C++
        # (num_bins x channels) power matrix.
        power.reshape(-1),
        win.mean(axis=1),
        win.std(axis=1, ddof=1),
        np.median(win, axis=1),
    ])


def extract_windows(mat, window, step):
    """Cut (channels x samples) into windows of `window` samples spaced `step`
    apart (step < window => overlapping), exactly like ExtractWindows in
    train.cpp.  No taper."""
    n = mat.shape[1]
    return [mat[:, s:s + window] for s in range(0, n - window + 1, step)]


# --------------------------------------------------------------------------
# Loading the real recordings.
# --------------------------------------------------------------------------

def label_of(path):
    """Label = file name without the trailing "_<date>" (matches LabelOf in
    train.cpp: everything before the last underscore)."""
    stem = Path(path).stem
    u = stem.rfind("_")
    return stem[:u] if u != -1 else stem


def load_recording(path):
    """Load one collect CSV.  Returns (channel_names, channels x samples,
    sample_rate_hz).  The first column is the Unix-us timestamp; it is used to
    estimate the sample rate and then dropped, leaving the sensor channels."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    ts = raw[:, 0]                                 # timestamp_unix_us
    channels = raw[:, 1:].T                        # (channels x samples)
    names = header[1:]

    # Sample rate from the median timestamp step (robust to a few jittered rows).
    if len(ts) > 1:
        dt_us = np.median(np.diff(ts))
        fs = 1e6 / dt_us if dt_us > 0 else 100.0
    else:
        fs = 100.0
    return names, channels, fs


def load_dataset(data_dir):
    """Load every *.csv under data_dir, grouped by movement label.  Returns
    dict label -> list of (channel_names, channels x samples, fs), plus the
    shared channel-name list and sample rate (from the first file)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no CSV files found under {data_dir!r}")

    by_label = {}
    names0, fs0 = None, None
    for path in files:
        names, mat, fs = load_recording(path)
        if names0 is None:
            names0, fs0 = names, fs
        by_label.setdefault(label_of(path), []).append((names, mat, fs))
    return by_label, names0, fs0


# --------------------------------------------------------------------------
# Plotting helpers (same visual language as the MFE/MFCC figures).
# --------------------------------------------------------------------------

PANEL_FIGSIZE = (4.5, 2.4)
MARGINS = dict(left=0.18, right=0.97, top=0.93, bottom=0.22)


def new_panel():
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    fig.subplots_adjust(**MARGINS)
    return fig, ax


def strip(ax, *, xlabel=None, ylabel=None):
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#333333")
    ax.tick_params(axis="both", which="major",
                   labelsize=9, length=3, width=0.6, color="#333333")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=9.5, labelpad=2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9.5, labelpad=2)


def add_label(ax, text):
    ax.text(0.018, 0.92, text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))


def add_caption(fig, ax, text, dy=0.085):
    """Place a bold step title *under* an axes, centred on it, `dy` figure-
    fractions below the axes' bottom edge (leaving room for the x-label)."""
    box = ax.get_position()
    return fig.text(box.x0 + box.width / 2, box.y0 - dy, text,
                    ha="center", va="top", fontsize=13, fontweight="bold",
                    color="#222222")


def save(fig, outdir, name):
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches=None)
    fig.savefig(outdir / f"{name}.svg", bbox_inches=None)
    print(f"  wrote {name}.png  +  {name}.svg")


def save_panel(fig, ax, outdir, name, extra_artists=(), pad=0.06):
    """Crop a single axes out of the combined figure and save it on its own.

    The crop box is the axes' tight bounding box (which already includes its
    tick labels and axis titles) unioned with any `extra_artists` (e.g. the
    step caption underneath), grown by `pad` inches so nothing is clipped."""
    fig.canvas.draw()  # tight bboxes are only valid after a draw
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer)]
    boxes += [a.get_window_extent(renderer) for a in extra_artists]
    from matplotlib.transforms import Bbox
    bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1.0, 1.0).padded(pad)
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches=bbox)
    fig.savefig(outdir / f"{name}.svg", bbox_inches=bbox)
    print(f"  wrote {name}.png  +  {name}.svg")


def channel_index(names, want):
    """Index of channel `want` (e.g. 'az'); falls back to the first channel."""
    return names.index(want) if want in names else 0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="directory of collect *.csv recordings")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--window", type=int, default=256,
                    help="samples per window (FFT length); matches train's "
                         "default of 256, which resolves the sub-Hz cadence "
                         "differences between movements")
    ap.add_argument("--channel", default="az",
                    help="axis whose row carries the per-movement arrows in "
                         "panel (c) (the strongest-signal axis; default az)")
    ap.add_argument("--fview", type=float, default=5.0,
                    help="upper frequency (Hz) shown in the cadence-band panel (c)")
    ap.add_argument("--step", type=int, default=None,
                    help="samples between consecutive windows; default window/2 "
                         "(50%% overlap), matching train's default")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    by_label, names, fs = load_dataset(args.data)
    window = args.window
    step = args.step if args.step else max(1, window // 2)
    num_bins = window // 2 + 1
    freqs = np.arange(num_bins) * fs / window
    labels = sorted(by_label)
    colors = {lab: CLASS_COLORS[i % len(CLASS_COLORS)]
              for i, lab in enumerate(labels)}
    ch = channel_index(names, args.channel)

    print(f"dataset: {len(labels)} movements {labels}")
    print(f"channels: {names}")
    print(f"sample rate ~ {fs:.1f} Hz, window {window}, step {step} "
          f"({num_bins} FFT bins up to {fs/2:.1f} Hz), spectrum channel '{names[ch]}'")

    # For each movement, gather all its windows once (reused by several panels).
    windows_by_label = {}
    for lab in labels:
        wins = []
        for _, mat, _ in by_label[lab]:
            wins.extend(extract_windows(mat, window, step))
        windows_by_label[lab] = wins
        if not wins:
            print(f"  warning: '{lab}' has no full window of {window} samples")

    # Pick a "reference" movement for the single-window illustrative panels
    # (a), (b), (c), (e): the most *dynamic* one (largest mean per-channel
    # variance), so the raw trace and its spectrum are visually informative --
    # a flat "sitting" recording would make dull panels.  Only movements that
    # actually produced a full window are eligible.
    def dynamism(lab):
        return float(np.mean([w.var() for w in windows_by_label[lab]])) \
            if windows_by_label[lab] else -1.0
    ref_label = max(labels, key=dynamism)
    ref_mat = by_label[ref_label][0][1]

    # ======================================================================
    # One combined figure, three stacked stages of the feature pipeline:
    #   (a) the raw recording,
    #   (b) the overlapping sliding windows train cuts it into,
    #   (c) each movement's FFT power spectrum (vertical az axis), with an arrow
    #       to its dominant cadence frequency.
    # This is the single figure embedded in the mlpack tutorial.
    # ======================================================================
    fview = min(args.fview, fs / 2)
    az = names[ch]  # the spectrum axis (default az)

    # Flow-diagram layout: (a) and (b) sit side by side on the top row with an
    # arrow (a) -> (b) between them; (c) sits on its own row below, with an
    # arrow (b) -> (c) leading down into it.  Each panel's title is placed
    # *under* its axes (below the x-label), like the caption of a flow step.
    fig = plt.figure(figsize=(10.5, 8.6))
    TOP_B, TOP_T = 0.66, 0.96          # top-row axes vertical span
    BOT_B, BOT_T = 0.15, 0.45          # bottom-row axes vertical span
    #                                    (gap between rows leaves room for the
    #                                     right-angled (b) -> (c) elbow arrow)
    AW = 0.375                          # top-panel width
    AX_L, BX_L = 0.075, 0.575          # left edges of (a) and (b)
    CX_L, CW = 0.190, 0.620            # (c) left edge and width
    axA = fig.add_axes([AX_L, TOP_B, AW, TOP_T - TOP_B])
    axB = fig.add_axes([BX_L, TOP_B, AW, TOP_T - TOP_B])
    axC = fig.add_axes([CX_L, BOT_B, CW, BOT_T - BOT_B])

    # ---- (a) raw signal: the reference recording's three accel axes ----------
    n_show = min(ref_mat.shape[1], window + 3 * step)
    t = np.arange(n_show) / fs
    accel_axes = [n for n in ("ax", "ay", "az") if n in names] or names[:3]
    palette = ["#1f4e8c", "#3a78c2", "#7faedb", "#c1440e", "#1f6f3a"]
    for i, nm in enumerate(accel_axes):
        axA.plot(t, ref_mat[names.index(nm)][:n_show],
                 color=palette[i % len(palette)], linewidth=1.0, label=nm)
    axA.set_xlim(0, t[-1] if n_show else 1)
    axA.margins(y=0.22)
    axA.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.85,
               edgecolor="none", ncol=len(accel_axes))
    axA.text(0.03, 0.06, f"'{ref_label}' recording", transform=axA.transAxes,
             fontsize=9.5, color="#1f4e8c", va="bottom",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    strip(axA, xlabel="Time (s)", ylabel="Acceleration")

    # ---- (b) overlapping sliding windows -------------------------------------
    sig = ref_mat[ch][:n_show]
    tt = np.arange(len(sig)) / fs
    axB.plot(tt, sig, color="#1f4e8c", linewidth=1.0, zorder=3)
    axB.margins(y=0.30)
    lo, hi = axB.get_ylim()
    band_colors = ["#1f6f3a", "#c47a00", "#7a3ea0", "#0e7c7b"]
    starts = list(range(0, len(sig) - window + 1, step))[:4]
    band_h = (hi - lo) * 0.11
    for k, start in enumerate(starts):
        y0 = hi - (k + 1) * band_h
        axB.axvspan(start / fs, (start + window) / fs,
                    ymin=(y0 - lo) / (hi - lo),
                    ymax=(y0 + band_h * 0.8 - lo) / (hi - lo),
                    color=band_colors[k % len(band_colors)], alpha=0.5, zorder=2)
    if len(starts) >= 2:
        y_arrow = hi - (len(starts) + 0.6) * band_h
        axB.annotate("", xy=(starts[1] / fs, y_arrow),
                     xytext=(starts[0] / fs, y_arrow),
                     arrowprops=dict(arrowstyle="<->", color="#333", lw=0.9))
        axB.text((starts[0] + starts[1]) / 2 / fs, y_arrow - band_h * 0.1,
                 f"step = window/{round(window/step)}", ha="center", va="top",
                 fontsize=8, color="#333")
    axB.text(0.98, 0.04, "overlapping windows (no taper)", transform=axB.transAxes,
             ha="right", va="bottom", fontsize=8.5, color="#a06000",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    axB.set_xlim(0, tt[-1] if len(tt) else 1)
    strip(axB, xlabel="Time (s)", ylabel=f"{az}")

    # ---- (c) per-movement FFT power spectrum (az), DC removed, arrowed --------
    # DC (bin 0) is gravity on the vertical axis -- huge and identical across
    # movements, so it is dropped here for legibility (but kept in the features
    # the model trains on).  The x-axis is zoomed to the cadence band, where at
    # this window length each movement's step rate lands in a distinct bin.
    spec = {lab: np.mean([window_power(w)[ch] for w in windows_by_label[lab]],
                         axis=0)
            for lab in labels if windows_by_label[lab]}
    ymax = max((s[1:].max() for s in spec.values() if num_bins > 1), default=1.0)
    for lab in labels:
        if lab in spec:
            axC.plot(freqs[1:], spec[lab][1:], color=colors[lab], linewidth=1.4,
                     label=lab)
    peaks = {lab: (freqs[int(np.argmax(s[1:]) + 1) if num_bins > 1 else 0],
                   s[int(np.argmax(s[1:]) + 1) if num_bins > 1 else 0])
             for lab, s in spec.items()}
    axC.set_xlim(0, fview)
    axC.set_ylim(0, ymax * 2.05 if ymax > 0 else 1)
    order = sorted(peaks, key=lambda l: peaks[l][0])
    n = len(order)
    row_y = [ymax * 1.42, ymax * 1.70]
    # A movement with essentially no power in the cadence band is not moving:
    # its "peak" is just sensor noise (e.g. sitting, whose az is a flat ~1 g of
    # gravity that lives entirely in the dropped DC bin).  Reporting a frequency
    # for it would be meaningless, so we caption it in plain language instead.
    still_cutoff = ymax * 1e-3
    for i, lab in enumerate(order):
        fpk, ppk = peaks[lab]
        is_still = ppk < still_cutoff
        note = "flat line\n(no motion)" if is_still else f"~{fpk:.1f} Hz"
        # Spread labels left-to-right by peak order, but start past the "(c)"
        # corner badge so the leftmost one does not sit under it.
        xt = fview * (0.20 + 0.74 * (i / (n - 1))) if n > 1 else fview * 0.5
        axC.annotate(f"{lab}\n{note}",
                     xy=(min(fpk, fview), ppk), xytext=(xt, row_y[i % 2]),
                     fontsize=7.5, color=colors[lab], fontweight="bold",
                     ha="center", va="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", color=colors[lab],
                                     lw=0.8, shrinkA=3, shrinkB=2))
    strip(axC, xlabel="Frequency (Hz)", ylabel=f"Power ({az}, DC removed)")

    # ---- step titles under each panel + flow arrows between the stages --------
    capA = add_caption(fig, axA, "(a) raw recording")
    capB = add_caption(fig, axB, "(b) sliding windows")
    capC = add_caption(fig, axC, "(c) FFT power per movement")

    # Export each stage on its own FIRST, before the inter-panel arrows are
    # drawn, so a neighbouring arrowhead never bleeds into a standalone crop.
    # The panels stay pixel-identical to the combined figure's panels.
    save_panel(fig, axA, outdir, "movement_fft_stage_a_raw", (capA,))
    save_panel(fig, axB, outdir, "movement_fft_stage_b_windows", (capB,))
    save_panel(fig, axC, outdir, "movement_fft_stage_c_spectrum", (capC,))

    posA, posB, posC = axA.get_position(), axB.get_position(), axC.get_position()
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=22, lw=2.2,
                    color="#333333", shrinkA=0, shrinkB=0)
    # (a) -> (b): horizontal, across the gap on the top row (stops short of
    # (b)'s "az" y-label).
    y_mid = (posA.y0 + posA.y1) / 2
    fig.add_artist(FancyArrowPatch(
        (posA.x1 + 0.015, y_mid), (posB.x0 - 0.050, y_mid), **arrow_kw))
    # (b) -> (c): a right-angled elbow -- drop straight down out of (b), turn
    # 90 degrees left to line up over (c)'s centre, then turn 90 degrees down
    # into the top of (c).
    xb = (posB.x0 + posB.x1) / 2
    xc = (posC.x0 + posC.x1) / 2
    y_top = posB.y0 - 0.125            # start, just below (b)'s title
    y_bot = posC.y1 + 0.018            # end, just above (c)
    y_turn = (y_top + y_bot) / 2       # height of the horizontal run
    elbow = MplPath(
        [(xb, y_top), (xb, y_turn), (xc, y_turn), (xc, y_bot)],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    fig.add_artist(FancyArrowPatch(path=elbow, **arrow_kw))

    save(fig, outdir, "movement_fft_pipeline")
    plt.close(fig)

    # Raw arrays for cross-checking against the C++ features.
    np.savez(outdir / "data.npz",
             freqs=freqs, sample_rate=fs, window=window, step=step,
             channel_names=np.array(names, dtype=object),
             ref_label=ref_label,
             class_spectra={lab: s for lab, s in spec.items()})
    print(f"\nCombined figure + data.npz saved under  {outdir.resolve()}")


if __name__ == "__main__":
    main()
