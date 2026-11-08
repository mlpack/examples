# Binary size: analysis, results, and how to reproduce

This document records the size analysis of the `train` and `infer` binaries for
the Milk-V Duo (SOPHGO CV1800B, RISC-V C906, ~28 MB usable RAM), and the exact
commands used to analyse them. It is meant to double as the supporting material
for the upstream mlpack `DISABLE_STB` work (see
[Disabling mlpack's image (STB) support](#disabling-mlpacks-image-stb-support)).

Note that this tutorial does **not** minimize neural-network serialization: it
uses mlpack's full serialization (the `mnist_simple_f32` pattern), which is the
dominant contributor to `train`/`infer` size.  Shrinking that is left to a
follow-up tutorial (see [Step 1](#step-1-nn-serialization)); the size work
recorded here is the image/STB and CLI11 changes.

All numbers are for the static, stripped RISC-V binaries built with the
cross-compile flow in `CMakeLists.txt` (already size-optimised:
`-Os -ffunction-sections -fdata-sections -fvisibility=hidden -flto
-fno-unwind-tables`, static link with `--gc-sections`).

## Results

Two independent size changes are applied in this example (both measured on the
same toolchain).  Note that the network is serialized with mlpack's full
neural-network serialization (`MLPACK_ENABLE_ANN_SERIALIZATION` +
`CEREAL_REGISTER_MLPACK_LAYERS`, the `mnist_simple_f32` pattern); trimming that
is intentionally out of scope here (see [Step 1](#step-1-nn-serialization)).

| Change | What | `train` | `infer` |
|--------|------|--------:|--------:|
| image/STB | mlpack image/STB compiled out (`-DDISABLE_STB=ON`) | −84 KB | −62 KB |
| CLI11 | CLI11 dropped (hand-rolled arg parsing) | −144 KB | −151 KB |

### Step 1 — NN serialization

`train` and `infer` save and load the network with mlpack's own neural-network
serialization, exactly as the `mnist_simple_f32` example does: they `#define
MLPACK_ENABLE_ANN_SERIALIZATION` and call `CEREAL_REGISTER_MLPACK_LAYERS(arma::fmat)`,
which registers every mlpack layer type for cereal so an `FFN` can be written and
read back with a single `data::Save`/`data::Load`.

This registration pulls in code for layers the network never uses, which does add
to the binary.  Trimming that is deliberately **not** done in this tutorial —
keeping the standard, copy-pasteable mlpack save/load is worth more here than the
bytes.  A follow-up tutorial will cover shrinking serialization for
resource-constrained targets.

### Step 2 — image/STB ([details below](#disabling-mlpacks-image-stb-support))

`data::Load`/`Save` always instantiate `LoadImage`/`SaveImage`, which pull the
STB PNG/JPEG/resize implementation (~50 KB) even though this example only loads
CSVs. mlpack's opt-out is the CMake option `-DDISABLE_STB=ON`, which defines
`MLPACK_DISABLE_STB`.

### Step 3 — CLI11

mlpack already isolates CLI11 to its `bindings/cli/` (it is **not** pulled by
`<mlpack.hpp>`). The CLI11 in these binaries came from `train`/`infer` using it
for argument parsing; replacing it with a small hand-rolled parser (like
`collect`'s) removed it.

## `.text` breakdown

The category breakdown below was taken from an earlier, size-minimized build that
serialized only the trained weights and rebuilt the network by hand.  This
tutorial deliberately no longer does that — it uses mlpack's full neural-network
serialization (see [Step 1](#step-1-nn-serialization)) — so the **mlpack** row is
now considerably larger (registering every layer type).  The other categories,
and the analysis commands below, are unchanged.

| Category | Size | Removable? |
|----------|-----:|-----------|
| libstdc++ | 429 KB | mostly no — locale facets pulled by cereal/armadillo/iostream |
| musl libc | 291 KB | no — static C runtime |
| CLI11 | 116 KB | yes — the CLI11 change |
| mlpack | (now much larger) | full layer serialization — intentionally kept |
| armadillo | 94 KB | no — core matrix |
| image (STB) | 51 KB | yes — the image/STB change |
| BLAS/LAPACK | 6 KB | no — *not* the bloat, despite the 36 MB archive |

## How to analyse the binary

Tools are in the cross toolchain (`$TC/bin/riscv64-buildroot-linux-musl-*`).

### 1. Section sizes (quick first look)

```sh
TC=/meta/toolchains/riscv64-lp64d--musl--stable-2024.02-1
$TC/bin/riscv64-buildroot-linux-musl-size -A -d train.milkv-riscv64 | sort -k2 -n -r | head
```
`.text` is the code; `.eh_frame` + `.gcc_except_table` are C++ exception unwind
data; `.rodata` constants.

### 2. Attribute `.text` to symbols (the useful view)

The shipped binaries are stripped, so relink **without** `-s` first, then list
symbols by size:

```sh
cd build-milkv
# re-run the link command with -s removed, into a temp file:
sed 's/ -s / /g' CMakeFiles/train.dir/link.txt | sed 's#-o train #-o /tmp/train.uns #' | sh

NM=$TC/bin/riscv64-buildroot-linux-musl-nm
$NM --print-size --size-sort --radix=d /tmp/train.uns | awk '{print $2, $4}' > /tmp/syms.txt

# top individual symbols:
sort -k1 -n -r /tmp/syms.txt | head -25 | awk '{printf "%8.1f KB  %s\n", $1/1024, $2}'
```

### 3. Bucket symbols by library / feature

```sh
awk '
{ s=$1; y=$2; c="other/musl";
  if (y ~ /mlpack/)            c="mlpack";
  else if (y ~ /4arma/)        c="armadillo";
  else if (y ~ /3CLI/)         c="CLI11";
  else if (y ~ /stbi|LoadImage|stb_/) c="image(stb)";
  else if (y ~ /_ZNSt|_ZNKSt|_ZSt/)  c="libstdc++";
  else if (y ~ /_$/)           c="fortran/lapack/blas";
  t[c]+=s }
END { for (k in t) printf "%9.1f KB  %s\n", t[k]/1024, k }' /tmp/syms.txt | sort -k1 -n -r
```

To break mlpack down by feature (ANN layers vs serialization vs image), bucket
on the demangled class names (`BatchNorm|LSTM|MultiheadAttention|…`,
`cereal|binding_name`, `stbi|LoadImage`, …).

## Comparing with and without the image patch

```sh
# WITHOUT STB disabled (default mlpack): build normally, measure.
# WITH it disabled: define MLPACK_DISABLE_STB for the build, then rebuild.
# (This example's CMakeLists already does add_compile_definitions(MLPACK_DISABLE_STB);
#  it is a no-op on an mlpack that predates the switch.)
cd build-milkv && make train infer
$TC/bin/riscv64-buildroot-linux-musl-size -A -d train | grep -E '\.text|Total'
```

A quick host A/B on the mlpack tree, proving the macro removes the codec
entirely:

```sh
g++ -std=c++17 -O2 train/train.cpp -o /tmp/with    -isystem /path/to/mlpack/src -larmadillo
g++ -std=c++17 -O2 -DMLPACK_DISABLE_STB train/train.cpp -o /tmp/without \
    -isystem /path/to/mlpack/src -larmadillo
nm /tmp/with    | grep -c stbi_     # -> 110
nm /tmp/without | grep -c stbi_     # -> 0
```

## Disabling mlpack's image (STB) support

mlpack provides the CMake build option **`-DDISABLE_STB=ON`**, which defines
**`MLPACK_DISABLE_STB`** (gated in `mlpack/config.hpp`). Image support stays on
by default — no behaviour change for existing users. For a header-only consumer
like this example (which does not run mlpack's CMake), the equivalent is the
compile define, so `CMakeLists.txt` uses
`add_compile_definitions(MLPACK_DISABLE_STB)`.

When defined:

- `core/stb/stb.hpp` expands to nothing, so no STB codec is compiled.
- `core/data/load_image.hpp` / `save_image.hpp` provide stubs that return
  `HandleError("... image support was disabled ...")`, so the `data::Load`/`Save`
  dispatch still links.
- `core/data/image_resize_crop.hpp` / `image_letterbox.hpp` (resize/crop/
  letterbox, used by image load and YOLOv3) are compiled out too.

Verified against the upstream `stb_isolation` branch: the macro removes **all**
STB symbols (`nm | grep -c stbi_`: 110 → 0). Measured saving — RISC-V: `train`
−84 KB, `infer` −62 KB; host (x86-64): `train` −136 KB.

## Notes on what was *not* worth changing

- **OpenBLAS/LAPACK is not the bloat** (~38 KB linked; the 36 MB archive is
  almost entirely garbage-collected).
- **Replacing iostream with stdio** would not help: of libstdc++'s 429 KB,
  ~274 KB is locale/format facets that get instantiated whenever *any*
  `std::locale`/stream is constructed — and cereal (model saving), Armadillo
  disk I/O and mlpack logging all do that regardless of our code.

> The staged `*.milkv-riscv64` binaries are built with all three steps, using an
> mlpack that has the `DISABLE_STB` switch (the upstream `stb_isolation` work). A
> clean checkout that fetches an older mlpack without the switch will rebuild
> ~50–80 KB larger, because step 2 becomes a no-op there; steps 1 and 3 are in
> this repo's source and always apply.

---

## Experiment: disabling C++ exceptions (`-fno-exceptions`)

A separate experiment measured how much the C++ exception machinery costs. This
is **not** enabled by default (see "Why it isn't default" below).

### Results (RISC-V, static, stripped; identical source, only the flag differs)

| Binary | WITH exceptions | `-fno-exceptions` | Δ | `-fno-exceptions` + strip dead `.eh_frame` | Δ |
|--------|----------------:|------------------:|----------------:|-------------------------------------------:|------------------:|
| `train` |   843,800 |   806,512 | −37,288 (−4.4%) |   740,816 | −102,984 (−12.2%) |
| `infer` | 1,059,672 | 1,029,944 | −29,728 (−2.8%) |   968,344 |  −91,328 (−8.6%)  |

`train` section breakdown (bytes):

| Section | WITH exceptions | `-fno-exceptions` |
|---------|----------------:|------------------:|
| `.text`             | 654,820 | 639,044 |
| `.eh_frame`         |  74,108 |  56,304 |
| `.gcc_except_table` |  14,701 |   9,733 |

### What had to change

`-fno-exceptions` is a hard error on **every** `throw`/`try`/`catch` the compiler
compiles, so all of them were rewritten to print a message and `std::exit(1)`:

- **Armadillo** — `debug.hpp` (4 `arma_stop_*` sinks), `arma_rng.hpp` (2
  `try/catch`), `diskio_meat.hpp` (28 `try/catch` in load/save).
- **mlpack** — `Log::Fatal` (`prefixedoutstream_impl.hpp`, 2) plus ~296 scattered
  `throw std::...` across the tree.
- **cereal** (pulled in by `<mlpack.hpp>` even though this example no longer
  serializes objects) — 37 `throw Exception` → a `cereal_fail()` print+exit
  helper; its bundled **rapidjson** (assert-as-throw) and **rapidxml** (forced
  into no-exceptions mode); `xml.hpp` `try/catch`.
- **ensmallen** — 40 `throw std::...`.

### Why the saving is modest

`-fno-exceptions` only drops unwind tables from **our** code and the header-only
libraries (mlpack/armadillo/cereal/ensmallen).  Most of the remaining
`.eh_frame` (56 KB in `train`) comes from the **prebuilt static libraries** —
libstdc++, libgfortran, OpenBLAS, musl — which were compiled *with* exceptions.
Stripping that now-dead `.eh_frame`/`.gcc_except_table` post-link recovers the
rest:

```sh
riscv64-...-objcopy --remove-section=.eh_frame --remove-section=.gcc_except_table in out
```

Caveat: with the tables removed, a throw originating *inside* libstdc++/libgfortran
would `abort()` instead of terminating cleanly — fine on a device where errors
just exit, but it is the aggressive option.  Going further would mean rebuilding
libstdc++/musl/OpenBLAS themselves with
`-fno-exceptions -fno-asynchronous-unwind-tables`.

### Why it isn't default

Unlike the localized `DISABLE_STB` switch, disabling exceptions requires patching
four upstream libraries (mlpack, Armadillo, cereal, ensmallen).  Those patches
are not in a clean dependency fetch, so enabling `-fno-exceptions` in a fresh
checkout would fail to compile.  Making it permanent is an upstream-scale effort.

---

## Experiment: dr_libs (audio) footprint — building against dev mlpack

mlpack **4.7.0** (what this example fetches) has no audio support, so its binary
contains **zero** dr_libs symbols.  The **dev** mlpack bundles the audio loaders
(`dr_mp3.h` + `dr_wav.h` in `core/audio/bundled/`), pulled unconditionally by
`core/data/load.hpp` — the audio analog of STB, but with **no `DISABLE` switch**
(only `USE_SYSTEM_DR_LIBS`).

To measure it, `train` was built twice with **identical** cross flags
(`-Os -flto`, static, stripped, exceptions on, `MLPACK_DISABLE_STB`, same
armadillo/ensmallen/cereal/OpenBLAS) — only the mlpack version differs:

| Train build | Size |
|--------------------------------|----------------------|
| mlpack 4.7.0 (current example) | 843,800 B |
| mlpack master + dr_libs        | 917,120 B (+73,320 B) |

`nm` on the (unstripped) dev binary, summing `drwav_`/`drmp3_`/`drflac_` symbols:

```
dr_libs: 106 symbols, 32.0 KB linked into the binary
```

Top contributors: `drmp3dec_decode_frame` (≈7.2 KB), the MP3 L3 Huffman tables
(≈4.3 KB), `drwav__write_or_count_metadata` (≈3.6 KB), `drmp3d_synth_granule`
(≈2.6 KB), …

### Interpretation

- Upgrading the example to the dev mlpack costs **+73 KB** total; **~32 KB of
  that is dr_libs** (MP3 + WAV decoders), the rest (~41 KB) is unrelated
  dev-vs-4.7.0 changes.
- The 32 KB is **not garbage-collected**, even though `train`/`infer` never load
  audio: `data::Load` instantiates the audio-loading path, which references the
  decoder functions — exactly why STB's image code also survives `--gc-sections`.
- It is pure `.text`/`.rodata` dead weight (never executed unless you load
  `.wav`/`.mp3`), so the cost is **binary size, not runtime**.
- A `DISABLE_DR_LIBS` switch mirroring `DISABLE_STB` would reclaim that ~32 KB
  for sensor/CSV-only embedded builds.
