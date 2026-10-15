# Movement recognition on a Milk-V Duo with mlpack

On-device human movement recognition (walking, sitting, squats, stairs) from a
GY-89 9-DOF IMU on a Milk-V Duo, using mlpack.  Four small programs: `imu_test`
(sensor check), `collect` (record CSV), `train`, and `infer`.

**Full walkthrough:** see the tutorial at
<https://www.mlpack.org/doc/tutorials/movement_recognition.html>.

## Wiring

Wire the GY-89 to the Milk-V Duo's I²C0 bus (the default `/dev/i2c-0`):

| GY-89 pin | Milk-V Duo        |
|-----------|-------------------|
| VIN/VCC   | 3V3(OUT), pin 36  |
| GND       | GND, pin 38       |
| SCL       | IIC0_SCL (GP0), pin 1 |
| SDA       | IIC0_SDA (GP1), pin 2 |

<center>
<img src="doc/wiring_src/figures/wiring_gy89_duo.png" width="620" alt="Wiring the GY-89 IMU breakout to the Milk-V Duo over I2C0: SCL to pin 1 (GP0), SDA to pin 2 (GP1), VIN to pin 36 (3V3 out), and GND to pin 38" />
</center>

GP0/GP1 usually default to another function, so mux them to I2C first:

```sh
duo-pinmux -p GP0 -f IIC0_SCL
duo-pinmux -p GP1 -f IIC0_SDA
i2cdetect -y -r 0             # should show devices at 0x1d and 0x6b
```

## Building

All of the Four binaries can be built with one CMake project, `imu_test`
and `collect` need only the Linux I2C headers, While `train` and `infer`
link against mlpack and its dependencies. All of them are fetched by CMake.

### Host build

```sh
mkdir build && cd build
cmake ..
make                          # builds imu_test, collect, train, infer
```

### Cross-compiling for the Duo

The Duo runs the code on-device, so cross-compile with a `riscv64-lp64d` **musl**
toolchain from [toolchains.bootlin.com](https://toolchains.bootlin.com/):

```sh
mkdir build && cd build
TC=/path/to/riscv64-lp64d--musl--stable-2025.08-1
cmake -DCMAKE_CROSSCOMPILING=ON -DARCH_NAME=RV64GCV \
      -DCMAKE_TOOLCHAIN_FILE=../CMake/crosscompile-toolchain.cmake \
      -DTOOLCHAIN_PREFIX=$TC/bin/riscv64-buildroot-linux-musl- \
      -DCMAKE_SYSROOT=$TC/riscv64-buildroot-linux-musl/sysroot \
      -DOPENBLAS_PATCHES=../CMake/patches/openblas-riscv64-low-memory.patch ..
make                          # -> static build/{imu_test,collect,train,infer}
```

Copy the static binaries to the board:

```sh
scp -O imu_test collect train infer  root@192.168.42.1:/root/
```
