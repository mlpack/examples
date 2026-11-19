# GY-89 ↔ Milk-V Duo wiring diagram

`generate_wiring.py` draws the connection diagram for wiring the GY-89 IMU
breakout to the Milk-V Duo over I²C0, matching the README's "Wiring to the
Milk-V Duo" table.

It is a schematic diagram (drawn from the documented pinout), not a photo of the
board: the GY-89's four pads, the Duo's 40-pin header at real physical pin
positions, and the four wires between them.

Pin positions are from the official Milk-V Duo pinout:
- `VIN`  → pin 36 `3V3(OUT)`
- `GND`  → pin 3 `GND`
- `SCL`  → pin 1 `GP0 / IIC0_SCL`
- `SDA`  → pin 2 `GP1 / IIC0_SDA`

I²C1 alternative: `SCL` → pin 6 (`GP4`), `SDA` → pin 7 (`GP5`), then pass
`--device /dev/i2c-1` (mux the pins with `duo-pinmux` first).

## Regenerate

```sh
# uses the shared venv created for the FFT figure
../fft_pipeline_src/.venv/bin/python generate_wiring.py
```

Outputs `figures/wiring_gy89_duo.png` (300 dpi) and `.svg`.
