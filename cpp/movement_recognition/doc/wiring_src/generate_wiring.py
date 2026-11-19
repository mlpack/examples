#!/usr/bin/env python3
"""
Wiring diagram for the movement-recognition example: a GY-89 IMU breakout wired
to a Milk-V Duo over I2C0, exactly as described in the README's "Wiring to the
Milk-V Duo" section.

This is a schematic connection diagram (not a photo-real render).  The Duo is a
DIP-style board (like a Raspberry Pi Pico): the 40 pins run sequentially DOWN
each long edge -- pins 1..20 top-to-bottom on the LEFT edge, pins 40..21
top-to-bottom on the RIGHT edge (so pin 1 sits opposite pin 40, near the USB-C).
The layout and every signal name below match the official Milk-V Duo pinout.

Default I2C0 wiring:
    GY-89 SCL -> pin 1  (GP0 / IIC0_SCL)      left edge, top
    GY-89 SDA -> pin 2  (GP1 / IIC0_SDA)      left edge, 2nd
    GY-89 VIN -> pin 36 (3V3 OUT)             right edge, near top
    GY-89 GND -> pin 38 (GND)                 right edge, 2 pins above 3V3

Usage:
    python3 generate_wiring.py            # writes figures/wiring_gy89_duo.png +.svg
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

mpl.rcParams.update({
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})

# Colours for each pin category, matching the official pinout's colour code.
KIND_COL = {
    "gpio":  ("#5a8a3c", "white"),   # GPIO           -> green
    "gnd":   ("#3a3a3a", "white"),   # ground         -> dark grey
    "power": ("#b5321f", "white"),   # VBUS/VSYS/3V3  -> red
    "run":   ("#e6b8c2", "#333"),    # RUN            -> pink
    "nc":    ("#d0d0d0", "#777"),    # not connected  -> light grey
}

# The header, exactly as the board silkscreen reads (see the official pinout).
# Left edge: pins 1..20 top->bottom.  Right edge: pins 40..21 top->bottom.
LEFT = [                              # (pin, label, kind)  row 0 = top
    (1, "GP0", "gpio"), (2, "GP1", "gpio"), (3, "GND", "gnd"),
    (4, "GP2", "gpio"), (5, "GP3", "gpio"), (6, "GP4", "gpio"),
    (7, "GP5", "gpio"), (8, "GND", "gnd"), (9, "GP6", "gpio"),
    (10, "GP7", "gpio"), (11, "GP8", "gpio"), (12, "GP9", "gpio"),
    (13, "GND", "gnd"), (14, "GP10", "gpio"), (15, "GP11", "gpio"),
    (16, "GP12", "gpio"), (17, "GP13", "gpio"), (18, "GND", "gnd"),
    (19, "GP14", "gpio"), (20, "GP15", "gpio"),
]
RIGHT = [                             # row 0 = top (pin 40)
    (40, "VBUS", "power"), (39, "VSYS", "power"), (38, "GND", "gnd"),
    (37, "3V3_EN", "power"), (36, "3V3(OUT)", "power"), (35, "", "nc"),
    (34, "", "nc"), (33, "GND", "gnd"), (32, "GP27", "gpio"),
    (31, "GP26", "gpio"), (30, "RUN", "run"), (29, "GP22", "gpio"),
    (28, "GND", "gnd"), (27, "GP21", "gpio"), (26, "GP20", "gpio"),
    (25, "GP19", "gpio"), (24, "GP18", "gpio"), (23, "GND", "gnd"),
    (22, "GP17", "gpio"), (21, "GP16", "gpio"),
]

# The four connections (GY-89 pad -> Duo pin, wire colour, description).
WIRES = [
    ("VIN", 36, "#d11a1a", "power 3.3 V"),
    ("GND", 38, "#222222", "ground"),
    ("SCL",  1, "#f0a500", "I2C clock"),
    ("SDA",  2, "#1f7a1f", "I2C data"),
]
USED = {pin for _, pin, _, _ in WIRES}


def main():
    outdir = Path("figures")
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.0)
    ax.axis("off")

    # -- GY-89 breakout board (left) ---------------------------------------
    gx, gy, gw, gh = 0.5, 2.5, 2.4, 3.1
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=1.5, edgecolor="#2a4d69", facecolor="#eaf1f7"))
    ax.text(gx + gw / 2, gy + gh + 0.25, "GY-89 IMU breakout",
            ha="center", fontsize=13, fontweight="bold", color="#2a4d69")
    ax.text(gx + gw / 2, gy + gh - 0.22,
            "L3GD20H gyro (0x6b)\nLSM303D accel/mag (0x1d)",
            ha="center", va="top", fontsize=8.5, color="#54708a")

    # Pads live in the lower part of the board so their labels never overlap
    # the two-line chip description printed just under the board title.
    pad_edge = gx + gw
    pad_y = {"VIN": 4.55, "GND": 3.95, "SCL": 3.35, "SDA": 2.75}
    for pad, py in pad_y.items():
        ax.add_patch(Circle((pad_edge, py), 0.07, color="#333", zorder=6))
        ax.text(pad_edge - 0.16, py, pad, ha="right", va="center",
                fontsize=10.5, fontweight="bold", color="#2a4d69")

    # -- Milk-V Duo board (right) ------------------------------------------
    dy = 0.295
    ytop = 6.95
    Lx, Rx = 7.7, 10.9                     # left/right pin-hole edge x
    lab_w, lab_h = 0.98, 0.245
    board_x0, board_x1 = Lx - 0.02, Rx + 0.02
    ax.add_patch(FancyBboxPatch((board_x0, ytop - 19 * dy - 0.55),
                 board_x1 - board_x0, 19 * dy + 1.15,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=1.6, edgecolor="#2f6d2f", facecolor="#e9f3e4",
                 zorder=1))
    ax.text((board_x0 + board_x1) / 2, ytop + 1.55,
            "Milk-V Duo  (40-pin header)", ha="center",
            fontsize=13, fontweight="bold", color="#2f6d2f")
    ax.text((board_x0 + board_x1) / 2, ytop + 0.40, "USB TYPE-C  ▲",
            ha="center", fontsize=7.5, color="#8aa88a")

    hole_xy = {}
    for col, x, inner_dx, ha in ((LEFT, Lx, lab_w / 2 + 0.16, "left"),
                                 (RIGHT, Rx, -(lab_w / 2 + 0.16), "right")):
        for row, (pin, label, kind) in enumerate(col):
            y = ytop - row * dy
            hole_xy[pin] = (x, y)
            used = pin in USED
            # the pin hole at the board edge
            ax.add_patch(Circle((x, y), 0.062, zorder=6,
                         facecolor="#f5f5f5", edgecolor="#555", lw=1))
            # the label box, inboard of the hole
            bx = x + inner_dx - lab_w / 2
            face, txt = KIND_COL[kind]
            ax.add_patch(FancyBboxPatch((bx, y - lab_h / 2), lab_w, lab_h,
                         boxstyle="round,pad=0.01,rounding_size=0.04",
                         facecolor=face, edgecolor="#111" if used else "none",
                         linewidth=1.8 if used else 0, zorder=4))
            if label:
                ax.text(bx + lab_w / 2, y, label, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=txt, zorder=5)
            # pin number, just outboard of the hole
            numx = x - 0.14 if ha == "left" else x + 0.14
            ax.text(numx, y, str(pin), ha="center", va="center",
                    fontsize=6.5, color="#999")

    # -- the four wires: orthogonal (straight + 90-degree corners) ----------
    def elbow(points, colour):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=colour, linewidth=2.6, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")

    # Left-edge pins (SCL->1, SDA->2): come in horizontally from the left.
    elbow([(pad_edge, pad_y["SCL"]), (5.5, pad_y["SCL"]),
           (5.5, hole_xy[1][1]), (Lx, hole_xy[1][1])], "#f0a500")
    elbow([(pad_edge, pad_y["SDA"]), (5.9, pad_y["SDA"]),
           (5.9, hole_xy[2][1]), (Lx, hole_xy[2][1])], "#1f7a1f")
    # Right-edge pins (VIN->36, GND->38): route up over the top of the board
    # and down the right side, entering each hole from the right.
    elbow([(pad_edge, pad_y["VIN"]), (6.6, pad_y["VIN"]), (6.6, 7.75),
           (11.5, 7.75), (11.5, hole_xy[36][1]), (Rx, hole_xy[36][1])],
          "#d11a1a")
    elbow([(pad_edge, pad_y["GND"]), (6.3, pad_y["GND"]), (6.3, 7.95),
           (11.75, 7.95), (11.75, hole_xy[38][1]), (Rx, hole_xy[38][1])],
          "#222222")

    # -- legend -------------------------------------------------------------
    lx, ly = 3.5, 1.75
    ax.text(lx, ly + 0.5, "Connections (default /dev/i2c-0)",
            ha="left", fontsize=11, fontweight="bold", color="#222")
    for i, (pad, pin, colour, desc) in enumerate(WIRES):
        y = ly - i * 0.34
        ax.plot([lx, lx + 0.45], [y, y], color=colour, lw=3, zorder=3)
        ax.text(lx + 0.6, y, f"{pad}  →  pin {pin}   ({desc})",
                ha="left", va="center", fontsize=9.5, color="#222")
    ax.text(lx, ly - 4 * 0.34 - 0.05,
            "I2C1 alternative: SCL→pin 6 (GP4), SDA→pin 7 (GP5),\n"
            "then --device /dev/i2c-1.  Mux the pins with duo-pinmux first.",
            ha="left", va="top", fontsize=8, color="#666", style="italic")

    for ext in ("png", "svg"):
        fig.savefig(outdir / f"wiring_gy89_duo.{ext}", bbox_inches="tight")
        print(f"  wrote wiring_gy89_duo.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
