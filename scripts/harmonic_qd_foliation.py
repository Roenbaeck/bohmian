"""Generate a placeholder quadratic-differential foliation figure for the draft.

This uses the harmonic-oscillator first excited state (up to a constant)

  psi(z) ∝ z * exp(-z^2/2)

which is entire, so log psi has a branch point at z=0 and

  w(z) := d^2/dz^2 log psi(z) = -1 - 1/z^2.

Horizontal trajectories of w(z) dz^2 satisfy arg(dz) = -0.5 arg(w(z)) (mod π).
We visualise this by integrating the direction field

  dz/dτ = exp(i * θ(z)),   θ(z) = -0.5 * arg(w(z)).

The result is a qualitative organiser of the node/pole neighbourhood; it is not
(yet) a Bohmian trajectory plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def w_of_z(z: complex) -> complex:
    if z == 0:
        return complex("nan")
    return -1.0 - 1.0 / (z * z)


def theta_of_z(z: complex) -> float:
    w = w_of_z(z)
    return -0.5 * math.atan2(w.imag, w.real)


@dataclass(frozen=True)
class IntegratorConfig:
    step: float = 0.02
    n_steps: int = 4000
    escape_radius: float = 3.5
    min_radius: float = 0.08


def rk4_step(z: complex, h: float) -> complex:
    def v(zz: complex) -> complex:
        th = theta_of_z(zz)
        return complex(math.cos(th), math.sin(th))

    k1 = v(z)
    k2 = v(z + 0.5 * h * k1)
    k3 = v(z + 0.5 * h * k2)
    k4 = v(z + h * k3)
    return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_streamline(z0: complex, *, direction: int, cfg: IntegratorConfig) -> np.ndarray:
    h = float(direction) * cfg.step
    points: list[complex] = []
    z = z0

    for _ in range(cfg.n_steps):
        r = abs(z)
        if r > cfg.escape_radius:
            break
        if r < cfg.min_radius:
            break
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            break

        points.append(z)
        z = rk4_step(z, h)

    if not points:
        return np.empty((0, 2), dtype=float)

    arr = np.array([(p.real, p.imag) for p in points], dtype=float)
    return arr


def main() -> None:
    out_path = "figures/bohm_foliation_harmonic.pdf"

    # Background field: log|w(z)| on a grid, with a mask near the pole at z=0.
    x = np.linspace(-3.0, 3.0, 600)
    y = np.linspace(-3.0, 3.0, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        W = -1.0 - 1.0 / (Z * Z)
        R = np.abs(Z)
        log_abs_w = np.log(np.abs(W))
        log_abs_w = np.where(R < 0.08, np.nan, log_abs_w)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    im = ax.imshow(
        log_abs_w,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="viridis",
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log|w(z)|$", rotation=90)

    cfg = IntegratorConfig()

    # Seed points on circles around the pole.
    radii = [0.18, 0.28, 0.42, 0.62, 0.9, 1.3, 1.8, 2.4]
    n_angles = 16

    for r in radii:
        for k in range(n_angles):
            angle = 2.0 * math.pi * (k / n_angles)
            z0 = complex(r * math.cos(angle), r * math.sin(angle))

            pts_fwd = integrate_streamline(z0, direction=+1, cfg=cfg)
            pts_bwd = integrate_streamline(z0, direction=-1, cfg=cfg)

            if pts_fwd.shape[0] > 2:
                ax.plot(pts_fwd[:, 0], pts_fwd[:, 1], color="white", lw=0.6, alpha=0.85)
            if pts_bwd.shape[0] > 2:
                ax.plot(pts_bwd[:, 0], pts_bwd[:, 1], color="white", lw=0.6, alpha=0.85)

    # Mark the pole/node and the real axis (configuration line).
    ax.plot([x.min(), x.max()], [0.0, 0.0], color="crimson", lw=1.2, alpha=0.9)
    ax.scatter([0.0], [0.0], s=30, color="black", zorder=5)

    # Mark the simple zeros of w(z) = -1 - 1/z^2 at z = ± i.
    ax.scatter([0.0, 0.0], [1.0, -1.0], s=22, color="white", edgecolor="black", linewidth=0.6, zorder=6)
    ax.text(0.06, 1.02, r"$w=0$", color="white", fontsize=9, ha="left", va="bottom")
    ax.text(0.06, -0.98, r"$w=0$", color="white", fontsize=9, ha="left", va="top")

    ax.set_title(r"Quadratic differential foliation for $w(z)=\partial_z^2\log(z e^{-z^2/2})=-1-1/z^2$")
    ax.set_xlabel(r"$\Re z$")
    ax.set_ylabel(r"$\Im z$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
