"""Symmetry-broken supporting example: shifted vortex.

We use
  psi(x,y) = (z - z0) * exp(-(x^2 + y^2)/2),   z=x+iy,
with z0 != 0 so the node is off-centre relative to the Gaussian envelope.

This breaks the obvious rotational symmetry about the origin but keeps a
nonzero circulating probability current around the node.

We compare:
  J = Im(conj(psi) grad psi)
against the quadratic-differential direction field derived from
  w(z) = ∂_z^2 log psi(z).

For this psi one has analytically w(z) = -1/(z-z0)^2 (Gaussian term drops), so
this is a clean robustness check of the numerical pipeline away from the
origin-centred case.

Outputs:
  figures/qd_vs_current_shifted_vortex.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Grid:
    x_min: float = -3.0
    x_max: float = 3.0
    y_min: float = -3.0
    y_max: float = 3.0
    n: int = 401


@dataclass(frozen=True)
class TraceConfig:
    step: float = 0.02
    n_steps: int = 6000
    stop_radius: float = 0.10


def grad_f(f: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    df_dy, df_dx = np.gradient(f, dy, dx, edge_order=2)
    return df_dx, df_dy


def unitize(vx: np.ndarray, vy: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.sqrt(vx * vx + vy * vy)
    inv = np.zeros_like(speed)
    np.divide(1.0, speed, out=inv, where=speed > eps)
    return vx * inv, vy * inv, speed


def angle_mod_pi(a: np.ndarray) -> np.ndarray:
    return (a + 0.5 * np.pi) % np.pi - 0.5 * np.pi


def main() -> None:
    out_path = "figures/qd_vs_current_shifted_vortex.pdf"
    grid = Grid()
    cfg = TraceConfig()

    z0 = complex(1.0, 0.6)

    x = np.linspace(grid.x_min, grid.x_max, grid.n)
    y = np.linspace(grid.y_min, grid.y_max, grid.n)
    X, Y = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    Z = X + 1j * Y
    r2 = X * X + Y * Y
    psi = (Z - z0) * np.exp(-0.5 * r2)

    # Current
    psi_x, psi_y = grad_f(psi, dx, dy)
    Jx = np.imag(np.conj(psi) * psi_x)
    Jy = np.imag(np.conj(psi) * psi_y)
    Jx_u, Jy_u, J_speed = unitize(Jx, Jy)

    # Analytic w
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        w = -1.0 / ((Z - z0) ** 2)

    theta_qd = -0.5 * np.angle(w)
    Uq = np.cos(theta_qd)
    Vq = np.sin(theta_qd)
    Uq, Vq, _ = unitize(Uq, Vq)

    Rpsi = np.abs(psi)
    node_mask = Rpsi < 5e-3
    valid = (~node_mask) & np.isfinite(w.real) & np.isfinite(w.imag) & (J_speed > 1e-6)

    ang_J = np.arctan2(Jy_u, Jx_u)
    ang_Q = np.arctan2(Vq, Uq)
    d_valid = angle_mod_pi(ang_J - ang_Q)[valid]

    abs_d_deg = np.abs(d_valid) * (180.0 / math.pi)
    rms_deg = float(np.sqrt(np.mean(d_valid**2)) * (180.0 / math.pi))
    p95_deg = float(np.quantile(abs_d_deg, 0.95))
    max_deg = float(np.max(abs_d_deg))

    alpha = float(np.mean(np.abs(np.sin(d_valid))))

    q99 = float(np.quantile(abs_d_deg, 0.99))
    x_max = float(np.clip(3.0 * q99, 0.05, 30.0))
    in_range = abs_d_deg <= x_max
    tail_frac = float(1.0 - np.mean(in_range))

    # --- streamline tracer ---
    def in_bounds(px: float, py: float) -> bool:
        return (grid.x_min <= px <= grid.x_max) and (grid.y_min <= py <= grid.y_max)

    def dist_to_node(px: float, py: float) -> float:
        return math.hypot(px - z0.real, py - z0.imag)

    def bilinear_sample(field: np.ndarray, px: float, py: float) -> float:
        if not in_bounds(px, py):
            return 0.0
        u = (px - grid.x_min) / dx
        v = (py - grid.y_min) / dy
        i = int(math.floor(u))
        j = int(math.floor(v))
        if i < 0 or j < 0 or i >= grid.n - 1 or j >= grid.n - 1:
            return 0.0
        a0 = u - i
        b0 = v - j
        f00 = field[j, i]
        f10 = field[j, i + 1]
        f01 = field[j + 1, i]
        f11 = field[j + 1, i + 1]
        return (1 - a0) * (1 - b0) * f00 + a0 * (1 - b0) * f10 + (1 - a0) * b0 * f01 + a0 * b0 * f11

    def v_field(px: float, py: float, U: np.ndarray, V: np.ndarray) -> tuple[float, float]:
        ux = bilinear_sample(U, px, py)
        uy = bilinear_sample(V, px, py)
        s = math.hypot(ux, uy)
        if s <= 1e-12:
            return 0.0, 0.0
        return ux / s, uy / s

    def rk4_step(px: float, py: float, h: float, U: np.ndarray, V: np.ndarray) -> tuple[float, float]:
        k1x, k1y = v_field(px, py, U, V)
        k2x, k2y = v_field(px + 0.5 * h * k1x, py + 0.5 * h * k1y, U, V)
        k3x, k3y = v_field(px + 0.5 * h * k2x, py + 0.5 * h * k2y, U, V)
        k4x, k4y = v_field(px + h * k3x, py + h * k3y, U, V)
        return (
            px + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x),
            py + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y),
        )

    def trace(seed: tuple[float, float], direction: int, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        h = float(direction) * cfg.step
        px, py = seed
        pts: list[tuple[float, float]] = []
        for _ in range(cfg.n_steps):
            if not in_bounds(px, py):
                break
            if dist_to_node(px, py) < cfg.stop_radius:
                break
            if not (math.isfinite(px) and math.isfinite(py)):
                break
            pts.append((px, py))
            px, py = rk4_step(px, py, h, U, V)
        if not pts:
            return np.empty((0, 2), dtype=float)
        return np.asarray(pts, dtype=float)

    # --- plot ---
    fig = plt.figure(figsize=(10.5, 4.5))

    ax0 = fig.add_subplot(1, 2, 1)
    ax0.set_title("Shifted vortex: current streamlines vs QD horizontal field")

    with np.errstate(divide="ignore", invalid="ignore"):
        bg = np.log(np.abs(psi))

    im = ax0.imshow(
        bg,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="magma",
        interpolation="bilinear",
        vmin=-8,
        vmax=0,
    )
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cb.set_label(r"$\log|\psi|$", rotation=90)

    radii = [0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.5]
    n_angles = 20

    for r in radii:
        for k in range(n_angles):
            ang = 2.0 * math.pi * (k / n_angles)
            seed = (r * math.cos(ang), r * math.sin(ang))

            for sgn in (+1, -1):
                pts = trace(seed, sgn, Jx_u, Jy_u)
                if pts.shape[0] > 2:
                    ax0.plot(pts[:, 0], pts[:, 1], color="cyan", lw=0.8, alpha=0.85)

            for sgn in (+1, -1):
                pts = trace(seed, sgn, Uq, Vq)
                if pts.shape[0] > 2:
                    ax0.plot(pts[:, 0], pts[:, 1], color="white", lw=0.55, alpha=0.8)

    ax0.scatter([z0.real], [z0.imag], s=35, color="black", zorder=6)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlim(x.min(), x.max())
    ax0.set_ylim(y.min(), y.max())
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title("Angle misalignment (mod π)")
    bins = np.linspace(0.0, x_max, 90)
    ax1.hist(abs_d_deg[in_range], bins=bins, color="#4C72B0", alpha=0.9)
    ax1.set_xlabel("|Δθ| (degrees)")
    ax1.set_ylabel("count")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(0.0, x_max)
    ax1.text(
        0.98,
        0.98,
        f"zoom: [0, {x_max:.3g}]°\n"
        f"tail > {x_max:.3g}°: {tail_frac*100:.3g}%\n"
        f"max: {max_deg:.3g}°",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    fig.suptitle(
        rf"QD vs current on ψ=(z-z0)e^{{-|z|^2/2}}, z0={z0.real:g}+{z0.imag:g}i   RMS={rms_deg:.2f}°   95%={p95_deg:.2f}°",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Wrote {out_path}")
    print(
        f"alpha=<|sin(Δθ)|>={alpha:.6g}  RMS(deg)={rms_deg:.6g}  P95(deg)={p95_deg:.6g}  "
        f"max(deg)={max_deg:.6g}  tail%={tail_frac*100:.6g}"
    )


if __name__ == "__main__":
    main()
