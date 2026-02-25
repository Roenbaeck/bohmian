"""Symmetry-broken experimental test: two-node state.

We use a simple non-rotationally-symmetric state with two nodes,

  psi(x,y) = (z^2 - a^2) * exp(-(x^2 + y^2)/2),   z=x+iy,

which has zeros at z=±a on the real axis and a nontrivial probability current
associated to the phase arg(z^2-a^2).

We compare probability-current streamlines to the quadratic-differential
horizontal direction field defined from

  w(z) := ∂_z^2 log psi(z).

For this psi, the Gaussian factor contributes nothing to w, and one can compute
w analytically:

  w(z) = (log(z^2-a^2))'' = -2 (z^2 + a^2) / (z^2 - a^2)^2.

We visualise both direction fields and quantify the angular misalignment
(mod π), with an automatically zoomed histogram scale.

Outputs:
  figures/qd_vs_current_two_nodes.pdf

Dependencies:
  numpy, matplotlib
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
    out_path = "figures/qd_vs_current_two_nodes.pdf"
    grid = Grid()
    cfg = TraceConfig()

    a = 1.0
    node_centers = [(+a, 0.0), (-a, 0.0)]

    x = np.linspace(grid.x_min, grid.x_max, grid.n)
    y = np.linspace(grid.y_min, grid.y_max, grid.n)
    X, Y = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    Z = X + 1j * Y
    r2 = X * X + Y * Y
    psi = (Z * Z - a * a) * np.exp(-0.5 * r2)

    # Probability current J = Im(conj(psi) grad psi)
    psi_x, psi_y = grad_f(psi, dx, dy)
    Jx = np.imag(np.conj(psi) * psi_x)
    Jy = np.imag(np.conj(psi) * psi_y)
    Jx_u, Jy_u, J_speed = unitize(Jx, Jy)

    # Analytic w(z) = ∂_z^2 log psi(z) = (log(z^2-a^2))''
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        w = -2.0 * (Z * Z + a * a) / ((Z * Z - a * a) ** 2)

    theta_h = -0.5 * np.angle(w)  # horizontal family (mod π)
    theta_v = theta_h + 0.5 * np.pi  # vertical family

    Uh = np.cos(theta_h)
    Vh = np.sin(theta_h)
    Uh, Vh, _ = unitize(Uh, Vh)

    Uv = np.cos(theta_v)
    Vv = np.sin(theta_v)
    Uv, Vv, _ = unitize(Uv, Vv)

    # Valid comparison region: away from nodes/poles and away from zero-current zones.
    Rpsi = np.abs(psi)
    node_mask = Rpsi < 5e-3
    valid = (~node_mask) & np.isfinite(w.real) & np.isfinite(w.imag) & (J_speed > 1e-6)

    ang_J = np.arctan2(Jy_u, Jx_u)
    ang_Qh = np.arctan2(Vh, Uh)
    ang_Qv = np.arctan2(Vv, Uv)

    d_h = angle_mod_pi(ang_J - ang_Qh)[valid]
    d_v = angle_mod_pi(ang_J - ang_Qv)[valid]

    rms_h = float(np.sqrt(np.mean(d_h**2)) * (180.0 / math.pi))
    rms_v = float(np.sqrt(np.mean(d_v**2)) * (180.0 / math.pi))

    use_vertical = rms_v < rms_h
    d_valid = d_v if use_vertical else d_h
    Uq, Vq = (Uv, Vv) if use_vertical else (Uh, Vh)
    family = "vertical" if use_vertical else "horizontal"

    if d_valid.size == 0:
        raise RuntimeError("No valid points for comparison; adjust masks/resolution")

    abs_d_deg = np.abs(d_valid) * (180.0 / math.pi)
    rms_deg = float(np.sqrt(np.mean(d_valid**2)) * (180.0 / math.pi))
    p95_deg = float(np.quantile(abs_d_deg, 0.95))
    max_deg = float(np.max(abs_d_deg))

    alpha = float(np.mean(np.abs(np.sin(d_valid))))

    q99 = float(np.quantile(abs_d_deg, 0.99))
    x_max = float(np.clip(3.0 * q99, 0.05, 30.0))
    in_range = abs_d_deg <= x_max
    tail_frac = float(1.0 - np.mean(in_range))

    # --- streamline tracer on precomputed unit direction fields ---
    def in_bounds(px: float, py: float) -> bool:
        return (grid.x_min <= px <= grid.x_max) and (grid.y_min <= py <= grid.y_max)

    def min_dist_to_nodes(px: float, py: float) -> float:
        return min(math.hypot(px - cx, py - cy) for (cx, cy) in node_centers)

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
            if min_dist_to_nodes(px, py) < cfg.stop_radius:
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
    ax0.set_title(f"Two-node state: current streamlines vs QD {family} field")

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

    # Seed streamlines on circles around the origin (covers multiple flow regions).
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

    ax0.scatter([+a, -a], [0.0, 0.0], s=35, color="black", zorder=6)
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
        rf"QD vs current on ψ=(z^2-a^2)e^{{-|z|^2/2}}, a={a:g} ({family})   RMS={rms_deg:.2f}°   95%={p95_deg:.2f}°",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Wrote {out_path}")
    print(
        f"family={family}  rms_h(deg)={rms_h:.4g}  rms_v(deg)={rms_v:.4g}  "
        f"alpha=<|sin(Δθ)|>={alpha:.6g}  RMS(deg)={rms_deg:.6g}  P95(deg)={p95_deg:.6g}  "
        f"max(deg)={max_deg:.6g}  tail%={tail_frac*100:.6g}"
    )


if __name__ == "__main__":
    main()
