#!/usr/bin/env python3
"""
@file: smooth_trajectory.py
@breif: Suavizado (B-Spline parametrica centripeta) de la trayectoria cruda
        de Theta* generada por generate_trajectory.py sobre el mapa SLAM de
        AutoDRIVE.

Reutiliza plan_track_path() de generate_trajectory.py (mismos checkpoints,
mismo sesgo al centro, mismo punto de spawn como inicio/fin) para obtener el
camino crudo, y smooth_path_safe() de planning_utils.py (B-Spline de
interpolacion con reintento verificando colisiones, cada vez con mas puntos
de control hasta que la curva completa no caiga sobre ningun obstaculo) para
suavizarlo.

Guarda:
  - waypoints/theta_star_waypoints_smooth.csv  (trayectoria final, la que
    publica global_path_publisher.py para la visualizacion en el simulador)
  - waypoints/theta_star_waypoints_smooth.png  (overlay solo)
  - waypoints/comparacion_crudo_vs_suavizado.png (crudo vs. suavizado)
"""
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planning_utils import (  # noqa: E402
    smooth_path_safe, resample_by_arclength, open_loop_gap, save_waypoints_csv,
)
from generate_trajectory import plan_track_path, save_overlay_plot, SPAWN_XY  # noqa: E402


def save_comparison_plot(map_bin, resolution, origin, raw_waypoints, smooth_waypoints, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = map_bin.shape
    extent = [origin[0], origin[0] + w * resolution, origin[1], origin[1] + h * resolution]

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    for ax, pts, title in zip(axes, [raw_waypoints, smooth_waypoints],
                              ["Theta* crudo", "Theta* suavizado (B-Spline)"]):
        ax.imshow(1 - map_bin, cmap="gray", extent=extent)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color="red", markersize=2, linewidth=1)
        ax.plot(xs[0], ys[0], "og", markersize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    import argparse

    share = Path(get_package_share_directory('global_planner'))

    parser = argparse.ArgumentParser(description="Suavizado B-Spline de la trayectoria Theta* (AutoDRIVE)")
    parser.add_argument("--map", default=str(share / "maps" / "F1tenth_Map.yaml"))
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--penalty-weight", type=float, default=3.0)
    parser.add_argument("--spacing", type=float, default=0.3)
    parser.add_argument("--smooth-step", type=float, default=0.0008,
                         help="Densidad de la curva suavizada (1/step puntos sobre la vuelta)")
    parser.add_argument("--start-x", type=float, default=SPAWN_XY[0],
                         help="Punto de inicio/fin de la vuelta -- coordenada X del spawn del auto en AutoDRIVE")
    parser.add_argument("--start-y", type=float, default=SPAWN_XY[1],
                         help="Punto de inicio/fin de la vuelta -- coordenada Y del spawn del auto en AutoDRIVE")
    parser.add_argument("--gap", type=float, default=2.0,
                         help="Separacion [m] entre el final y el inicio de la vuelta (no la cierra del todo)")
    parser.add_argument("--outdir", default=str(share / "waypoints"))
    args = parser.parse_args()

    world_path, map_bin, resolution, origin = plan_track_path(
        args.map, args.checkpoints, args.segments, args.penalty_weight,
        start_xy=(args.start_x, args.start_y))
    print(f"Theta* crudo: {len(world_path)} puntos")

    raw_waypoints = resample_by_arclength(world_path, args.spacing)
    raw_waypoints = open_loop_gap(raw_waypoints, args.spacing, gap=args.gap)

    smoothed, mcp_used = smooth_path_safe(world_path, map_bin, resolution, origin, step=args.smooth_step)
    print(f"Theta* suavizado: {len(smoothed)} puntos (max_control_points={mcp_used})")

    smooth_waypoints = resample_by_arclength(smoothed, args.spacing)
    smooth_waypoints = open_loop_gap(smooth_waypoints, args.spacing, gap=args.gap)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "theta_star_waypoints_smooth.csv"
    save_waypoints_csv(smooth_waypoints, csv_path)

    png_path = outdir / "theta_star_waypoints_smooth.png"
    save_overlay_plot(map_bin, resolution, origin, smooth_waypoints, png_path,
                       f"Theta* suavizado - AutoDRIVE SLAM map ({len(smooth_waypoints)} waypoints)")

    comparison_png = outdir / "comparacion_crudo_vs_suavizado.png"
    save_comparison_plot(map_bin, resolution, origin, raw_waypoints, smooth_waypoints, comparison_png)

    print(f"{len(smooth_waypoints)} waypoints suavizados -> {csv_path}")
    print(f"Overlay -> {png_path}")
    print(f"Comparacion -> {comparison_png}")


if __name__ == "__main__":
    main()
