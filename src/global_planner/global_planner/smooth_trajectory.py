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

Como cada uno de los 8 tramos se planifica por separado (ver
plan_track_path()), la direccion de llegada a un checkpoint y la direccion
de salida del siguiente tramo no tienen por que coincidir -- en dos de los
checkpoints de esta pista, ambos caen sobre curvas muy cerradas, eso se nota
como un vertice/pico marcado en la trayectoria suavizada, porque la spline
de interpolacion esta obligada a pasar exactamente por ese punto. Por eso,
antes de la B-Spline se aplica un pre-redondeo de esquinas (algoritmo de
Chaikin, chaikin_smooth() mas abajo) sobre el camino crudo completo:
reemplaza cada vertice por dos puntos mas cercanos al segmento que lo forma,
lo que suaviza cualquier vertice marcado (incluidos los empalmes entre
tramos) antes de que la spline tenga que pasar por ahi. Solo se usa para la
trayectoria suavizada -- la comparacion "cruda" sigue siendo la salida cruda
real de Theta*, sin este paso.

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


def chaikin_smooth(path_xy, iterations=2):
    """Chaikin corner-cutting, keeping both endpoints exactly fixed: each
    iteration replaces every interior vertex pair (p0, p1) with two new
    points at 1/4 and 3/4 along the segment between them. path_xy is a full
    lap that starts and ends at the same point (the vehicle's spawn,
    anchor_checkpoints() in generate_trajectory.py) -- that point must stay
    exact (the delivered trajectory is required to start precisely where the
    car spawns), so this does NOT wrap around and round it like any other
    vertex; only the checkpoint junctions strictly *between* start and end
    get rounded. Each new point is a convex combination of two original,
    consecutive points, so the rounded polyline never swings wider than the
    original one did -- it only cuts corners inward, it can't introduce a
    new collision that wasn't already there (smooth_path_safe() still
    re-verifies this afterwards regardless).
    """
    pts = list(path_xy)
    for _ in range(iterations):
        n = len(pts)
        new_pts = [pts[0]]
        for i in range(n - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            new_pts.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            new_pts.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


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
    parser.add_argument("--spacing", type=float, default=0.15,
                         help="Espaciado final [m]. Mas fino que el de la ruta cruda a proposito: en la "
                              "curva mas cerrada, un espaciado grueso deja muy pocos waypoints para "
                              "representar el arco y las lineas rectas entre ellos se ven como un vertice, "
                              "aun cuando la curva de fondo (B-Spline) sea continua")
    parser.add_argument("--smooth-step", type=float, default=0.0008,
                         help="Densidad de la curva suavizada (1/step puntos sobre la vuelta)")
    parser.add_argument("--start-x", type=float, default=SPAWN_XY[0],
                         help="Punto de inicio/fin de la vuelta -- coordenada X del spawn del auto en AutoDRIVE")
    parser.add_argument("--start-y", type=float, default=SPAWN_XY[1],
                         help="Punto de inicio/fin de la vuelta -- coordenada Y del spawn del auto en AutoDRIVE")
    parser.add_argument("--gap", type=float, default=2.0,
                         help="Separacion [m] entre el final y el inicio de la vuelta (no la cierra del todo)")
    parser.add_argument("--corner-rounding-iterations", type=int, default=2,
                         help="Iteraciones de pre-redondeo de esquinas (Chaikin) antes de la B-Spline; "
                              "0 lo desactiva")
    parser.add_argument("--outdir", default=str(share / "waypoints"))
    args = parser.parse_args()

    world_path, map_bin, resolution, origin = plan_track_path(
        args.map, args.checkpoints, args.segments, args.penalty_weight,
        start_xy=(args.start_x, args.start_y))
    print(f"Theta* crudo: {len(world_path)} puntos")

    raw_waypoints = resample_by_arclength(world_path, args.spacing)
    raw_waypoints = open_loop_gap(raw_waypoints, args.spacing, gap=args.gap)

    rounded_path = world_path
    if args.corner_rounding_iterations > 0:
        rounded_path = chaikin_smooth(world_path, iterations=args.corner_rounding_iterations)
        print(f"Pre-redondeo de esquinas (Chaikin): {len(world_path)} -> {len(rounded_path)} puntos")

    smoothed, mcp_used = smooth_path_safe(rounded_path, map_bin, resolution, origin, step=args.smooth_step)
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
