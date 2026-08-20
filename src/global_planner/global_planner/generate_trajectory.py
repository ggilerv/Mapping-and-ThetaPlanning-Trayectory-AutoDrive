#!/usr/bin/env python3
"""
@file: generate_trajectory.py
@breif: Offline Theta* global trajectory generation over the SLAM-built
        occupancy map (maps/F1tenth_Map.yaml), for the AutoDRIVE mapping +
        planning project.

This is a plain script, not a spinning ROS 2 node: it runs once (via
`ros2 run global_planner generate_trajectory`) to produce a waypoint CSV
(waypoints/theta_star_waypoints.csv), which the ROS 2 node
(global_path_publisher.py) then just loads and republishes.

Reuses CenteredThetaStar and the ring/checkpoint/full-lap machinery from
planning_utils.py (a self-contained module vendored in this same package,
along with the python_motion_planning library it depends on) instead of
reimplementing Theta* from scratch.

The map loader here differs from planning_utils.load_map() in how the
occupancy grid is read: F1tenth_Map.pgm is a *trinary* SLAM map (occupied /
free / unknown), unlike a plain black-and-white image. The pixel values are
exactly 0 (occupied), 205 (unknown) and 254 (free) -- confirmed by
inspecting the raw file, not assumed -- so only 254 counts as drivable free
space here; unknown cells (unexplored by SLAM) are treated as *not* free,
same as occupied ones, since the planner has no actual evidence the car can
drive there.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planning_utils import (  # noqa: E402
    grid_from_map, build_clearance_grid, CenteredThetaStar,
    world_to_map, map_to_world, is_free_cell,
    order_checkpoints_by_track, concat_segments,
    resample_by_arclength, open_loop_gap, save_waypoints_csv,
)

# Vehicle spawn point in AutoDRIVE (read from /autodrive/f1tenth_1/ips right
# after connecting, before driving) -- the route is anchored to start/end
# here instead of an arbitrary angularly-placed checkpoint, so it lines up
# with where the car actually appears in the simulator.
SPAWN_XY = (0.741, 3.158)


def find_track_mask(map_bin):
    """Identify the drivable track as the LARGEST connected free-space
    component, unlike planning_utils.find_ring_mask() (which picks the
    *lowest fill-ratio* component -- the right heuristic for a synthetic
    racetrack image that always has exactly three sizeable free regions:
    infield, outfield and the thin ring between them, with the ring
    reliably the thinnest).

    A real SLAM-built trinary map has no such infield/outfield split (the
    "unknown" class already keeps everything outside the explored track out
    of the free mask), but it does have dozens of tiny 1-40 px noise specks
    from sensor artifacts -- and at least one of those, in this particular
    map, happens to have a *lower* fill ratio than the real track, which
    would fool the lowest-fill-ratio heuristic. The real track is ~65x
    larger than the next-biggest connected component here, so picking the
    largest area is unambiguous and far more robust for this kind of map.
    """
    free = (map_bin == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free, connectivity=4)
    if num_labels < 2:
        raise RuntimeError("No se encontraron regiones libres en el mapa.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    track_label = 1 + int(np.argmax(areas))
    return labels == track_label, centroids[track_label]


def track_checkpoints(map_bin, resolution, origin, n_checkpoints=8):
    """Same angular-binning placement as planning_utils.ring_checkpoints(),
    but built on find_track_mask() (largest component) instead of
    find_ring_mask() (lowest fill-ratio) -- see find_track_mask()."""
    h, w = map_bin.shape
    track_mask, track_centroid = find_track_mask(map_bin)
    occ_mask = (map_bin == 1).astype(np.uint8)
    dist = cv2.distanceTransform(1 - occ_mask, cv2.DIST_L2, 5)

    cy, cx = track_centroid[1], track_centroid[0]
    ys, xs = np.where(track_mask)
    ang = np.degrees(np.arctan2(ys - cy, xs - cx))
    ang = (ang + 360) % 360

    bin_width = 360.0 / n_checkpoints
    checkpoints = []
    for k in range(n_checkpoints):
        target = k * bin_width
        d = np.abs(((ang - target + 180) % 360) - 180)
        sel = np.where(d < (bin_width / 2 + 2))[0]
        if len(sel) == 0:
            continue
        dvals = dist[ys[sel], xs[sel]]
        best = sel[np.argmax(dvals)]
        col, row = int(xs[best]), int(ys[best])
        gx, gy = col, h - 1 - row
        checkpoints.append(map_to_world(gx, gy, resolution, origin))

    if len(checkpoints) < 3:
        raise RuntimeError("No se pudieron ubicar suficientes checkpoints sobre la pista.")
    return checkpoints


def anchor_checkpoints(checkpoints, anchor_xy):
    """Replace whichever checkpoint is closest to anchor_xy with the exact
    anchor point, and rotate the list so it comes first.

    Keeps the total checkpoint count (and therefore full-lap coverage)
    unchanged -- only the single nearest angularly-placed checkpoint gets
    swapped out -- while guaranteeing the route starts/ends exactly at
    anchor_xy instead of wherever the angular binning happened to land.
    order_checkpoints_by_track() always keeps position 0 fixed while it
    searches for the best order of the rest, so putting the anchor there
    is what actually pins it as the route's start/end point.
    """
    if anchor_xy is None:
        return checkpoints
    nearest_idx = min(range(len(checkpoints)),
                       key=lambda i: math.hypot(checkpoints[i][0] - anchor_xy[0],
                                                 checkpoints[i][1] - anchor_xy[1]))
    checkpoints = list(checkpoints)
    checkpoints[nearest_idx] = anchor_xy
    return checkpoints[nearest_idx:] + checkpoints[:nearest_idx]


def load_slam_map(yaml_path):
    """Load a trinary ROS map_server-style map (occupied=0, unknown=205,
    free=254). Returns (map_bin, resolution, origin) in the same convention
    planning_utils expects: map_bin[y, x] == 1 means NOT free (blocked)."""
    yaml_path = Path(yaml_path)
    with yaml_path.open('r') as f:
        cfg = yaml.safe_load(f)

    img_path = Path(cfg['image'])
    if not img_path.is_absolute():
        img_path = (yaml_path.parent / img_path).resolve()
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    free = (img == 254)
    map_bin = np.where(free, 0, 1).astype(np.uint8)
    return map_bin, float(cfg['resolution']), cfg['origin']


def plan_track_path(map_yaml, n_checkpoints=8, n_segments=8, penalty_weight=3.0, start_xy=SPAWN_XY):
    map_bin, resolution, origin = load_slam_map(map_yaml)
    env = grid_from_map(map_bin)
    clearance = build_clearance_grid(map_bin)

    checkpoints = track_checkpoints(map_bin, resolution, origin, n_checkpoints)
    checkpoints = anchor_checkpoints(checkpoints, start_xy)
    checkpoints = order_checkpoints_by_track(map_bin, resolution, origin, checkpoints)

    print(f"Checkpoints alrededor de la pista ({len(checkpoints)}):")
    for i, (x, y) in enumerate(checkpoints):
        print(f"  {i}: ({x:.2f}, {y:.2f})")

    segments = []
    for i in range(n_segments):
        a = checkpoints[i % n_checkpoints]
        b = checkpoints[(i + 1) % n_checkpoints]

        start = world_to_map(a[0], a[1], resolution, origin)
        goal = world_to_map(b[0], b[1], resolution, origin)
        if not is_free_cell(start, map_bin) or not is_free_cell(goal, map_bin):
            raise ValueError("Start o goal caen en un obstaculo/zona desconocida.")

        planner = CenteredThetaStar(start=start, goal=goal, env=env, clearance=clearance,
                                     heuristic_type="euclidean", penalty_weight=penalty_weight)
        cost, path, _ = planner.plan()
        if not path:
            raise RuntimeError(f"Theta* no encontro ruta para el tramo {i + 1}/{n_segments}.")
        path = path[::-1]
        seg = [map_to_world(gx, gy, resolution, origin) for gx, gy in path]
        segments.append(seg)
        print(f"  tramo {i + 1}/{n_segments}: {len(seg)} puntos crudos")

    world_path = concat_segments(segments)
    return world_path, map_bin, resolution, origin


def save_overlay_plot(map_bin, resolution, origin, waypoints, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = map_bin.shape
    extent = [origin[0], origin[0] + w * resolution, origin[1], origin[1] + h * resolution]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(1 - map_bin, cmap="gray", extent=extent)
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    ax.plot(xs, ys, "-o", color="red", markersize=2, linewidth=1)
    ax.plot(xs[0], ys[0], "og", markersize=8, label="start/goal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    import argparse

    share = Path(get_package_share_directory('global_planner'))

    parser = argparse.ArgumentParser(description="Theta* sobre el mapa SLAM de AutoDRIVE")
    parser.add_argument("--map", default=str(share / "maps" / "F1tenth_Map.yaml"))
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--penalty-weight", type=float, default=3.0)
    parser.add_argument("--spacing", type=float, default=0.3,
                         help="Espaciado entre waypoints [m]")
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
    print(f"Theta* crudo: {len(world_path)} puntos, resolucion {resolution:.3f} m/celda")

    waypoints = resample_by_arclength(world_path, args.spacing)
    waypoints = open_loop_gap(waypoints, args.spacing, gap=args.gap)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "theta_star_waypoints.csv"
    save_waypoints_csv(waypoints, csv_path)
    png_path = outdir / "theta_star_waypoints.png"
    save_overlay_plot(map_bin, resolution, origin, waypoints, png_path,
                       f"Theta* - AutoDRIVE SLAM map ({len(waypoints)} waypoints)")
    print(f"{len(waypoints)} waypoints -> {csv_path}")
    print(f"Overlay -> {png_path}")


if __name__ == "__main__":
    main()
