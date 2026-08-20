#!/usr/bin/env python3
"""
@file: generate_gif.py
@breif: Renders an animated GIF of Theta* searching the SLAM map, one frame
        per subsampled batch of expanded nodes, for each of the 8 track
        segments in sequence -- the ".gif del algoritmo generando la
        trayectoria" the project's video-evidence rubric asks for.

Reuses CenteredThetaStar directly (not plan_track_path(), since that
discards the per-segment `expand` list of explored nodes that this needs
to animate) with the same checkpoints/order used by generate_trajectory.py,
so the animated route matches the one actually delivered.
"""
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planning_utils import (  # noqa: E402
    grid_from_map, build_clearance_grid, CenteredThetaStar,
    world_to_map, is_free_cell, order_checkpoints_by_track,
)
from generate_trajectory import (  # noqa: E402
    load_slam_map, track_checkpoints, anchor_checkpoints, SPAWN_XY,
)


def main():
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    share = Path(get_package_share_directory('global_planner'))

    parser = argparse.ArgumentParser(description="GIF de Theta* buscando sobre el mapa SLAM de AutoDRIVE")
    parser.add_argument("--map", default=str(share / "maps" / "F1tenth_Map.yaml"))
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--penalty-weight", type=float, default=3.0)
    parser.add_argument("--frames-per-segment", type=int, default=25,
                         help="Cuadros de expansion mostrados por tramo (se submuestrea el expand real)")
    parser.add_argument("--hold-frames", type=int, default=6,
                         help="Cuadros que se mantiene el camino final de cada tramo antes de pasar al siguiente")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--start-x", type=float, default=SPAWN_XY[0])
    parser.add_argument("--start-y", type=float, default=SPAWN_XY[1])
    parser.add_argument("--out", default=str(share / "waypoints" / "theta_star_search.gif"))
    args = parser.parse_args()

    map_bin, resolution, origin = load_slam_map(args.map)
    env = grid_from_map(map_bin)
    clearance = build_clearance_grid(map_bin)
    h, w = map_bin.shape
    extent = [origin[0], origin[0] + w * resolution, origin[1], origin[1] + h * resolution]

    checkpoints = track_checkpoints(map_bin, resolution, origin, args.checkpoints)
    checkpoints = anchor_checkpoints(checkpoints, (args.start_x, args.start_y))
    checkpoints = order_checkpoints_by_track(map_bin, resolution, origin, checkpoints)

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(1 - map_bin, cmap="gray", extent=extent)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    title = ax.set_title("Theta* buscando...")
    expand_scatter = ax.scatter([], [], s=4, c="#5aa0ff", alpha=0.5)
    solved_paths, = ax.plot([], [], "-", color="lime", linewidth=2)
    current_path, = ax.plot([], [], "-", color="yellow", linewidth=1.5)

    solved_x, solved_y = [], []
    frame_specs = []  # list of ('expand'|'solved', segment_idx, data)

    for i in range(args.segments):
        a = checkpoints[i % args.checkpoints]
        b = checkpoints[(i + 1) % args.checkpoints]
        start = world_to_map(a[0], a[1], resolution, origin)
        goal = world_to_map(b[0], b[1], resolution, origin)
        if not is_free_cell(start, map_bin) or not is_free_cell(goal, map_bin):
            raise ValueError(f"Start/goal invalido en el tramo {i + 1}")

        planner = CenteredThetaStar(start=start, goal=goal, env=env, clearance=clearance,
                                     heuristic_type="euclidean", penalty_weight=args.penalty_weight)
        cost, path, expand = planner.plan()
        if not path:
            raise RuntimeError(f"Theta* no encontro ruta en el tramo {i + 1}")
        print(f"tramo {i + 1}/{args.segments}: {len(expand)} nodos expandidos, {len(path)} puntos de camino")

        def world_of(node):
            return node.x * resolution + origin[0], node.y * resolution + origin[1]

        n = max(1, len(expand) // args.frames_per_segment)
        batches = [expand[k:k + n] for k in range(0, len(expand), n)]
        for batch in batches:
            frame_specs.append(('expand', i, [world_of(node) for node in batch]))

        path_xy = [(gx * resolution + origin[0], gy * resolution + origin[1])
                   for gx, gy in path[::-1]]
        for _ in range(args.hold_frames):
            frame_specs.append(('solved', i, path_xy))

    import numpy as np

    def update(frame_idx):
        kind, seg_idx, data = frame_specs[frame_idx]
        title.set_text(f"Theta* - tramo {seg_idx + 1}/{args.segments}")
        if kind == 'expand':
            prev = expand_scatter.get_offsets()
            new_pts = np.array(data) if len(data) else np.empty((0, 2))
            combined = np.vstack([prev, new_pts]) if len(prev) else new_pts
            expand_scatter.set_offsets(combined)
        else:  # 'solved': hold_frames repeats of the same finished path
            xs = [p[0] for p in data]
            ys = [p[1] for p in data]
            current_path.set_data(xs, ys)
            is_first_solved_frame = (
                frame_idx == 0
                or frame_specs[frame_idx - 1][0] != 'solved'
                or frame_specs[frame_idx - 1][1] != seg_idx)
            if is_first_solved_frame:
                solved_x.extend(xs)
                solved_y.extend(ys)
                solved_paths.set_data(solved_x, solved_y)
                expand_scatter.set_offsets(np.empty((0, 2)))
        return expand_scatter, solved_paths, current_path, title

    ani = animation.FuncAnimation(fig, update, frames=len(frame_specs), blit=False)
    writer = animation.PillowWriter(fps=args.fps)
    ani.save(args.out, writer=writer)
    print(f"GIF guardado en {args.out} ({len(frame_specs)} cuadros)")


if __name__ == "__main__":
    main()
