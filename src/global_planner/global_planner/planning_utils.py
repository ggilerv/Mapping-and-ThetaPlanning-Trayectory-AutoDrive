"""
@file: planning_utils.py
@breif: Shared helpers for running global planners (Theta*, RRT) on F1TENTH maps
        and post-processing their raw output into fixed-spacing waypoint files.
"""
import csv
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# python_motion_planning is vendored as a sibling of this file (see
# python_motion_planning/LICENSE for its own GPL-3.0 license) so this package
# is self-contained -- no external repo needs to exist on the machine it
# runs on. Adding this file's own directory to sys.path (rather than some
# fixed number of parent hops) makes the import work the same way whether
# this is run from source or from wherever colcon installs it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from python_motion_planning.curve_generation import BSpline
from python_motion_planning.utils import Grid, Map, Node, SearchFactory
from python_motion_planning.global_planner.sample_search.rrt import RRT
from python_motion_planning.global_planner.graph_search.theta_star import ThetaStar


def load_map(yaml_path, downsample_factor=1):
    yaml_path = Path(yaml_path)
    with yaml_path.open('r') as f:
        map_config = yaml.safe_load(f)

    img_path = Path(map_config['image'])
    if not img_path.is_absolute():
        img_path = (yaml_path.parent / img_path).resolve()
    map_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    resolution = map_config['resolution']
    origin = map_config['origin']

    # Binarizar: 1 = ocupado, 0 = libre
    map_bin = np.zeros_like(map_img, dtype=np.uint8)
    map_bin[map_img < int(0.45 * 255)] = 1

    # Engrosar obstaculos segun el factor
    if downsample_factor > 12:
        map_bin = cv2.dilate(map_bin, np.ones((5, 5), np.uint8), iterations=2)
    elif downsample_factor >= 4:
        map_bin = cv2.dilate(map_bin, np.ones((3, 3), np.uint8), iterations=1)

    map_bin = map_bin.astype(np.float32)
    h, w = map_bin.shape
    new_h, new_w = h // downsample_factor, w // downsample_factor
    map_bin = cv2.resize(map_bin, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if downsample_factor > 12:
        map_bin = (map_bin > 0.10).astype(np.uint8)
    elif downsample_factor >= 4:
        map_bin = (map_bin > 0.25).astype(np.uint8)
    else:
        map_bin = (map_bin >= 0.5).astype(np.uint8)

    resolution *= downsample_factor
    return map_bin, resolution, origin


def inflate_obstacles(map_bin, margin_cells=1):
    """Grow every occupied cell by `margin_cells` in all directions.

    Used to build a "safe" grid to plan against instead of the true
    occupancy grid: planning against the inflated grid keeps the resulting
    path at least `margin_cells` away from every wall, which is what pushes
    it toward the centerline of the corridor instead of hugging an edge.
    This happens entirely at the planning-input level (a different, more
    conservative map) rather than by post-processing the output path, so it
    stays compatible with using the raw, unsmoothed planner output.

    Where the corridor is already narrower than 2*margin_cells (e.g. a tight
    chicane), inflating removes *all* free cells there, so no path exists on
    the inflated grid through that section -- callers should fall back to
    planning against the true (non-inflated) grid for whichever segment that
    affects, rather than lowering the margin for the whole track.
    """
    if margin_cells <= 0:
        return map_bin
    kernel = np.ones((2 * margin_cells + 1, 2 * margin_cells + 1), np.uint8)
    return cv2.dilate(map_bin.astype(np.uint8), kernel, iterations=1)


def grid_from_map(map_bin):
    """Build a discrete Grid env. obstacles are stored as (x, y) with y flipped
    relative to image rows, matching python_motion_planning's bottom-left origin."""
    h, w = map_bin.shape
    env = Grid(w, h)
    obstacles = {(x, h - 1 - y) for y in range(h) for x in range(w) if map_bin[y, x] == 1}
    env.update(obstacles)
    return env


def build_clearance_grid(map_bin):
    """Distance (in cells) from every cell to the nearest obstacle, indexed
    as clearance[gx, gy] to match this module's other grid helpers
    (world_to_map/is_free_cell/grid_from_map). Used to weight path cost
    toward open space (see CenteredThetaStar) without hard-blocking any cell.
    """
    h, w = map_bin.shape
    occ = (map_bin == 1).astype(np.uint8)
    dist = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 5)
    clearance = np.zeros((w, h), dtype=np.float32)
    for y in range(h):
        clearance[:, h - 1 - y] = dist[y, :]
    return clearance


class CenteredThetaStar(ThetaStar):
    """Theta* whose step cost (both the regular grid steps and the any-angle
    line-of-sight shortcuts) includes a clearance penalty, so the search's
    own notion of "shortest path" already prefers the centerline -- even in
    a wide corridor, where a fixed obstacle-inflation margin (see
    inflate_obstacles) still leaves enough room for the true shortest path
    to cut the inside of a curve.

    Concretely, the cost of a step is `geometric_distance * (1 + weight /
    (min_clearance_along_step + 1))`, so steps that pass close to a wall
    anywhere along their length cost noticeably more than steps that stay in
    open corridor space throughout, while the heuristic used to guide the
    search stays the plain Euclidean distance -- still an admissible
    (non-overestimating) lower bound on this larger weighted cost, so the
    search is still a valid best-first shortest-path search, just over a
    different notion of "distance". This mirrors how real costmap-based
    planners (e.g. in ROS navigation) keep a robot away from obstacles, and
    is the graph-search analogue of how CenteredRRT biases RRT's sampling.

    Using the *minimum* clearance along the step (sampled at several points),
    not just at its two endpoints, matters specifically for Theta*'s
    any-angle line-of-sight shortcuts: a shortcut can span many cells in a
    single straight line, and cutting the inside of a curve shows up as a
    dip in clearance in the *middle* of that line, not at its endpoints --
    checking only the endpoints would miss exactly the "hugs the wall
    through the turn" case this is meant to avoid.
    """
    def __init__(self, start, goal, env, clearance, heuristic_type="euclidean", penalty_weight=3.0):
        super().__init__(start, goal, env, heuristic_type)
        self.clearance = clearance
        self.penalty_weight = penalty_weight

    def __str__(self):
        return "Theta* - sesgado al centro"

    def _clearance_at(self, x, y):
        gx = min(max(int(round(x)), 0), self.clearance.shape[0] - 1)
        gy = min(max(int(round(y)), 0), self.clearance.shape[1] - 1)
        return self.clearance[gx, gy]

    def _min_clearance_along(self, node1, node2):
        n_samples = max(2, int(round(self.dist(node1, node2) / 0.5)) + 1)
        ts = np.linspace(0.0, 1.0, n_samples)
        return min(
            self._clearance_at(node1.x + t * (node2.x - node1.x), node1.y + t * (node2.y - node1.y))
            for t in ts
        )

    def _weighted_dist(self, node1, node2):
        base = self.dist(node1, node2)
        min_clearance = self._min_clearance_along(node1, node2)
        return base * (1.0 + self.penalty_weight / (min_clearance + 1.0))

    def getNeighbor(self, node):
        neighbors = []
        for motion in self.motions:
            node_n = node + motion
            if self.isCollision(node, node_n):
                continue
            node_n.g = node.g + self._weighted_dist(node, node_n)
            neighbors.append(node_n)
        return neighbors

    def updateVertex(self, node_p, node_c):
        if self.lineOfSight(node_c, node_p):
            new_g = node_p.g + self._weighted_dist(node_c, node_p)
            if new_g <= node_c.g:
                node_c.g = new_g
                node_c.parent = node_p.current


def grid_to_map_env(map_bin, pad=0.15):
    """Build a continuous Map env (rectangle obstacles) from the same occupancy
    grid used for Grid, so Theta* and RRT plan over an identical obstacle layout.

    RRT's collision check (SampleSearcher.isCollision) walks the full obs_rect
    list for every candidate edge, so instead of emitting one 1x1 rect per
    occupied cell we merge consecutive occupied cells within each grid row into
    a single rect. This keeps obs_rect small enough for RRT to run in reasonable
    time on a downsampled race-track map (mostly thin wall obstacles).

    Each merged rect is exactly 1 cell tall, so a diagonal (or staircase-shaped)
    wall becomes a stack of 1-cell-tall rects that only touch their neighbors at
    a single corner point -- an RRT edge can cut diagonally through that corner
    notch without registering a collision against either rect, "leaking" through
    a wall that Theta*'s per-cell Grid (which has no such gap) correctly blocks.
    Padding every rect by `pad` cells on all sides makes diagonally-adjacent
    rects overlap, closing the notch, at the cost of a small (sub-cell) growth
    of the obstacle footprint.
    """
    h, w = map_bin.shape
    occ = np.zeros((w, h), dtype=bool)
    for y in range(h):
        occ[:, h - 1 - y] = map_bin[y, :] == 1

    obs_rect = []
    for gy in range(h):
        gx = 0
        while gx < w:
            if occ[gx, gy]:
                gx0 = gx
                while gx < w and occ[gx, gy]:
                    gx += 1
                obs_rect.append([gx0 - pad, gy - pad, (gx - gx0) + 2 * pad, 1 + 2 * pad])
            else:
                gx += 1

    env = Map(w, h)
    env.update(boundary=env.boundary, obs_circ=[], obs_rect=obs_rect)
    return env


class CenteredRRT(RRT):
    """RRT whose random sampling is drawn from the track's drivable ring
    (planning_utils.build_ring_samples) instead of uniformly over the whole
    map, weighted toward pixels with more wall clearance -- so the tree it
    grows (and the path extracted from it) tends to hug the corridor
    centerline more than plain uniform-random RRT, while every point on the
    ring (including the ones right against a wall) can still be sampled, so
    it can still solve a tight chicane, just samples it less eagerly.

    Sampling from the ring specifically (rather than rejection-sampling the
    whole map by clearance) matters because the ring is a tiny fraction of
    the map's area: biasing uniform-over-the-map sampling by clearance barely
    changes anything, since it's dominated by the huge, irrelevant open
    background outside the track. Sampling from the ring's own pixels keeps
    every draw relevant to the corridor, where the clearance weighting can
    actually make a difference.

    This changes how candidate nodes are *sampled* during the search, not
    the path afterwards, so the result is still the algorithm's raw output
    (no post-processing / smoothing of a finished path).
    """
    def __init__(self, start, goal, env, ring_points, ring_clearance, max_dist=1.5, sample_num=10000,
                 goal_sample_rate=0.15, bias_power=1.5, jitter=0.5):
        super().__init__(start, goal, env, max_dist, sample_num, goal_sample_rate)
        self.ring_points = ring_points
        weights = np.clip(ring_clearance, 0.15, None) ** bias_power
        self.ring_weights = weights / weights.sum()
        self.jitter = jitter

    def __str__(self):
        return "Rapidly-exploring Random Tree(RRT) - sesgado al centro"

    def generateRandomNode(self):
        if np.random.random() <= self.goal_sample_rate:
            return self.goal
        idx = np.random.choice(len(self.ring_points), p=self.ring_weights)
        x, y = self.ring_points[idx]
        x += np.random.uniform(-self.jitter, self.jitter)
        y += np.random.uniform(-self.jitter, self.jitter)
        return Node((x, y), None, 0, 0)


def world_to_map(x_world, y_world, resolution, origin):
    x_map = int((x_world - origin[0]) / resolution)
    y_map = int((y_world - origin[1]) / resolution)
    return (x_map, y_map)


def map_to_world(x_map, y_map, resolution, origin):
    x_world = x_map * resolution + origin[0]
    y_world = y_map * resolution + origin[1]
    return (x_world, y_world)


def is_free_cell(cell, map_bin):
    h, w = map_bin.shape
    i, j = cell
    if not (0 <= i < w and 0 <= j < h):
        return False
    return map_bin[h - 1 - j, i] == 0


def find_ring_mask(map_bin):
    """Identify the connected free-space component that is the drivable ring
    of a closed race-track map.

    A race-track PNG only draws the wall curves, so the free space of the
    binarized map is really three disconnected regions: the outfield (outside
    the outer wall), the infield (inside the inner wall), and the actual
    drivable ring between them. The ring is the component with the lowest
    fill ratio relative to its bounding box, since it's a thin loop, unlike
    the much more solid outfield/infield blobs.

    Returns a boolean mask the same shape as map_bin (image row/col indexed).
    """
    h, w = map_bin.shape
    free = (map_bin == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free, connectivity=4)

    candidates = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 0.005 * h * w:
            continue
        bbox_area = stats[lbl, cv2.CC_STAT_WIDTH] * stats[lbl, cv2.CC_STAT_HEIGHT]
        fill_ratio = area / bbox_area if bbox_area else 1.0
        candidates.append((fill_ratio, lbl))
    if not candidates:
        raise RuntimeError("No se encontraron regiones libres suficientemente grandes en el mapa.")
    candidates.sort()
    ring_label = candidates[0][1]
    return labels == ring_label, centroids[ring_label]


def build_ring_samples(map_bin):
    """Grid-frame (gx, gy) coordinates and per-point wall clearance (cells)
    for every pixel of the track's drivable ring, used to sample RRT
    candidate nodes directly from the corridor -- weighted toward higher
    clearance -- instead of uniformly over the whole map (see CenteredRRT).
    """
    h, w = map_bin.shape
    ring_mask, _ = find_ring_mask(map_bin)
    occ_mask = (map_bin == 1).astype(np.uint8)
    dist = cv2.distanceTransform(1 - occ_mask, cv2.DIST_L2, 5)
    ys, xs = np.where(ring_mask)
    points = np.stack([xs, h - 1 - ys], axis=1).astype(float)
    clearance = dist[ys, xs]
    return points, clearance


def ring_checkpoints(map_bin, resolution, origin, n_checkpoints=8):
    """Find `n_checkpoints` points spread evenly by angle around the drivable
    ring of a closed race-track map, in consistent rotational order.

    A point-to-point planner (Theta*/RRT) always returns the *shorter* arc
    between two points on the ring, so it can never produce a full lap by
    itself -- the only way to force a full lap is to chain several
    checkpoints around the ring and plan consecutive segments between them
    (see concat_segments below).

    For each angular bin around the ring's centroid, picks the free pixel
    farthest from any obstacle (a distance transform), i.e. a
    corridor-centerline point.
    """
    h, w = map_bin.shape
    ring_mask, ring_centroid = find_ring_mask(map_bin)
    occ_mask = (map_bin == 1).astype(np.uint8)
    dist = cv2.distanceTransform(1 - occ_mask, cv2.DIST_L2, 5)

    cy, cx = ring_centroid[1], ring_centroid[0]
    ys, xs = np.where(ring_mask)
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
        raise RuntimeError("No se pudieron ubicar suficientes checkpoints sobre el anillo de la pista.")
    return checkpoints


def order_checkpoints_by_track(map_bin, resolution, origin, checkpoints):
    """Reorder checkpoints into the cycle that actually follows the track,
    instead of trusting their raw angle-from-centroid order.

    ring_checkpoints() places points at even angles around the ring's
    centroid, which works for a roughly convex/star-shaped track but breaks
    down when the track has a tight chicane that pokes back in toward the
    centroid (like this map's S-curve): two checkpoints that are angular
    neighbors can end up far apart *along the track*, forcing a huge detour
    between them that doubles back over ground another segment already
    covers (this is what makes the plotted trajectory look "thick"/doubled
    in that area).

    Fixes it by computing the true Theta* path cost between every pair of
    checkpoints (fast and deterministic, regardless of which algorithm will
    actually plan the final segments) and picking the cyclic ordering with
    the lowest total cost -- brute force, since there are only a handful of
    checkpoints (n=8 -> 5040 permutations, trivial).
    """
    n = len(checkpoints)
    env = grid_from_map(map_bin)
    grid_pts = [world_to_map(x, y, resolution, origin) for x, y in checkpoints]

    cost = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            planner = SearchFactory()("theta_star", start=grid_pts[i], goal=grid_pts[j],
                                       env=env, heuristic_type="euclidean")
            c, path, _ = planner.plan()
            d = c if path else float("inf")
            cost[i][j] = cost[j][i] = d

    best_order, best_total = None, float("inf")
    for perm in itertools.permutations(range(1, n)):
        order = (0,) + perm
        total = sum(cost[order[k]][order[(k + 1) % n]] for k in range(n))
        if total < best_total:
            best_total, best_order = total, order

    return [checkpoints[i] for i in best_order]


def concat_segments(segments):
    """Join consecutive start->goal path segments into a single continuous path,
    dropping the duplicated joint point between one segment's end and the next
    segment's start."""
    full = list(segments[0])
    for seg in segments[1:]:
        full.extend(seg[1:])
    return full


def smooth_path(path_xy, step=0.01, k=3, max_control_points=150):
    """Smooth a raw planner path with a parametric (centripetal) B-Spline.

    Uses python_motion_planning.curve_generation.BSpline, which parametrizes by
    chord length rather than assuming monotonic x, so it works on race-track
    paths that loop back on themselves. Raw paths from the grid planners can
    have hundreds of points, and BSpline's interpolation mode inverts an NxN
    matrix (N = number of control points), so long paths are first thinned to
    at most `max_control_points` points before smoothing.
    """
    path_xy = [tuple(p) for p in path_xy]
    if len(path_xy) <= k:
        return path_xy

    control_pts = path_xy
    if len(control_pts) > max_control_points:
        idx = sorted(set(np.linspace(0, len(control_pts) - 1, max_control_points).astype(int).tolist()))
        control_pts = [control_pts[i] for i in idx]
        if control_pts[-1] != path_xy[-1]:
            control_pts[-1] = path_xy[-1]

    generator = BSpline(step=step, k=k, param_mode="centripetal", spline_mode="interpolation")
    return generator.run(control_pts, display=False)


def smooth_path_safe(path_xy, map_bin, resolution, origin, step=0.0005,
                      control_point_options=(90, 100, 110, 130, 150, 200)):
    """Smooth a raw path with smooth_path(), retrying with progressively more
    control points until the resulting curve is fully collision-free.

    smooth_path()'s B-spline is an *interpolating* spline: it must pass
    exactly through every control point it's given, so thinning the raw path
    down to few, widely-spaced control points gives the smoothest-looking
    curve (it has more freedom to swing between them) but occasionally lets
    it swing wide enough to clip a wall in the tightest part of the track --
    especially for RRT, whose raw path is a random sample each run, so a
    single fixed max_control_points value can be safe on one run and clip a
    corner on the next. More control points means a tighter fit to the raw
    path (which is itself already collision-free, since it came straight out
    of the planner), so retrying with more of them until the curve checks out
    against the occupancy grid gets the smoothest result that's still safe,
    instead of gambling on one fixed amount of smoothing.

    Checks the dense smoothed curve itself (not just the final waypoints
    after resampling to 0.5 m/1 m), since a stray excursion between two
    resampled waypoints could otherwise go unnoticed.

    Returns (smoothed_path, max_control_points_used).
    """
    smoothed = None
    for max_control_points in control_point_options:
        smoothed = smooth_path(path_xy, step=step, max_control_points=max_control_points)
        if all(is_free_cell(world_to_map(x, y, resolution, origin), map_bin) for x, y in smoothed):
            return smoothed, max_control_points
    return smoothed, control_point_options[-1]


def resample_by_arclength(path_xy, spacing):
    """Resample a path to waypoints spaced `spacing` meters apart along arc length.
    Always keeps the first and last point of the original path."""
    pts = np.asarray(path_xy, dtype=float)
    if len(pts) < 2:
        return [tuple(p) for p in path_xy]

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < spacing:
        return [tuple(pts[0]), tuple(pts[-1])]

    n_points = int(total // spacing) + 1
    sample_s = np.arange(n_points) * spacing
    if sample_s[-1] < total:
        sample_s = np.append(sample_s, total)

    x = np.interp(sample_s, cum, pts[:, 0])
    y = np.interp(sample_s, cum, pts[:, 1])
    return list(zip(x.tolist(), y.tolist()))


def open_loop_gap(waypoints, spacing, gap=3.0):
    """If a resampled waypoint path is a closed loop (first and last points
    coincide, e.g. a full-lap run), drop enough waypoints from the end to
    leave a gap of about `gap` meters between goal and start, so the two
    markers are visibly distinct instead of overlapping at the same point.
    Leaves open (non-closed) paths untouched.
    """
    if len(waypoints) < 2:
        return waypoints
    closing = np.hypot(waypoints[0][0] - waypoints[-1][0], waypoints[0][1] - waypoints[-1][1])
    if closing > 1e-6:
        return waypoints
    n_drop = max(1, int(round(gap / spacing)))
    if n_drop >= len(waypoints):
        return waypoints
    return waypoints[:-n_drop]


def save_waypoints_csv(path_xy, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in path_xy:
            writer.writerow([x, y])
