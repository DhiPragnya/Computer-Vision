"""Topic 3 maths, part 1: binary images, connectivity, boundaries, contour tracing, length and area,
polygonal approximation (Ramer-Douglas-Peucker) and shape measurements - written to be read by students.

Conventions (the same as core.edges)
-----------------------------------
* A binary image B is a bool array indexed B[row, col]; pixel (x, y) = (col, row), so y points DOWN.
* A contour is a float array of shape (N, 2) holding (x, y) = (col, row) - the order OpenCV uses.
* Pages that only do geometry (shoelace, RDP derivations) draw in maths axes (y up). The formulas are
  identical; only the meaning of "clockwise" flips (the shoelace page shows that).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from . import edges as E

# --------------------------------------------------------------------------
# 1. Scenes and binary images
# --------------------------------------------------------------------------


def _star(cx, cy, r_out, r_in, n=5, rot=-90.0):
    a = np.deg2rad(rot) + np.arange(2 * n) * np.pi / n
    r = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.int32(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)])


def shapes_scene(noise: float = 0.03, seed: int = 4) -> np.ndarray:
    """A grey image with simple objects of different brightness: rectangle, triangle, disc, a ring (an object
    with a hole) holding a small disc, a star and a long thin bar. Background ≈ 0.2."""
    h, w = 240, 340
    img = np.full((h, w), 0.2)
    cv2.rectangle(img, (18, 22), (108, 88), 0.85, -1)                                   # rectangle
    cv2.fillPoly(img, [np.int32([[28, 222], [118, 222], [70, 128]])], 0.7)               # triangle
    cv2.circle(img, (166, 58), 38, 0.62, -1)                                             # disc
    cv2.circle(img, (272, 72), 52, 0.9, -1)                                              # ring ...
    cv2.circle(img, (272, 72), 30, 0.2, -1)                                              # ... its hole
    cv2.circle(img, (272, 72), 11, 0.75, -1)                                             # object in the hole
    cv2.fillPoly(img, [_star(176, 178, 50, 21)], 0.8)                                    # star
    box = cv2.boxPoints(((285, 190), (110, 20), -28.0))                                  # thin rotated bar
    cv2.fillPoly(img, [np.int32(box)], 0.66)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    if noise > 0:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 1)


def ring_scene() -> np.ndarray:
    """Nested objects for the hierarchy lesson: a frame with a hole, in the hole a ring with its own hole,
    in that hole a dot; plus a separate blob outside. Grey values, background 0.15."""
    img = np.full((220, 300), 0.15)
    cv2.rectangle(img, (14, 14), (206, 206), 0.85, -1)       # C0: big square ...
    cv2.rectangle(img, (34, 34), (186, 186), 0.15, -1)       # ... with a square hole
    cv2.circle(img, (110, 110), 58, 0.85, -1)                # C2: ring inside the hole ...
    cv2.circle(img, (110, 110), 34, 0.15, -1)                # ... with its own hole
    cv2.circle(img, (110, 110), 13, 0.85, -1)                # C4: dot inside the ring's hole
    cv2.ellipse(img, (256, 70), (30, 44), 0, 0, 360, 0.85, -1)  # a separate object
    cv2.ellipse(img, (256, 170), (26, 26), 0, 0, 360, 0.85, -1)
    cv2.circle(img, (256, 170), 10, 0.15, -1)                # a small ring
    return cv2.GaussianBlur(img, (0, 0), 0.8)


SCENES = {
    "Shapes (synthetic)": shapes_scene,
    "Nested rings (synthetic)": ring_scene,
    "Coins (photo)": lambda: E.photo("coins.png"),
    "Cameraman (photo)": lambda: E.photo("cameraman.png"),
}


def binarize(img: np.ndarray, t: float, invert: bool = False) -> np.ndarray:
    """B(x, y) = 1 where the image is brighter than t (or darker, with invert=True)."""
    return (img <= t) if invert else (img > t)


def parse_grid(rows: list[str]) -> np.ndarray:
    """Tiny hand-drawn binary images for the animations: '#' = foreground, anything else = background."""
    return np.array([[ch == "#" for ch in r] for r in rows], bool)


# --------------------------------------------------------------------------
# 2. Connectivity and boundary pixels
# --------------------------------------------------------------------------

# (drow, dcol). N8 is listed CLOCKWISE on screen (y down), starting at "up" - the order Moore tracing uses.
N4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
N8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KERNEL = {4: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8), 8: np.ones((3, 3), np.uint8)}


def label(B: np.ndarray, conn: int = 8) -> tuple[int, np.ndarray]:
    """Connected components: (number of objects, label image with 0 = background)."""
    n, lab = cv2.connectedComponents(B.astype(np.uint8), connectivity=conn)
    return n - 1, lab


def label_layers(B: np.ndarray, conn: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Flood-fill labelling written out: every object grows from its first pixel (raster order), one ring of
    neighbours per step. Returns (labels, step at which each pixel was reached); -1 = background."""
    offs = N4 if conn == 4 else N8
    h, w = B.shape
    lab = np.full(B.shape, -1, int)
    depth = np.full(B.shape, -1, int)
    nxt = 0
    for r, c in zip(*np.nonzero(B)):
        if lab[r, c] >= 0:
            continue
        lab[r, c], depth[r, c] = nxt, 0
        q = deque([(r, c)])
        while q:
            pr, pc = q.popleft()
            for dr, dc in offs:
                rr, cc = pr + dr, pc + dc
                if 0 <= rr < h and 0 <= cc < w and B[rr, cc] and lab[rr, cc] < 0:
                    lab[rr, cc], depth[rr, cc] = nxt, depth[pr, pc] + 1
                    q.append((rr, cc))
        nxt += 1
    return lab, depth


def boundary(B: np.ndarray, conn: int = 4) -> np.ndarray:
    """∂Ω = {p ∈ Ω : N(p) ∩ Ωᶜ ≠ ∅}: foreground pixels with at least one background neighbour in N.

    The same as Ω minus its erosion. Pixels outside the image count as background (zero padding)."""
    P = np.pad(B.astype(np.uint8), 1)
    inner = cv2.erode(P, KERNEL[conn], borderType=cv2.BORDER_CONSTANT, borderValue=0)[1:-1, 1:-1]
    return B & ~inner.astype(bool)


def boundary_check(B: np.ndarray, r: int, c: int, conn: int = 4) -> tuple[list, list]:
    """For one pixel: its neighbours inside Ω and those in the background Ωᶜ (as (row, col) lists)."""
    offs = N4 if conn == 4 else N8
    inside, outside = [], []
    for dr, dc in offs:
        rr, cc = r + dr, c + dc
        fg = 0 <= rr < B.shape[0] and 0 <= cc < B.shape[1] and B[rr, cc]
        (inside if fg else outside).append((rr, cc))
    return inside, outside


# --------------------------------------------------------------------------
# 3. Contour tracing (Moore-neighbour) and chain codes
# --------------------------------------------------------------------------


def moore_trace(B: np.ndarray, start: tuple[int, int] | None = None, max_moves: int = 200_000,
                naive: bool = False) -> dict:
    """Trace the outer boundary of the object containing `start` (default: the first foreground pixel in raster
    order) with Moore-neighbour tracing.

    At the current pixel p we know one BACKGROUND neighbour b (where we came from). Turn clockwise around p,
    starting just after b; the first foreground pixel is the next contour pixel, and the background pixel
    checked just before it becomes the new b. Stop when we are back at the start AND about to make the same
    first move again (stopping at the first return is not enough: thin parts are visited twice).

    naive=True stops at the FIRST return to the start pixel instead (to show why that is not enough).

    Returns points [(row, col), ...] (closed: the last connects to the first) and a log of every move:
    dict(p, back, checks=[...], found)."""
    P = np.pad(B.astype(bool), 1)
    if start is None:
        fg = np.argwhere(P)
        if len(fg) == 0:
            return dict(points=[], log=[])
        s = (int(fg[0][0]), int(fg[0][1]))
    else:
        s = (start[0] + 1, start[1] + 1)
    b = (s[0], s[1] - 1)                        # raster order: the pixel to the left is background
    p, pts, log, first = s, [s], [], None
    for _ in range(max_moves):
        d = N8.index((b[0] - p[0], b[1] - p[1]))
        checks, found, prev = [], None, b
        for i in range(1, 9):
            dr, dc = N8[(d + i) % 8]
            q = (p[0] + dr, p[1] + dc)
            checks.append(q)
            if P[q]:
                found = q
                break
            prev = q
        if found is None:                       # an isolated pixel: its contour is the pixel itself
            break
        if first is None:
            first = found
        elif p == s and found == first:         # back at the start, about to repeat the first move
            break
        log.append(dict(p=p, back=b, checks=checks, found=found))
        pts.append(found)
        if naive and found == s:
            break
        p, b = found, prev
    if len(pts) > 1 and pts[-1] == s:
        pts = pts[:-1]
    un = lambda q: (q[0] - 1, q[1] - 1)          # undo the padding
    log = [dict(p=un(m["p"]), back=un(m["back"]), checks=[un(q) for q in m["checks"]], found=un(m["found"]))
           for m in log]
    return dict(points=[un(q) for q in pts], log=log)


# Freeman chain code: 0 = east, then counter-clockwise ON SCREEN: 1 = north-east (up-right), 2 = up, ...
_FREEMAN = {(0, 1): 0, (-1, 1): 1, (-1, 0): 2, (-1, -1): 3, (0, -1): 4, (1, -1): 5, (1, 0): 6, (1, 1): 7}
FREEMAN_ARROWS = "→↗↑↖←↙↓↘"


def chain_code(points_rc: list[tuple[int, int]]) -> list[int]:
    """Freeman 8-direction code of a closed pixel contour given as (row, col) points."""
    out = []
    for a, b in zip(points_rc, points_rc[1:] + points_rc[:1]):
        out.append(_FREEMAN[(b[0] - a[0], b[1] - a[1])])
    return out


# --------------------------------------------------------------------------
# 4. Length and area
# --------------------------------------------------------------------------


def perimeter(P: np.ndarray, closed: bool = True) -> float:
    """P = Σ ‖p_{i+1} − p_i‖ (+ ‖p_0 − p_{N−1}‖ when closed)."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return 0.0
    Q = np.vstack([P, P[:1]]) if closed else P
    return float(np.hypot(*np.diff(Q, axis=0).T).sum())


def shoelace_terms(P: np.ndarray) -> np.ndarray:
    """x_i y_{i+1} − x_{i+1} y_i  for i = 0 … N−1 (with p_N = p_0): twice the signed area of triangle O p_i p_{i+1}."""
    P = np.asarray(P, float)
    Q = np.roll(P, -1, axis=0)
    return P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1]


def signed_area(P: np.ndarray) -> float:
    """½ Σ (x_i y_{i+1} − x_{i+1} y_i). Positive = counter-clockwise in maths axes (y up)."""
    return 0.5 * float(shoelace_terms(P).sum()) if len(P) >= 3 else 0.0


def area(P: np.ndarray) -> float:
    return abs(signed_area(P))


# --------------------------------------------------------------------------
# 5. Polygonal approximation: Ramer-Douglas-Peucker
# --------------------------------------------------------------------------


def line_coeffs(p0, pn) -> tuple[float, float, float]:
    """The line A x + B y + C = 0 through p0 and pn:  A = y0 − yn,  B = xn − x0,  C = x0 yn − xn y0."""
    (x0, y0), (xn, yn) = p0, pn
    return float(y0 - yn), float(xn - x0), float(x0 * yn - xn * y0)


def point_line_distance(P: np.ndarray, p0, pn) -> np.ndarray:
    """d_i = |A x_i + B y_i + C| / √(A² + B²). If p0 = pn the 'line' is a point: plain distance to it."""
    P = np.asarray(P, float).reshape(-1, 2)
    A, B, C = line_coeffs(p0, pn)
    nrm = np.hypot(A, B)
    if nrm < 1e-12:
        return np.hypot(P[:, 0] - p0[0], P[:, 1] - p0[1])
    return np.abs(A * P[:, 0] + B * P[:, 1] + C) / nrm


def rdp(P: np.ndarray, eps: float, trace: bool = False):
    """Ramer-Douglas-Peucker on an OPEN curve. Returns the kept indices (sorted).

    RDP(P, ε) = {p0, pn}                      if d_max ≤ ε
              = RDP(P1, ε) ∪ RDP(P2, ε)       otherwise, split at the farthest point p_k.

    trace=True also returns the list of segments in the order the recursion visits them:
    dict(i0, i1, d (distances of the interior points), k, dmax, split) and the number of distance evaluations."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return (list(range(n)), [], 0) if trace else list(range(n))
    keep = {0, n - 1}
    events, ops = [], 0
    stack = [(0, n - 1)]                # the recursion written with an explicit stack (left half first),
    while stack:                        # so a worst-case curve cannot hit Python's recursion limit
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            if trace:
                events.append(dict(i0=i0, i1=i1, d=np.array([]), k=None, dmax=0.0, split=False))
            continue
        d = point_line_distance(P[i0 + 1:i1], P[i0], P[i1])
        ops += len(d)
        j = int(np.argmax(d))
        k, dmax = i0 + 1 + j, float(d[j])
        split = dmax > eps
        if trace:
            events.append(dict(i0=i0, i1=i1, d=d, k=k, dmax=dmax, split=split))
        if split:
            keep.add(k)
            stack.append((k, i1))
            stack.append((i0, k))
    idx = sorted(keep)
    return (idx, events, ops) if trace else idx


def rdp_closed(P: np.ndarray, eps: float) -> list[int]:
    """RDP on a CLOSED contour: there are no natural end points, so first split the loop at two far-apart
    points (here: p0 and the point farthest from it, then the point farthest from that one), and run RDP on
    the two halves - essentially what cv2.approxPolyDP does."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 4:
        return list(range(n))
    a = int(np.argmax(np.hypot(*(P - P[0]).T)))
    b = int(np.argmax(np.hypot(*(P - P[a]).T)))
    i, j = sorted((a, b))
    if i == j:
        return [0]
    first = rdp(P[i:j + 1], eps)
    loop = np.vstack([P[j:], P[:i + 1]])
    second = rdp(loop, eps)
    idx = {i + k for k in first} | {(j + k) % n for k in second}
    return sorted(idx)


def optimal_vertices(P: np.ndarray, eps: float) -> list[int]:
    """The FEWEST vertices for an open curve under the same rule as RDP (every skipped point within ε of the
    chord's line): a shortest path from p0 to pn in the graph of all 'allowed shortcuts' i -> j. O(n³): small n only."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return list(range(n))
    prev = [-1] * n
    dist = [np.inf] * n
    dist[0] = 0
    for i in range(n - 1):              # edges only go forward, so one sweep in index order is a valid BFS/DP
        if not np.isfinite(dist[i]):
            continue
        for j in range(i + 1, n):
            if j - i >= 2 and point_line_distance(P[i + 1:j], P[i], P[j]).max() > eps:
                continue
            if dist[i] + 1 < dist[j]:
                dist[j], prev[j] = dist[i] + 1, i
    out, k = [], n - 1
    while k >= 0:
        out.append(k)
        k = prev[k]
    return out[::-1]


def max_deviation(P: np.ndarray, kept: list[int], closed: bool = False) -> float:
    """The largest distance of any original point from the polygon segment that replaced it."""
    P = np.asarray(P, float)
    kept = sorted(kept)
    pairs = list(zip(kept, kept[1:]))
    if closed and len(kept) > 1:
        pairs.append((kept[-1], kept[0] + len(P)))
    worst = 0.0
    for a, b in pairs:
        idx = np.arange(a, b + 1) % len(P)
        if len(idx) > 2:
            seg = P[idx]
            p0, p1 = seg[0], seg[-1]
            # distance to the SEGMENT (not the infinite line), clipped at its ends
            v = p1 - p0
            L2 = float(v @ v)
            t = np.clip(((seg - p0) @ v) / L2, 0, 1) if L2 > 1e-12 else np.zeros(len(seg))
            proj = p0 + t[:, None] * v
            worst = max(worst, float(np.hypot(*(seg - proj).T).max()))
    return worst


# --------------------------------------------------------------------------
# 6. Shape measurements
# --------------------------------------------------------------------------


def measure(P: np.ndarray, eps_frac: float = 0.02) -> dict:
    """Area, perimeter, bounding box, centroid (from moments), convex hull, solidity, aspect ratio, extent,
    circularity, convexity and the RDP vertex count of one closed contour (x, y) array."""
    P = np.asarray(P, float).reshape(-1, 2)
    A, L = area(P), perimeter(P)
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    w, h = x1 - x0, y1 - y0
    m = cv2.moments(P.astype(np.float32))
    cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if abs(m["m00"]) > 1e-9 else (P[:, 0].mean(), P[:, 1].mean())
    hull = cv2.convexHull(P.astype(np.float32)).reshape(-1, 2)
    hA = area(hull)
    (_, _), (rw, rh), ang = cv2.minAreaRect(P.astype(np.float32))
    approx = rdp_closed(P, eps_frac * L)
    return dict(area=A, perimeter=L, bbox=(float(x0), float(y0), float(w), float(h)), centroid=(float(cx), float(cy)),
                hull=hull, hull_area=hA, solidity=A / hA if hA > 1e-9 else 0.0,
                aspect=w / h if h > 1e-9 else np.inf, extent=A / (w * h) if w * h > 1e-9 else 0.0,
                rot_aspect=max(rw, rh) / max(min(rw, rh), 1e-9),
                circularity=4 * np.pi * A / L ** 2 if L > 1e-9 else 0.0,
                convex=bool(cv2.isContourConvex(np.round(P[approx]).astype(np.int32))) if len(approx) >= 3 else False,
                approx=approx, vertices=len(approx))


def classify(m: dict) -> str:
    """A simple rule-based shape label from the measurements (the kind of rule students write in practice)."""
    n, circ, sol = m["vertices"], m["circularity"], m["solidity"]
    if m["area"] < 30:
        return "speck"
    if sol < 0.8:
        return "concave (star-like)"
    if n == 3:
        return "triangle"
    if n == 4:
        return "square" if m["rot_aspect"] < 1.15 else ("bar" if m["rot_aspect"] > 3 else "rectangle")
    if circ > 0.85:
        return "circle"
    if n == 5:
        return "pentagon"
    return "ellipse / blob" if circ > 0.6 else f"{n}-gon"


# --------------------------------------------------------------------------
# 7. OpenCV contours
# --------------------------------------------------------------------------

MODES = {"EXTERNAL (outer only)": cv2.RETR_EXTERNAL, "LIST (all, flat)": cv2.RETR_LIST,
         "CCOMP (2 levels)": cv2.RETR_CCOMP, "TREE (full hierarchy)": cv2.RETR_TREE}
APPROX = {"NONE (every pixel)": cv2.CHAIN_APPROX_NONE, "SIMPLE (corners of straight runs)": cv2.CHAIN_APPROX_SIMPLE}


def find_contours(B: np.ndarray, mode=cv2.RETR_TREE, approx=cv2.CHAIN_APPROX_NONE, min_points: int = 1):
    """cv2.findContours on a bool image -> (list of (N, 2) float (x, y) arrays, hierarchy (n, 4) int array).

    hierarchy[i] = [next, previous, first_child, parent], −1 = none."""
    cs, hier = cv2.findContours(B.astype(np.uint8), mode, approx)
    cs = [c.reshape(-1, 2).astype(float) for c in cs]
    hier = hier.reshape(-1, 4) if hier is not None else np.zeros((0, 4), int)
    if min_points > 1:
        keep = [i for i, c in enumerate(cs) if len(c) >= min_points]
        if len(keep) < len(cs):        # re-index the hierarchy: drop tiny contours (noise specks)
            remap = {old: new for new, old in enumerate(keep)}
            cs = [cs[i] for i in keep]
            hier = np.array([[remap.get(v, -1) if v >= 0 else -1 for v in hier[i]] for i in keep], int).reshape(-1, 4)
    return cs, hier


def depths(hier: np.ndarray) -> list[int]:
    """Nesting depth of every contour: 0 = outermost, 1 = a hole in it, 2 = an object inside that hole, ..."""
    out = []
    for i in range(len(hier)):
        d, p = 0, hier[i][3]
        while p >= 0:
            d, p = d + 1, hier[p][3]
        out.append(d)
    return out


# --------------------------------------------------------------------------
# 8. Digital shapes (for "measured" claims about pixels)
# --------------------------------------------------------------------------


def disk(r: float, pad: int = 3) -> np.ndarray:
    """A digital disc: pixels whose CENTRE lies within r of the centre."""
    n = int(np.ceil(r)) + pad
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return xx ** 2 + yy ** 2 <= r ** 2


def staircase(B: np.ndarray) -> dict:
    """Three ways to measure the perimeter of a pixel object:
    crack  = number of pixel SIDES between object and background (walk along the pixel edges, 4-connected steps)
    chain8 = length of the 8-connected boundary-pixel chain (steps of 1 and √2) - what cv2.arcLength measures
    simple = the same chain after CHAIN_APPROX_SIMPLE (identical length: it only drops points on straight runs)."""
    P = np.pad(B.astype(np.int8), 1)
    crack = int(np.abs(np.diff(P, axis=0)).sum() + np.abs(np.diff(P, axis=1)).sum())
    cs, _ = find_contours(B, cv2.RETR_EXTERNAL)
    c = max(cs, key=len) if cs else np.zeros((0, 2))
    return dict(crack=crack, chain8=perimeter(c), contour=c, pixels=int(B.sum()), poly_area=area(c))


# --------------------------------------------------------------------------
# 9. Display helpers (uint8 RGB images for st.image)
# --------------------------------------------------------------------------

DARK = (26, 26, 25)
BLUE = (42, 120, 214)
ORANGE = (235, 104, 52)
GREEN = (27, 175, 122)
AMBER = (237, 161, 0)
RED = (227, 73, 72)
MAGENTA = (213, 81, 129)
GREY = (143, 141, 135)
PALETTE = [BLUE, ORANGE, GREEN, MAGENTA, AMBER, (120, 94, 240), (0, 160, 190), RED]


def binary_rgb(B: np.ndarray, fg=(235, 235, 232), bg=DARK) -> np.ndarray:
    out = np.empty(B.shape + (3,), np.uint8)
    out[...] = bg
    out[B] = fg
    return out


def labels_rgb(lab: np.ndarray) -> np.ndarray:
    """Each label its own colour; background (0 or −1) dark."""
    out = np.empty(lab.shape + (3,), np.uint8)
    out[...] = DARK
    ids = [v for v in np.unique(lab) if v > 0]
    for i, v in enumerate(ids):
        out[lab == v] = PALETTE[i % len(PALETTE)]
    return out


def draw_contours(rgb: np.ndarray, contours: list[np.ndarray], colors=None, scale: int = 2, thickness: int = 2,
                  closed: bool = True, dots: bool = False) -> np.ndarray:
    """Enlarge the image (nearest neighbour) and draw contours through the pixel CENTRES on top."""
    big = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).copy()
    for i, c in enumerate(contours):
        if len(c) == 0:
            continue
        col = (colors[i % len(colors)] if isinstance(colors, list) else colors) if colors is not None else BLUE
        pts = np.int32(np.round((np.asarray(c, float) + 0.5) * scale)).reshape(-1, 1, 2)
        cv2.polylines(big, [pts], closed and len(c) > 2, col, thickness, cv2.LINE_AA)
        if dots:
            for x, y in pts.reshape(-1, 2):
                cv2.circle(big, (int(x), int(y)), thickness + 2, col, -1, cv2.LINE_AA)
    return big


def gray_rgb(img: np.ndarray, dim: float = 1.0) -> np.ndarray:
    g = (np.clip(img, 0, 1) * 255 * dim).astype(np.uint8)
    return np.dstack([g] * 3)
