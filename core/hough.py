"""Topic 3 maths, part 2: the Hough transform = voting in parameter space (lines and circles).

Coordinates
-----------
* Image pages use image coordinates: x = column, y = row (DOWN), origin at the top-left pixel. That is the
  convention of cv2.HoughLines:  x cos θ + y sin θ = ρ  with θ ∈ [0°, 180°) and ρ ∈ [−D, D] (D = diagonal).
* The slope-intercept lesson uses school axes (y up), as the notes write y = m x + c.

Every accumulator here is a plain NumPy array filled with np.add.at - one vote per (point, parameter) pair.
"""
from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 1. Synthetic edge points
# --------------------------------------------------------------------------


def clip_line(theta_deg: float, rho: float, w: float, h: float):
    """End points of the line x cos θ + y sin θ = ρ inside the box [0, w] × [0, h] (None if it misses)."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    pts = []
    if abs(s) > 1e-9:
        for x in (0.0, w):
            y = (rho - x * c) / s
            if -1e-9 <= y <= h + 1e-9:
                pts.append((x, y))
    if abs(c) > 1e-9:
        for y in (0.0, h):
            x = (rho - y * s) / c
            if -1e-9 <= x <= w + 1e-9:
                pts.append((x, y))
    pts = sorted(set((round(px, 9), round(py, 9)) for px, py in pts))
    if len(pts) < 2:
        return None
    pts = np.array(pts, float)
    d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return pts[i], pts[j]


def line_points(theta_deg: float, rho: float, w: int, h: int, n: int = 60, noise: float = 0.0,
                gaps: list[tuple[float, float]] | None = None, span: tuple[float, float] = (0.08, 0.92),
                seed: int = 0) -> np.ndarray:
    """n edge points along the visible part of a line (fractions `span` of its length), minus the `gaps`
    (fractions too), jittered by Gaussian noise of std `noise` pixels."""
    seg = clip_line(theta_deg, rho, w - 1, h - 1)
    if seg is None:
        return np.zeros((0, 2))
    a, b = seg
    t = np.linspace(span[0], span[1], n)
    for g0, g1 in gaps or []:
        t = t[(t < g0) | (t > g1)]
    P = a + t[:, None] * (b - a)
    if noise > 0:
        P = P + np.random.default_rng(seed).normal(0, noise, P.shape)
    return P


def clutter(n: int, w: int, h: int, seed: int = 1) -> np.ndarray:
    """n random 'noise' edge points."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(0, w - 1, n), rng.uniform(0, h - 1, n)]


def circle_points(cx, cy, r, n=48, noise=0.0, arc=(0.0, 360.0), seed=0, outward=True):
    """Points on a circle plus the gradient direction θ at each (radians, image coordinates).

    outward=True: the gradient points AWAY from the centre (a dark disc on a bright background).
    outward=False: it points TOWARDS the centre (a bright disc on a dark background)."""
    a = np.deg2rad(np.linspace(arc[0], arc[1], n, endpoint=(arc[1] - arc[0]) < 360))
    rng = np.random.default_rng(seed)
    rr = r + (rng.normal(0, noise, n) if noise > 0 else 0)
    P = np.c_[cx + rr * np.cos(a), cy + rr * np.sin(a)]
    theta = a if outward else a + np.pi
    if noise > 0:
        theta = theta + rng.normal(0, np.deg2rad(4) * noise, n)
    return P, theta


def points_image(P: np.ndarray, w: int, h: int) -> np.ndarray:
    """Edge points -> binary uint8 edge image (what OpenCV's Hough functions want)."""
    img = np.zeros((h, w), np.uint8)
    q = np.round(P).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    img[q[ok, 1], q[ok, 0]] = 255
    return img


# --------------------------------------------------------------------------
# 2. Lines, slope-intercept form: c = y_i − m x_i
# --------------------------------------------------------------------------


def accumulator_mc(P: np.ndarray, ms: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """A(m, c): every point votes once for every m (its c = y − m x, rounded to the nearest c bin).
    Votes whose c falls outside the c range are LOST - the weakness of this form. Shape (len(cs), len(ms))."""
    P = np.asarray(P, float).reshape(-1, 2)
    A = np.zeros((len(cs), len(ms)), np.int32)
    if len(P) == 0:
        return A
    c = P[:, 1:2] - ms[None, :] * P[:, 0:1]                          # (points, m)
    dc = cs[1] - cs[0]
    ci = np.round((c - cs[0]) / dc).astype(int)
    mi = np.broadcast_to(np.arange(len(ms))[None, :], ci.shape)
    ok = (ci >= 0) & (ci < len(cs))
    np.add.at(A, (ci[ok], mi[ok]), 1)
    return A


# --------------------------------------------------------------------------
# 3. Lines, normal form: ρ = x cos θ + y sin θ
# --------------------------------------------------------------------------


def rho_of(x, y, thetas_rad):
    """The sinusoid of one point: ρ(θ) = x cos θ + y sin θ."""
    return x * np.cos(thetas_rad) + y * np.sin(thetas_rad)


def rho_axis(w: int, h: int, rho_res: float = 1.0) -> np.ndarray:
    D = np.hypot(w, h)
    n = int(np.ceil(D / rho_res))
    return np.arange(-n, n + 1) * rho_res


def accumulator_rt(P: np.ndarray, w: int, h: int, n_theta: int = 180, rho_res: float = 1.0):
    """The discrete Hough accumulator A[ρ, θ] (rows = ρ bins, columns = θ bins, θ ∈ [0°, 180°)).

    For every point and every θ_k:  ρ = x cos θ_k + y sin θ_k  ->  nearest ρ bin gets +1."""
    thetas = np.arange(n_theta) * (180.0 / n_theta)
    rhos = rho_axis(w, h, rho_res)
    A = np.zeros((len(rhos), n_theta), np.int32)
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) == 0:
        return A, thetas, rhos
    r = rho_of(P[:, 0:1], P[:, 1:2], np.deg2rad(thetas)[None, :])
    ri = np.round((r - rhos[0]) / rho_res).astype(int)
    ti = np.broadcast_to(np.arange(n_theta)[None, :], ri.shape)
    ok = (ri >= 0) & (ri < len(rhos))
    np.add.at(A, (ri[ok], ti[ok]), 1)
    return A, thetas, rhos


def find_peaks(A: np.ndarray, k: int = 5, threshold: float = 1, nms: tuple[int, int] = (9, 9),
               wrap_cols: bool = False) -> list[tuple[int, int, int]]:
    """Local maxima of an accumulator, strongest first: (row, col, votes).

    A bin is a peak if it is the largest value in its nms window (non-maximum suppression, like Canny's NMS
    but in parameter space) and has at least `threshold` votes. wrap_cols: θ wraps around (θ = 0° ~ 180°)."""
    A = np.asarray(A, np.float32)
    kr, kc = nms
    pad_c = kc // 2
    src = np.pad(A, ((0, 0), (pad_c, pad_c)), mode="wrap") if wrap_cols else A
    dil = cv2.dilate(src, np.ones((kr, kc), np.uint8))
    if wrap_cols:
        dil = dil[:, pad_c:pad_c + A.shape[1]]
    cand = np.argwhere((A >= dil) & (A >= threshold) & (A > 0))
    cand = sorted(cand.tolist(), key=lambda rc: -A[rc[0], rc[1]])
    out = []
    for r, c in cand:           # plateaus: keep only one bin per window
        if all(abs(r - r2) > kr // 2 or abs(c - c2) > kc // 2 for r2, c2, _ in out):
            out.append((r, c, int(A[r, c])))
        if len(out) >= k:
            break
    return out


def cv_hough_lines(P: np.ndarray, w: int, h: int, rho_res: float = 1.0, n_theta: int = 180, threshold: int = 10,
                   k: int = 5):
    """cv2.HoughLines on the same points: list of (rho, theta_deg)."""
    img = points_image(P, w, h)
    L = cv2.HoughLines(img, rho_res, np.pi / n_theta, threshold)
    if L is None:
        return []
    return [(float(r), float(np.rad2deg(t))) for r, t in L.reshape(-1, 2)[:k]]


def angle_error(theta1, theta2) -> float:
    """Smallest difference between two line angles (degrees, modulo 180°)."""
    d = abs((theta1 - theta2) % 180.0)
    return min(d, 180.0 - d)


def ransac_line(P: np.ndarray, iters: int = 200, tol: float = 2.0, seed: int = 0):
    """RANSAC for comparison: pick 2 points, count inliers within tol, keep the best. Returns (θ°, ρ, inlier mask)."""
    P = np.asarray(P, float)
    rng = np.random.default_rng(seed)
    best = (0.0, 0.0, np.zeros(len(P), bool))
    if len(P) < 2:
        return best
    for _ in range(iters):
        i, j = rng.choice(len(P), 2, replace=False)
        d = P[j] - P[i]
        if np.hypot(*d) < 1e-9:
            continue
        n = np.array([-d[1], d[0]]) / np.hypot(*d)
        rho = float(n @ P[i])
        if rho < 0:
            n, rho = -n, -rho
        inl = np.abs(P @ n - rho) < tol
        if inl.sum() > best[2].sum():
            th = np.rad2deg(np.arctan2(n[1], n[0])) % 360
            if th >= 180:
                th, rho = th - 180, -rho
            best = (float(th), rho, inl)
    return best


# --------------------------------------------------------------------------
# 4. Circles: (x − a)² + (y − b)² = r²
# --------------------------------------------------------------------------


def _vote_2d(A: np.ndarray, pid: np.ndarray, ax: np.ndarray, by: np.ndarray) -> None:
    """Add one vote per (point, bin) - a point never votes twice for the same bin."""
    h, w = A.shape
    ai, bi = np.round(ax).astype(int), np.round(by).astype(int)
    ok = (ai >= 0) & (ai < w) & (bi >= 0) & (bi < h)
    key = np.unique(pid[ok].astype(np.int64) * (h * w) + bi[ok] * w + ai[ok])
    flat = key % (h * w)
    np.add.at(A.reshape(-1), flat, 1)


def accumulator_circle(P: np.ndarray, r: float, w: int, h: int, n_angles: int = 120) -> np.ndarray:
    """Fixed radius r: every edge point votes for all centres at distance r - a circle in (a, b) space."""
    A = np.zeros((h, w), np.int32)
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) == 0 or r <= 0:
        return A
    phi = np.linspace(0, 2 * np.pi, max(8, int(n_angles)), endpoint=False)
    ax = P[:, 0:1] + r * np.cos(phi)[None, :]
    by = P[:, 1:2] + r * np.sin(phi)[None, :]
    pid = np.broadcast_to(np.arange(len(P))[:, None], ax.shape)
    _vote_2d(A, pid.ravel(), ax.ravel(), by.ravel())
    return A


def accumulator_circle_3d(P: np.ndarray, radii, w: int, h: int, n_angles: int = 90) -> np.ndarray:
    """Unknown radius: A(a, b, r) - one 2D slice per radius (a cone of votes per point). Shape (R, h, w)."""
    return np.stack([accumulator_circle(P, r, w, h, n_angles=max(16, int(n_angles * r / max(radii)))) for r in radii])


def accumulator_circle_gradient(P: np.ndarray, theta: np.ndarray, r: float, w: int, h: int,
                                sign: str = "both") -> np.ndarray:
    """With the gradient direction θ: the centre must lie along it, at distance r:
    (a, b) = (x ∓ r cos θ, y ∓ r sin θ). sign = 'minus' (gradient points outward), 'plus' (inward) or 'both'."""
    A = np.zeros((h, w), np.int32)
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) == 0:
        return A
    signs = {"minus": [-1.0], "plus": [1.0], "both": [-1.0, 1.0]}[sign]
    for s in signs:
        ax = P[:, 0] + s * r * np.cos(theta)
        by = P[:, 1] + s * r * np.sin(theta)
        _vote_2d(A, np.arange(len(P)), ax, by)
    return A


def accumulator_circle_gradient_3d(P, theta, radii, w, h, sign="both") -> np.ndarray:
    return np.stack([accumulator_circle_gradient(P, theta, r, w, h, sign) for r in radii])


def cv_hough_circles(gray01: np.ndarray, min_r: int, max_r: int, min_dist: float = 20, canny_high: float = 120,
                     acc_thresh: float = 30, dp: float = 1.0):
    """cv2.HoughCircles (the two-stage gradient method): list of (x, y, r)."""
    u8 = np.clip(gray01 * 255, 0, 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (0, 0), 1.5)
    C = cv2.HoughCircles(u8, cv2.HOUGH_GRADIENT, dp, min_dist, param1=canny_high, param2=acc_thresh,
                         minRadius=int(min_r), maxRadius=int(max_r))
    return [] if C is None else [tuple(map(float, c)) for c in C.reshape(-1, 3)]


def circles_image(size=(200, 280), circles=((80, 90, 42), (200, 120, 30)), noise=0.03, seed=2, occlude=True):
    """Grey test image: dark discs on a bright background (gradient points outward), one partly occluded."""
    h, w = size
    img = np.full((h, w), 0.8)
    for cx, cy, r in circles:
        cv2.circle(img, (int(cx), int(cy)), int(r), 0.25, -1)
    if occlude:
        cv2.rectangle(img, (int(circles[0][0]) + 5, int(circles[0][1]) - 60), (int(circles[0][0]) + 60,
                      int(circles[0][1]) - 10), 0.8, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    if noise > 0:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 1)


# --------------------------------------------------------------------------
# 5. Display
# --------------------------------------------------------------------------

HEAT = [[0.0, "#16161a"], [0.25, "#3d2a5c"], [0.5, "#a3346b"], [0.75, "#ea7a2e"], [1.0, "#fde68a"]]


def draw_lines_rgb(rgb: np.ndarray, lines: list[tuple[float, float]], color=(237, 161, 0), thickness=2,
                   scale: int = 1) -> np.ndarray:
    """Draw (ρ, θ°) lines across an image (image coordinates)."""
    out = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale != 1 else rgb.copy()
    h, w = rgb.shape[:2]
    for rho, th in lines:
        seg = clip_line(th, rho, w - 1, h - 1)
        if seg is None:
            continue
        (x0, y0), (x1, y1) = seg
        cv2.line(out, (int(round((x0 + 0.5) * scale)), int(round((y0 + 0.5) * scale))),
                 (int(round((x1 + 0.5) * scale)), int(round((y1 + 0.5) * scale))), color, thickness, cv2.LINE_AA)
    return out


def draw_circles_rgb(rgb: np.ndarray, circles, color=(237, 161, 0), thickness=2) -> np.ndarray:
    out = rgb.copy()
    for x, y, r in circles:
        cv2.circle(out, (int(round(x)), int(round(y))), int(round(r)), color, thickness, cv2.LINE_AA)
        cv2.circle(out, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)
    return out
