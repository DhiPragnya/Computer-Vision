"""Edge detection, written to be read by students.

Conventions
-----------
* Images are float arrays in [0, 1] (grey) indexed img[row, col] = I(x = col, y = row).
* y therefore points DOWN. Iy > 0 means "brighter as you go down the image".
* Kernels are applied the way OpenCV (and most libraries) do it: by CORRELATION - the kernel is
  laid on the image as printed, without flipping. (True convolution flips it; for the symmetric
  Gaussian and Laplacian nothing changes, for Sobel only the sign flips.)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "images"

# --------------------------------------------------------------------------
# 1. Images
# --------------------------------------------------------------------------


def to_gray(img) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8) if img.dtype != np.uint8 else img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float64)
    return img / 255.0 if img.max() > 1.0 else img


@lru_cache(maxsize=16)
def _load_png(name: str) -> np.ndarray:
    img = cv2.imread(str(ASSETS / name), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(ASSETS / name)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def photo(name: str) -> np.ndarray:
    """Bundled photo as a grey float image in [0, 1]."""
    return to_gray(_load_png(name))


def photo_rgb(name: str) -> np.ndarray:
    img = _load_png(name)
    return img if img.ndim == 3 else np.dstack([img] * 3)


def shapes_image(size: int = 256) -> np.ndarray:
    """Synthetic scene with strong AND weak edges, a curve, a corner and a thin line."""
    img = np.full((size, size), 0.25)
    s = size / 256
    cv2.rectangle(img, (int(20 * s), int(25 * s)), (int(115 * s), int(120 * s)), 0.85, -1)
    cv2.circle(img, (int(185 * s), int(75 * s)), int(48 * s), 0.6, -1)
    tri = np.array([[30, 230], [120, 230], [75, 150]]) * s
    cv2.fillPoly(img, [np.int32(tri)], 0.7)
    cv2.rectangle(img, (int(150 * s), int(150 * s)), (int(235 * s), int(235 * s)), 0.32, -1)   # weak edge
    cv2.line(img, (int(140 * s), int(140 * s)), (int(245 * s), int(140 * s)), 0.75, max(1, int(2 * s)))
    return img


def step_image(size: int = 128, angle_deg: float = 0.0, blur: float = 0.0, lo: float = 0.2, hi: float = 0.8,
               offset: float = 0.0):
    """A straight step edge through the centre; the edge LINE makes angle_deg with the vertical."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2
    a = np.deg2rad(angle_deg)
    d = (xx - c) * np.cos(a) - (yy - c) * np.sin(a) - offset  # signed distance to the edge line
    if blur > 0:
        from math import erf
        v = np.vectorize(lambda t: 0.5 * (1 + erf(t / (np.sqrt(2) * blur))))
        prof = v(d)
    else:
        prof = (d >= 0).astype(float)
    return lo + (hi - lo) * prof


def texture_scene(size: int = 256, cx: float = 0.55, cy: float = 0.5, r: float = 0.28, period: float = 3.0,
                  wobble: float = 17.0, light_dir: float = 1.0, return_gt: bool = False):
    """An object (disc) on a strongly textured background, lit by a gradient - where classical detectors struggle.

    With return_gt=True also returns the true one-pixel object boundary (known because we drew it)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    stripes = 0.5 + 0.18 * np.sign(np.sin(xx / period + 0.6 * np.sin(yy / wobble)))
    light = 0.55 + 0.45 * (xx / size if light_dir > 0 else 1 - xx / size)
    img = stripes * light
    disc = (xx - size * cx) ** 2 + (yy - size * cy) ** 2 < (size * r) ** 2
    img[disc] = 0.5 * light[disc] + 0.06
    img = np.clip(img, 0, 1)
    if not return_gt:
        return img
    d8 = disc.astype(np.uint8)
    gt = (cv2.morphologyEx(d8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & disc
    return img, gt


def feature_layers(img: np.ndarray, sigma: float = 1.5, pool: float = 6.0, n_orient: int = 4) -> dict:
    """A tiny HAND-MADE 'network': oriented filters → rectification → pooling → boundary of the features.

    It imitates what the first layers of a CNN compute (filter, ReLU-like non-linearity, pooling); a real
    network LEARNS its filters from data and stacks many such layers."""
    Is = smooth(img, sigma)
    gx, gy = gradients(Is, "Central difference")
    responses, energies = [], []
    for k in range(n_orient):
        a = np.pi * k / n_orient
        r = np.cos(a) * gx + np.sin(a) * gy                 # layer 1: oriented derivative filters
        responses.append(r)
        energies.append(smooth(r ** 2, pool))               # layer 2+3: square (rectify) and pool
    F = np.stack(energies)
    F = F / (F.max() + 1e-12)
    bx = np.zeros_like(img)
    by = np.zeros_like(img)
    for ch in F:                                            # gradient of the feature maps
        cx_, cy_ = gradients(smooth(ch, 2.0), "Central difference")
        bx += cx_ ** 2
        by += cy_ ** 2
    boundary = np.sqrt(bx + by)
    gx2, gy2 = gradients(smooth(F.sum(0), 2.0), "Central difference")
    return dict(responses=responses, energies=list(F), boundary=boundary / (boundary.max() + 1e-12),
                theta=orientation(gx2, gy2))


def add_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    return img + rng.normal(0.0, sigma, img.shape)


SAMPLES = {
    "Shapes (synthetic)": lambda: shapes_image(256),
    "Cameraman (photo)": lambda: photo("cameraman.png"),
    "Coins (photo)": lambda: photo("coins.png"),
    "Coffee (photo)": lambda: photo("coffee.png"),
    "Cat (photo)": lambda: photo("cat.png"),
    "Astronaut (photo)": lambda: photo("astronaut.png"),
    "Bricks (texture)": lambda: photo("bricks.png"),
}


def load_upload(file, max_side: int = 400) -> np.ndarray:
    from PIL import Image
    import io

    im = Image.open(io.BytesIO(file.getvalue())).convert("L")
    im.thumbnail((max_side, max_side))
    return np.asarray(im, dtype=np.float64) / 255.0


# --------------------------------------------------------------------------
# 2. Kernels
# --------------------------------------------------------------------------

K = {
    "forward_x": np.array([[-1.0, 1.0]]),
    "forward_y": np.array([[-1.0], [1.0]]),
    "central_x": np.array([[-1.0, 0.0, 1.0]]) / 2,
    "central_y": np.array([[-1.0], [0.0], [1.0]]) / 2,
    "sobel_x": np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "sobel_y": np.array([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "prewitt_x": np.array([[-1.0, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    "prewitt_y": np.array([[-1.0, -1, -1], [0, 0, 0], [1, 1, 1]]),
    "roberts_x": np.array([[1.0, 0], [0, -1]]),
    "roberts_y": np.array([[0.0, 1], [-1, 0]]),
    "laplace4": np.array([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    "laplace8": np.array([[1.0, 1, 1], [1, -8, 1], [1, 1, 1]]),
}


def derivative_smoothing_kernel(w: float) -> tuple[np.ndarray, np.ndarray]:
    """Gx = S Dᵀ with S = (1, w, 1)ᵀ and D = (−1, 0, 1).  w = 1 → Prewitt, w = 2 → Sobel."""
    S = np.array([[1.0], [w], [1.0]])
    D = np.array([[-1.0, 0.0, 1.0]])
    return S @ D, D.T @ S.T


def gaussian_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    radius = int(np.ceil(3 * sigma)) if radius is None else radius
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_2d(sigma: float, radius: int | None = None) -> np.ndarray:
    g = gaussian_1d(sigma, radius)
    return np.outer(g, g)


def gaussian_formula(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)


def log_formula(x, y, sigma):
    """LoG(x, y) = −1/(πσ⁴) · (1 − r²/(2σ²)) · exp(−r²/(2σ²))   (= ∇²G, negative centre)."""
    r2 = x ** 2 + y ** 2
    return -1.0 / (np.pi * sigma ** 4) * (1 - r2 / (2 * sigma ** 2)) * np.exp(-r2 / (2 * sigma ** 2))


def log_kernel(sigma: float, radius: int | None = None, zero_sum: bool = True) -> np.ndarray:
    radius = int(np.ceil(4 * sigma)) if radius is None else radius
    x = np.arange(-radius, radius + 1, dtype=float)
    X, Y = np.meshgrid(x, x)
    k = log_formula(X, Y, sigma)
    if zero_sum:                       # truncation leaves a tiny DC response; remove it
        k = k - k.mean()
    return k


def binomial(n: int) -> np.ndarray:
    """Row n of Pascal's triangle, normalised - (1,1) convolved with itself n times."""
    k = np.array([1.0])
    for _ in range(n):
        k = np.convolve(k, [1.0, 1.0])
    return k / k.sum()


# --------------------------------------------------------------------------
# 3. Filtering
# --------------------------------------------------------------------------


def correlate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide the kernel over the image as printed (no flip) - what cv2.filter2D does."""
    k = np.asarray(kernel, dtype=np.float64)
    anchor = (k.shape[1] // 2 if k.shape[1] % 2 else k.shape[1] // 2 - 1,
              k.shape[0] // 2 if k.shape[0] % 2 else k.shape[0] // 2 - 1)
    return cv2.filter2D(np.asarray(img, dtype=np.float64), cv2.CV_64F, k, anchor=anchor,
                        borderType=cv2.BORDER_REFLECT)


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """True convolution = correlation with the kernel rotated by 180°."""
    return correlate(img, np.flip(np.asarray(kernel, dtype=np.float64)))


def smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(img, dtype=np.float64).copy()
    return cv2.GaussianBlur(np.asarray(img, dtype=np.float64), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def box_blur(img: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return img.copy()
    return cv2.blur(np.asarray(img, dtype=np.float64), (size, size), borderType=cv2.BORDER_REFLECT)


OPERATORS = {
    "Forward difference": ("forward_x", "forward_y"),
    "Central difference": ("central_x", "central_y"),
    "Roberts": ("roberts_x", "roberts_y"),
    "Prewitt": ("prewitt_x", "prewitt_y"),
    "Sobel": ("sobel_x", "sobel_y"),
}


def gradients(img: np.ndarray, operator: str = "Sobel") -> tuple[np.ndarray, np.ndarray]:
    kx, ky = OPERATORS[operator]
    return correlate(img, K[kx]), correlate(img, K[ky])


def magnitude(Ix, Iy, norm: str = "L2") -> np.ndarray:
    return np.hypot(Ix, Iy) if norm == "L2" else np.abs(Ix) + np.abs(Iy)


def orientation(Ix, Iy) -> np.ndarray:
    """Gradient direction θ = atan2(Iy, Ix) in radians (use atan2, never atan(Iy/Ix))."""
    return np.arctan2(Iy, Ix)


def laplacian(img: np.ndarray, kind: int = 4) -> np.ndarray:
    return correlate(img, K["laplace4" if kind == 4 else "laplace8"])


def log_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    return correlate(img, log_kernel(sigma))


def zero_crossings(R: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Mark pixel p if R changes sign between p and a neighbour (right, down, two diagonals)
    and the jump |R(p) − R(q)| is larger than `threshold` (to ignore tiny noise crossings)."""
    zc = np.zeros(R.shape, dtype=bool)
    h, w = R.shape
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        r0, r1 = 0, h - dr
        c0, c1 = max(0, -dc), w - max(0, dc)
        a = R[r0:r1, c0:c1]
        b = R[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        zc[r0:r1, c0:c1] |= (a * b < 0) & (np.abs(a - b) > threshold)
    return zc


# --------------------------------------------------------------------------
# 4. Non-maximum suppression
# --------------------------------------------------------------------------

# neighbour offsets (dcol, drow) for each quantised gradient direction, IMAGE coordinates (y down)
NMS_OFFSETS = {0: ((-1, 0), (1, 0)), 45: ((-1, -1), (1, 1)), 90: ((0, -1), (0, 1)), 135: ((1, -1), (-1, 1))}


def quantize(theta: np.ndarray) -> np.ndarray:
    """Map gradient direction (radians) to the nearest of 0°, 45°, 90°, 135° (directions mod 180°)."""
    deg = np.degrees(theta) % 180.0
    q = (np.round(deg / 45.0) * 45) % 180
    return q.astype(int)


def _shift(M: np.ndarray, dc: int, dr: int) -> np.ndarray:
    """Value of the neighbour at (col + dc, row + dr), zero outside the image."""
    out = np.zeros_like(M)
    h, w = M.shape
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w, w - dc)
    out[r0:r1, c0:c1] = M[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def nms_quantized(M: np.ndarray, theta: np.ndarray) -> np.ndarray:
    q = quantize(theta)
    keep = np.zeros(M.shape, dtype=bool)
    for ang, ((dc1, dr1), (dc2, dr2)) in NMS_OFFSETS.items():
        sel = q == ang
        # ">" on one side and ">=" on the other breaks ties on flat-topped ridges (keeps exactly one pixel)
        keep |= sel & (M > _shift(M, dc1, dr1)) & (M >= _shift(M, dc2, dr2))
    return np.where(keep, M, 0.0)


def nms_interpolated(M: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compare with M at (x ± cos θ, y ± sin θ), using bilinear interpolation."""
    h, w = M.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    Mf = M.astype(np.float32)
    plus = cv2.remap(Mf, xx + c, yy + s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    minus = cv2.remap(Mf, xx - c, yy - s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where((Mf >= plus) & (Mf > minus), M, 0.0)


# --------------------------------------------------------------------------
# 5. Double threshold and hysteresis
# --------------------------------------------------------------------------

NONE, WEAK, STRONG = 0, 1, 2


def double_threshold(Mn: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    c = np.zeros(Mn.shape, dtype=np.uint8)
    c[(Mn >= t_low) & (Mn > 0)] = WEAK
    c[Mn >= t_high] = STRONG
    return c


def hysteresis(classes: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep weak pixels connected (through S ∪ W) to at least one strong pixel - via connected components."""
    cand = (classes > 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=connectivity)
    strong_labels = np.unique(lab[classes == STRONG])
    return np.isin(lab, strong_labels[strong_labels > 0])


def hysteresis_steps(classes: np.ndarray, max_steps: int = 400, connectivity: int = 8) -> list[np.ndarray]:
    """E(0) = S;  E(k+1) = E(k) ∪ {p ∈ W : N(p) ∩ E(k) ≠ ∅}  - returns every E(k) until it stops growing."""
    k = np.ones((3, 3), np.uint8) if connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    E = (classes == STRONG).astype(np.uint8)
    W = (classes == WEAK).astype(np.uint8)
    steps = [E.astype(bool)]
    for _ in range(max_steps):
        grown = E | (cv2.dilate(E, k) & W)
        if np.array_equal(grown, E):
            break
        E = grown
        steps.append(E.astype(bool))
    return steps


def canny(img: np.ndarray, sigma: float = 1.4, t_low: float = 0.1, t_high: float = 0.25, operator: str = "Sobel",
          nms: str = "quantized", relative: bool = True) -> dict:
    """The full pipeline, returning every intermediate stage.

    With relative=True the thresholds are fractions of the largest NMS magnitude (easier sliders).
    """
    Is = smooth(img, sigma)
    Ix, Iy = gradients(Is, operator)
    M = magnitude(Ix, Iy)
    theta = orientation(Ix, Iy)
    Mn = nms_quantized(M, theta) if nms == "quantized" else nms_interpolated(M, theta)
    scale = Mn.max() if (relative and Mn.max() > 0) else 1.0
    classes = double_threshold(Mn, t_low * scale, t_high * scale)
    edges = hysteresis(classes)
    return dict(smoothed=Is, Ix=Ix, Iy=Iy, M=M, theta=theta, Mn=Mn, classes=classes, edges=edges, scale=scale)


def cv_canny(img: np.ndarray, sigma: float, t_low: float, t_high: float) -> np.ndarray:
    """OpenCV's Canny on the same smoothed image; thresholds given as fractions of the max L2 Sobel magnitude."""
    Is = smooth(img, sigma)
    u8 = np.clip(Is * 255, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    mmax = np.hypot(gx, gy).max() or 1.0
    return cv2.Canny(u8, t_low * mmax, t_high * mmax, apertureSize=3, L2gradient=True) > 0


# --------------------------------------------------------------------------
# 6. Evaluation
# --------------------------------------------------------------------------


def match_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 1) -> tuple[float, float, float]:
    """Precision / recall / F1 with a tolerance of `tol` pixels (a predicted pixel near a true one counts)."""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    gt_d = cv2.dilate(gt, k)
    pred_d = cv2.dilate(pred, k)
    p = (pred & gt_d).sum() / max(pred.sum(), 1)
    r = (gt & pred_d).sum() / max(gt.sum(), 1)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return float(p), float(r), float(f)


# --------------------------------------------------------------------------
# 7. Display helpers (all return uint8 RGB images for st.image)
# --------------------------------------------------------------------------

def to_u8(img: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    lo = float(np.min(img)) if lo is None else lo
    hi = float(np.max(img)) if hi is None else hi
    out = (np.asarray(img, float) - lo) / (hi - lo + 1e-12)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


_NEG = np.array([42, 120, 214], float)     # blue  (negative)
_MID = np.array([240, 239, 236], float)    # neutral
_POS = np.array([227, 73, 72], float)      # red   (positive)


def signed_rgb(R: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Diverging colours: blue < 0 < red, neutral grey at 0 - so the SIGN of a response is visible."""
    vmax = float(np.percentile(np.abs(R), 99.5)) if vmax is None else vmax
    t = np.clip(R / (vmax + 1e-12), -1, 1)[..., None]
    out = np.where(t < 0, _MID + (-t) * (_NEG - _MID), _MID + t * (_POS - _MID))
    return out.astype(np.uint8)


def orientation_rgb(theta: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Hue = gradient direction, brightness = magnitude."""
    hue = ((np.degrees(theta) % 360) / 2).astype(np.uint8)              # OpenCV hue is 0..179
    val = to_u8(np.sqrt(mag / (mag.max() + 1e-12)))
    hsv = np.dstack([hue, np.full_like(hue, 230), val])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def classes_rgb(classes: np.ndarray) -> np.ndarray:
    out = np.zeros(classes.shape + (3,), np.uint8)
    out[...] = (26, 26, 25)
    out[classes == WEAK] = (237, 161, 0)       # amber  = weak
    out[classes == STRONG] = (255, 255, 255)   # white  = strong
    return out


def overlay(gray: np.ndarray, mask: np.ndarray, color=(235, 104, 52), dim: float = 0.55) -> np.ndarray:
    base = to_u8(gray, 0, 1).astype(float) * dim + 255 * (1 - dim) * 0.35
    rgb = np.dstack([base] * 3)
    rgb[mask] = color
    return rgb.clip(0, 255).astype(np.uint8)


def edges_rgb(mask: np.ndarray) -> np.ndarray:
    """White edges on black."""
    return np.dstack([(mask * 255).astype(np.uint8)] * 3)


def enlarge(img: np.ndarray, factor: int) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
