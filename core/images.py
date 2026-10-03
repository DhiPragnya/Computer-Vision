"""Sample images and image warping.

Two ways to warp an image with a transform H (pixel coords: x right, y DOWN):

* forward mapping  - push every SOURCE pixel to round(H x). Simple, but leaves holes.
* inverse mapping  - for every OUTPUT pixel x', pull the colour from H^-1 x'. No holes,
                     but H^-1 x' is rarely an integer -> we must INTERPOLATE.

Both are written in plain NumPy so students can read them; OpenCV is used for
comparison (cv2.warpPerspective does inverse mapping internally).
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from . import transforms as tf

INTERP_CV = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


# --------------------------------------------------------------------------
# sample images (generated, so the app works offline)
# --------------------------------------------------------------------------

def checkerboard(size: int = 320, n: int = 8) -> np.ndarray:
    cell = size // n
    img = np.zeros((size, size, 3), np.uint8)
    for i in range(n):
        for j in range(n):
            light = (i + j) % 2 == 0
            img[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = (240, 240, 236) if light else (52, 64, 84)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (235, 104, 52), 4)
    # mark the top-left so orientation is obvious
    cv2.circle(img, (cell // 2, cell // 2), cell // 4, (235, 104, 52), -1)
    return img


def poster(size: int = 320) -> np.ndarray:
    """A small 'scene': graph paper, a big asymmetric letter F, a circle, a triangle and text."""
    img = np.full((size, size, 3), 250, np.uint8)
    step = size // 16
    for k in range(0, size, step):
        cv2.line(img, (k, 0), (k, size), (222, 226, 232), 1)
        cv2.line(img, (0, k), (size, k), (222, 226, 232), 1)
    s = size / 320
    # letter F
    f = np.array([[40, 270], [80, 270], [80, 180], [150, 180], [150, 145], [80, 145], [80, 105],
                  [165, 105], [165, 70], [40, 70]], np.int32)
    cv2.fillPoly(img, [np.int32(f * s)], (42, 120, 214))
    cv2.circle(img, (int(235 * s), int(110 * s)), int(42 * s), (235, 104, 52), -1)
    tri = np.array([[190, 275], [290, 275], [240, 185]], np.int32)
    cv2.fillPoly(img, [np.int32(tri * s)], (27, 175, 122))
    cv2.putText(img, "CV", (int(205 * s), int(55 * s)), cv2.FONT_HERSHEY_DUPLEX, 1.3 * s, (40, 40, 40), 2,
                cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (60, 60, 60), 2)
    return img


def document(size: int = 320) -> np.ndarray:
    """A frontal 'page' with a title and lines of text (used for rectification demos)."""
    h, w = int(size * 1.25), size
    img = np.full((h, w, 3), 252, np.uint8)
    cv2.rectangle(img, (18, 18), (w - 18, 64), (42, 120, 214), -1)
    cv2.putText(img, "LAB NOTES", (32, 52), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    y = 92
    rng = np.random.default_rng(3)
    while y < h - 40:
        length = int(rng.uniform(0.55, 0.92) * (w - 48))
        cv2.line(img, (24, y), (24 + length, y), (120, 120, 120), 5, cv2.LINE_AA)
        y += 22
    cv2.rectangle(img, (w - 110, h - 100), (w - 24, h - 30), (235, 104, 52), 3)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 40, 40), 2)
    return img


def photo_of_document(doc: np.ndarray, corners_frac, canvas=(360, 480)):
    """Pretend we photographed `doc` at an angle: warp it onto a textured 'desk'.

    corners_frac: 4 (x, y) corners in [0,1] of the canvas (TL, TR, BR, BL).
    Returns the photo and the true corner pixels.
    """
    H_c, W_c = canvas
    yy, xx = np.mgrid[0:H_c, 0:W_c]
    desk = (np.stack([xx * 0 + 168, xx * 0 + 132, xx * 0 + 98], -1)
            + 14 * np.sin(xx[..., None] / 9.0 + 0.4 * np.sin(yy[..., None] / 23.0))).clip(0, 255).astype(np.uint8)
    h, w = doc.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = np.array(corners_frac, np.float32) * np.array([W_c - 1, H_c - 1], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, H, (W_c, H_c), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (W_c, H_c)) > 0
    out = desk.copy()
    out[mask] = warped[mask]
    return out, dst


SAMPLES = {"Poster (letter F)": poster, "Checkerboard": checkerboard}


def load_upload(file, max_side: int = 420) -> np.ndarray:
    img = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
    img.thumbnail((max_side, max_side))
    return np.array(img)


# --------------------------------------------------------------------------
# warping
# --------------------------------------------------------------------------

def about_center(M: np.ndarray, shape) -> np.ndarray:
    """Conjugate M so that it acts about the image centre:  T(c) . M . T(-c)."""
    h, w = shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    return tf.translation(cx, cy) @ M @ tf.translation(-cx, -cy)


def forward_warp(img: np.ndarray, H: np.ndarray, out_shape=None):
    """Push each source pixel to the nearest destination pixel. Returns (image, hole_mask)."""
    h, w = img.shape[:2]
    oh, ow = (out_shape or img.shape)[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    src = np.c_[xs.ravel(), ys.ravel()].astype(float)
    dst = tf.apply(H, src)
    ok = np.isfinite(dst).all(axis=1)
    xd = np.full(len(dst), -1)
    yd = np.full(len(dst), -1)
    xd[ok] = np.rint(dst[ok, 0]).astype(int)
    yd[ok] = np.rint(dst[ok, 1]).astype(int)
    inside = ok & (xd >= 0) & (xd < ow) & (yd >= 0) & (yd < oh)
    out = np.zeros((oh, ow, 3), np.uint8)
    hit = np.zeros((oh, ow), bool)
    out[yd[inside], xd[inside]] = img.reshape(-1, 3)[inside]
    hit[yd[inside], xd[inside]] = True
    return out, ~hit


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    a = (x - x0)[..., None]
    b = (y - y0)[..., None]
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    f = img.astype(float)
    top = (1 - a) * f[y0c, x0c] + a * f[y0c, x1c]
    bot = (1 - a) * f[y1c, x0c] + a * f[y1c, x1c]
    return (1 - b) * top + b * bot


def inverse_warp(img: np.ndarray, H: np.ndarray, out_shape=None, method: str = "bilinear") -> np.ndarray:
    """For every output pixel x', sample the source at H^-1 x'. Pixels that land outside -> black."""
    h, w = img.shape[:2]
    oh, ow = (out_shape or img.shape)[:2]
    if method == "bicubic":               # NumPy bicubic is long; let OpenCV do this one
        return cv2.warpPerspective(img, H, (ow, oh), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    Hinv = np.linalg.inv(H)
    ys, xs = np.mgrid[0:oh, 0:ow]
    src = tf.apply(Hinv, np.c_[xs.ravel(), ys.ravel()].astype(float))
    sx, sy = src[:, 0], src[:, 1]
    ok = np.isfinite(sx) & np.isfinite(sy)
    sx = np.where(ok, sx, -10)
    sy = np.where(ok, sy, -10)
    inside = (sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)
    if method == "nearest":
        vals = img[np.clip(np.rint(sy).astype(int), 0, h - 1), np.clip(np.rint(sx).astype(int), 0, w - 1)].astype(float)
    else:
        vals = _bilinear(img, sx, sy)
    vals[~inside] = 0
    return vals.reshape(oh, ow, 3).clip(0, 255).astype(np.uint8)


def cv_warp(img: np.ndarray, H: np.ndarray, out_shape=None, method: str = "bilinear") -> np.ndarray:
    oh, ow = (out_shape or img.shape)[:2]
    return cv2.warpPerspective(img, H, (ow, oh), flags=INTERP_CV[method], borderValue=(0, 0, 0))


def zoom(img: np.ndarray, cx: float, cy: float, half: int = 24, scale: int = 8) -> np.ndarray:
    """Crop a small window and enlarge it with NEAREST so the real pixels stay visible."""
    h, w = img.shape[:2]
    x0 = int(np.clip(cx - half, 0, max(w - 2 * half, 0)))
    y0 = int(np.clip(cy - half, 0, max(h - 2 * half, 0)))
    crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def draw_polygon(img: np.ndarray, pts, color=(235, 104, 52), thickness: int = 2, labels=None) -> np.ndarray:
    out = img.copy()
    p = np.int32(np.round(np.asarray(pts)))
    cv2.polylines(out, [p], True, color, thickness, cv2.LINE_AA)
    for i, q in enumerate(p):
        cv2.circle(out, tuple(int(v) for v in q), 6, color, -1, cv2.LINE_AA)
        if labels:
            cv2.putText(out, labels[i], (int(q[0]) + 8, int(q[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (20, 20, 20), 2, cv2.LINE_AA)
    return out
