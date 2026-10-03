"""The maths of 2D geometric transformations, written to be read by students.

Conventions
-----------
* Points are rows: an (N, 2) array of (x, y).
* Every transform is a 3x3 matrix acting on homogeneous column vectors
  [x, y, 1]^T, so  p' = H p.  In the text entries are written h11 ... h33
  (row, column, counting from 1) as in the lecture notes; in NumPy that is H[0,0] ... H[2,2].
* Angles in the public helpers are in DEGREES (friendlier for sliders).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9

# --------------------------------------------------------------------------
# 1. Building matrices
# --------------------------------------------------------------------------


def identity() -> np.ndarray:
    return np.eye(3)


def translation(tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty],
                     [0.0, 0.0, 1.0]])


def rotation(theta_deg: float = 0.0) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def scaling(sx: float = 1.0, sy: float | None = None) -> np.ndarray:
    sy = sx if sy is None else sy
    return np.array([[sx, 0.0, 0.0],
                     [0.0, sy, 0.0],
                     [0.0, 0.0, 1.0]])


def shear(kx: float = 0.0, ky: float = 0.0) -> np.ndarray:
    """x' = x + kx*y ,  y' = y + ky*x"""
    return np.array([[1.0, kx, 0.0],
                     [ky, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])


def reflection_x() -> np.ndarray:
    """Mirror across the x-axis (y -> -y). det = -1."""
    return np.diag([1.0, -1.0, 1.0])


def rotation_about(theta_deg: float, cx: float, cy: float) -> np.ndarray:
    """Rotate about the pivot c:  T(c) . R(theta) . T(-c)   (read right to left)."""
    return translation(cx, cy) @ rotation(theta_deg) @ translation(-cx, -cy)


def euclidean(theta_deg: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """Rigid motion: rotate, then translate."""
    return translation(tx, ty) @ rotation(theta_deg)


def similarity(s: float = 1.0, theta_deg: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """Scale, rotate, then translate:  [sR | t]."""
    return translation(tx, ty) @ rotation(theta_deg) @ scaling(s)


def affine(a=1.0, b=0.0, c=0.0, d=1.0, tx=0.0, ty=0.0) -> np.ndarray:
    """x' = a x + b y + tx ,  y' = c x + d y + ty   (lecture-notes notation)."""
    return np.array([[a, b, tx],
                     [c, d, ty],
                     [0.0, 0.0, 1.0]])


def normalize_h(H: np.ndarray) -> np.ndarray:
    """Homographies are defined only up to scale; fix the scale so H[2,2] = 1 (if possible)."""
    H = np.asarray(H, dtype=float)
    if abs(H[2, 2]) > EPS:
        return H / H[2, 2]
    n = np.linalg.norm(H)
    return H / n if n > EPS else H


# --------------------------------------------------------------------------
# 2. Homogeneous coordinates
# --------------------------------------------------------------------------


def to_homogeneous(pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return np.hstack([pts, np.ones((len(pts), 1))])


def from_homogeneous(ph) -> np.ndarray:
    """Divide by w. Points with w ~ 0 are 'at infinity' and come back as NaN."""
    ph = np.asarray(ph, dtype=float).reshape(-1, 3)
    w = ph[:, 2:3]
    out = np.full((len(ph), 2), np.nan)
    ok = np.abs(w[:, 0]) > EPS
    out[ok] = ph[ok, :2] / w[ok]
    return out


def apply_with_w(H, pts):
    """Apply H and also return the w' of every point (before dividing)."""
    ph = to_homogeneous(pts) @ np.asarray(H, dtype=float).T
    return from_homogeneous(ph), ph[:, 2]


def apply(H, pts) -> np.ndarray:
    return apply_with_w(H, pts)[0]


def lerp_from_identity(H, t: float) -> np.ndarray:
    """Straight-line blend between the identity and H (used for 'morph' animations)."""
    return (1.0 - t) * np.eye(3) + t * normalize_h(H)


# --------------------------------------------------------------------------
# 3. The hierarchy of transformations
# --------------------------------------------------------------------------

CLASSES = [
    dict(key="translation", name="Translation", dof=2,
         form=r"\begin{bmatrix} I & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}",
         preserves="orientation"),
    dict(key="euclidean", name="Euclidean (rigid)", dof=3,
         form=r"\begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}",
         preserves="lengths"),
    dict(key="similarity", name="Similarity", dof=4,
         form=r"\begin{bmatrix} sR & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}",
         preserves="angles"),
    dict(key="affine", name="Affine", dof=6,
         form=r"\begin{bmatrix} A & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}",
         preserves="parallelism"),
    dict(key="projective", name="Projective (homography)", dof=8,
         form=r"\tilde{H} \;(3\times3,\ \text{up to scale})",
         preserves="straight lines"),
]
CLASS_BY_KEY = {c["key"]: c for c in CLASSES}
CLASS_ORDER = [c["key"] for c in CLASSES]

# property -> set of classes that preserve it
INVARIANTS = [
    ("Orientation (directions stay the same)", {"translation"}),
    ("Lengths", {"translation", "euclidean"}),
    ("Angles", {"translation", "euclidean", "similarity"}),
    ("Shape (all length ratios)", {"translation", "euclidean", "similarity"}),
    ("Parallel lines stay parallel", {"translation", "euclidean", "similarity", "affine"}),
    ("Midpoints & area ratios", {"translation", "euclidean", "similarity", "affine"}),
    ("Straight lines stay straight", {"translation", "euclidean", "similarity", "affine", "projective"}),
    ("Cross-ratio of 4 collinear points", {"translation", "euclidean", "similarity", "affine", "projective"}),
]


def classify(H, tol: float = 1e-6) -> dict:
    """Return the MOST SPECIFIC class a matrix belongs to, plus a few facts about it."""
    H = normalize_h(H)
    A = H[:2, :2]
    t = H[:2, 2]
    detH = float(np.linalg.det(H))
    info = dict(det=float(np.linalg.det(A)), reflection=False, singular=False)

    if abs(detH) < 1e-9:
        return dict(info, key="degenerate", name="Degenerate (singular - not invertible)",
                    dof=None, singular=True)
    if abs(H[2, 0]) > tol or abs(H[2, 1]) > tol:
        return dict(info, key="projective", name="Projective (homography)", dof=8)
    if np.allclose(A, np.eye(2), atol=tol):
        name = "Identity" if np.allclose(t, 0, atol=tol) else "Translation"
        return dict(info, key="translation", name=name, dof=2)
    AtA = A.T @ A
    s2 = AtA[0, 0]
    if np.allclose(AtA, s2 * np.eye(2), atol=max(tol, tol * s2)):
        refl = info["det"] < 0
        key = "euclidean" if abs(s2 - 1.0) < 1e-6 else "similarity"
        name = CLASS_BY_KEY[key]["name"] + (" + reflection" if refl else "")
        return dict(info, key=key, name=name, dof=CLASS_BY_KEY[key]["dof"], reflection=refl)
    return dict(info, key="affine", name="Affine", dof=6, reflection=info["det"] < 0)


# --------------------------------------------------------------------------
# 4. Lines and invariants
# --------------------------------------------------------------------------


def line_through(p, q) -> np.ndarray:
    """Homogeneous line l = p x q   (l . x = 0 for every point x on the line)."""
    return np.cross(np.append(p, 1.0), np.append(q, 1.0))


def intersect(l1, l2) -> np.ndarray:
    """Homogeneous intersection point x = l1 x l2 (w = 0 means 'at infinity')."""
    return np.cross(l1, l2)


def cross_ratio(a, b, c, d) -> float:
    """Cross-ratio (A,B;C,D) = (AC * BD) / (BC * AD) for 4 collinear points (signed)."""
    pts = np.array([a, b, c, d], dtype=float)
    direction = pts[3] - pts[0]
    direction = direction / (np.linalg.norm(direction) + EPS)
    ta, tb, tc, td = (pts - pts[0]) @ direction
    return float(((tc - ta) * (td - tb)) / ((tc - tb) * (td - ta) + EPS))


# --------------------------------------------------------------------------
# 5. Decompositions (to *explain* a matrix)
# --------------------------------------------------------------------------


def svd_2x2(A):
    """A = U . diag(S) . Vt  -> 'rotate (Vt), stretch along axes (S), rotate (U)'."""
    U, S, Vt = np.linalg.svd(np.asarray(A, dtype=float)[:2, :2])
    # SVD signs are ambiguous: (U, Vt) and (-U, -Vt) give the same A. Pick the one whose
    # first rotation is smallest - it is the easiest to *see*.
    if np.linalg.det(U) < 0 and np.linalg.det(Vt) < 0:     # two reflections: flip one singular pair
        U[:, 1] *= -1
        Vt[1, :] *= -1
    if np.linalg.det(Vt) > 0 and abs(np.degrees(np.arctan2(Vt[1, 0], Vt[0, 0]))) > 90:
        U, Vt = -U, -Vt
    return U, S, Vt


def angle_of(R2) -> float:
    return float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0])))


def decompose_affine(H) -> dict:
    """A = R(theta) . [[sx, m], [0, sy]]  (rotation x upper-triangular 'scale + shear')."""
    H = normalize_h(H)
    A = H[:2, :2]
    Q, U = np.linalg.qr(A)
    # make the diagonal of U positive where possible (QR is unique up to signs)
    signs = np.sign(np.diag(U))
    signs[signs == 0] = 1
    Q = Q * signs
    U = (U.T * signs).T
    if np.linalg.det(Q) < 0:          # keep Q a proper rotation; push the flip into sy
        Q[:, 1] *= -1
        U[1, :] *= -1
    return dict(theta=angle_of(Q), sx=float(U[0, 0]), sy=float(U[1, 1]),
                shear=float(U[0, 1] / U[1, 1]) if abs(U[1, 1]) > EPS else float("nan"),
                tx=float(H[0, 2]), ty=float(H[1, 2]))


# --------------------------------------------------------------------------
# 6. Homography from 4+ correspondences: the Direct Linear Transform (DLT)
# --------------------------------------------------------------------------


def normalization_matrix(pts) -> np.ndarray:
    """Hartley normalisation: centre the points and scale mean distance to sqrt(2)."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d if d > EPS else 1.0
    return np.array([[s, 0, -s * c[0]],
                     [0, s, -s * c[1]],
                     [0, 0, 1.0]])


def dlt_rows(src, dst) -> np.ndarray:
    """Each correspondence (x,y) -> (u,v) gives two linear equations in the 9 entries of H:
        [x, y, 1, 0, 0, 0, -u x, -u y, -u] . h = 0
        [0, 0, 0, x, y, 1, -v x, -v y, -v] . h = 0
    """
    rows = []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float)):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    return np.array(rows)


def homography_dlt(src, dst, normalize: bool = True, details: bool = False):
    """Solve A h = 0 with ||h|| = 1: h is the right singular vector of the smallest singular value."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if normalize:
        T1, T2 = normalization_matrix(src), normalization_matrix(dst)
        s_n, d_n = apply(T1, src), apply(T2, dst)
    else:
        T1 = T2 = np.eye(3)
        s_n, d_n = src, dst
    A = dlt_rows(s_n, d_n)
    _, S, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = normalize_h(np.linalg.inv(T2) @ Hn @ T1)
    if details:
        return H, dict(A=A, singular_values=S, h=Vt[-1], T1=T1, T2=T2)
    return H


# --------------------------------------------------------------------------
# 7. Least-squares fitting of every class + RANSAC
# --------------------------------------------------------------------------


def fit_translation(src, dst):
    t = (np.asarray(dst) - np.asarray(src)).mean(axis=0)
    return translation(*t)


def _umeyama(src, dst, with_scale: bool):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = Xd.T @ Xs / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    c = 1.0
    if with_scale:
        var_s = (Xs ** 2).sum() / n
        c = float(np.trace(np.diag(D) @ S) / var_s) if var_s > EPS else 1.0
    t = mu_d - c * R @ mu_s
    H = np.eye(3)
    H[:2, :2] = c * R
    H[:2, 2] = t
    return H


def fit_euclidean(src, dst):
    return _umeyama(src, dst, with_scale=False)


def fit_similarity(src, dst):
    return _umeyama(src, dst, with_scale=True)


def fit_affine(src, dst):
    X = to_homogeneous(src)                         # (N, 3)
    P, *_ = np.linalg.lstsq(X, np.asarray(dst, float), rcond=None)   # (3, 2)
    H = np.eye(3)
    H[:2, :] = P.T
    return H


def fit_homography(src, dst):
    return homography_dlt(src, dst, normalize=True)


FITTERS = {
    "translation": (fit_translation, 1),
    "euclidean": (fit_euclidean, 2),
    "similarity": (fit_similarity, 2),
    "affine": (fit_affine, 3),
    "projective": (fit_homography, 4),
}


def residuals(H, src, dst) -> np.ndarray:
    pred = apply(H, src)
    err = np.linalg.norm(pred - np.asarray(dst, float), axis=1)
    return np.where(np.isfinite(err), err, np.inf)


def ransac_iterations(outlier_ratio: float, sample_size: int, confidence: float = 0.99) -> int:
    """N = log(1 - p) / log(1 - (1 - e)^s)"""
    inlier_prob = (1.0 - outlier_ratio) ** sample_size
    if inlier_prob >= 1.0:
        return 1
    if inlier_prob <= 0.0:
        return 10 ** 6
    return int(np.ceil(np.log(1 - confidence) / np.log(1 - inlier_prob)))


def ransac(model_key: str, src, dst, threshold: float, iterations: int, seed: int = 0):
    """Classic RANSAC: fit minimal samples, keep the model with most inliers, refit on them."""
    fit, m = FITTERS[model_key]
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    rng = np.random.default_rng(seed)
    n = len(src)
    best_inliers = np.zeros(n, dtype=bool)
    history = []
    for _ in range(iterations):
        idx = rng.choice(n, m, replace=False)
        try:
            H = fit(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue
        inl = residuals(H, src, dst) < threshold
        history.append(int(inl.sum()))
        if inl.sum() > best_inliers.sum():
            best_inliers = inl
    if best_inliers.sum() >= m:
        H = fit(src[best_inliers], dst[best_inliers])
    else:
        H = fit(src, dst)
        best_inliers = np.ones(n, dtype=bool)
    return H, best_inliers, history
