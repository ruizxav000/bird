"""Signed-distance-field modelling helpers.

All dimensions are metres in bird space: +Y forward (bill direction),
+Z up, +X the bird's right side, origin at the grip point on the perch.
Proportions were measured off the reference frames (total length
bill-tip to streamer-tip ~0.37 m, crown ~0.15 m above the perch).
"""
import numpy as np
from skimage import measure


def smin(a, b, k):
    """Polynomial smooth minimum (smooth union)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def rot_x(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ellipsoid(P, center, radii, R=None):
    q = P - np.asarray(center)
    if R is not None:
        q = q @ R  # into ellipsoid frame (R columns are its axes)
    r = np.asarray(radii)
    k0 = np.linalg.norm(q / r, axis=-1)
    k1 = np.linalg.norm(q / (r * r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def capsule(P, a, b, ra, rb):
    """Round cone between points a and b with radii ra, rb."""
    a = np.asarray(a); b = np.asarray(b)
    ba = b - a
    pa = P - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    d = np.linalg.norm(pa - h[..., None] * ba, axis=-1)
    return d - (ra + (rb - ra) * h)


def sphere(P, c, r):
    return np.linalg.norm(P - np.asarray(c), axis=-1) - r


def polygonize(fn, lo, hi, step):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    n = np.ceil((hi - lo) / step).astype(int) + 1
    xs = [lo[i] + np.arange(n[i]) * step for i in range(3)]
    G = np.stack(np.meshgrid(*xs, indexing="ij"), axis=-1)
    vol = fn(G)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(step,) * 3)
    verts += lo
    # marching_cubes winds faces so normals point toward lower values; flip
    faces = faces[:, ::-1]
    return verts, faces


# ---------------------------------------------------------------- the bird
BODY_TILT = 55.0  # torso long axis, degrees above horizontal

HEAD_PIVOT = np.array([0.0, 0.010, 0.100])
EYE_C = np.array([0.0196, 0.0300, 0.1300])  # right eye centre (mirror for left)
EYE_R = 0.0052
BILL_BASE = np.array([0.0, 0.0470, 0.1255])


def body_sdf(P):
    R = rot_x(BODY_TILT)
    torso = ellipsoid(P, (0, -0.014, 0.060), (0.037, 0.060, 0.041), R)
    breast = ellipsoid(P, (0, 0.020, 0.076), (0.035, 0.030, 0.040))
    belly = ellipsoid(P, (0, 0.004, 0.036), (0.032, 0.031, 0.030))
    tail = capsule(P, (0, -0.042, 0.026), (0, -0.080, -0.008), 0.014, 0.006)
    neck = capsule(P, (0, -0.002, 0.080), (0, 0.010, 0.112), 0.024, 0.021)
    d = smin(torso, breast, 0.020)
    d = smin(d, belly, 0.020)
    d = smin(d, tail, 0.015)
    d = smin(d, neck, 0.018)
    return d


def head_sdf(P):
    skull = ellipsoid(P, (0, 0.019, 0.1285), (0.0235, 0.0295, 0.0215))
    # face narrows into the bill base
    face = capsule(P, (0, 0.030, 0.1275), BILL_BASE + (0, 0.004, 0), 0.0150, 0.0062)
    # throat / chin pulled down and forward
    chin = capsule(P, (0, 0.024, 0.112), (0, 0.040, 0.118), 0.0150, 0.0075)
    # neck plug: a near-cylinder about the pivot axis so turning never gaps
    plug = capsule(P, HEAD_PIVOT + (0, 0, 0.022), HEAD_PIVOT - (0, 0, 0.012), 0.0205, 0.0215)
    d = smin(skull, face, 0.010)
    d = smin(d, chin, 0.010)
    d = smin(d, plug, 0.012)
    # eye sockets: shallow recess so lids/feathers frame the eye
    for sx in (1, -1):
        c = EYE_C * (sx, 1, 1)
        d = smax(d, -sphere(P, c + (0.0012 * sx, 0, 0), EYE_R * 0.92), 0.003)
    return d


def build_body():
    return polygonize(body_sdf, (-0.05, -0.10, -0.03), (0.05, 0.08, 0.13), 0.0012)


def build_head():
    return polygonize(head_sdf, (-0.035, -0.025, 0.07), (0.035, 0.065, 0.16), 0.0008)
