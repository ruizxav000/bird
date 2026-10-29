"""Stout, hook-tipped roller bill, lofted from superellipse sections."""
import numpy as np
from sdf import BILL_BASE

BILL_LEN = 0.0345     # from inside the feathered face to the hook tip
BILL_PITCH = -7.0     # degrees, tip slightly below horizontal


def _ring(w, h_top, h_bot, m=40, n=2.6):
    a = np.linspace(0, 2 * np.pi, m, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    x = w * np.sign(c) * np.abs(c) ** (2 / n)
    z = np.where(s >= 0, h_top, h_bot) * np.sign(s) * np.abs(s) ** (2 / n)
    return x, z, a / (2 * np.pi)


def _loft(sections, tip):
    """sections: list of (centre(3), x(m), z(m), ring_u(m)) in world coords
    with lateral X and up Z offsets already applied."""
    verts, uv, faces = [], [], []
    m = len(sections[0][1])
    for i, (pts, us, t) in enumerate(sections):
        verts.extend(pts)
        uv.extend([(t, u) for u in us])
    k = len(sections)
    for i in range(k - 1):
        for j in range(m):
            a = i * m + j
            b = i * m + (j + 1) % m
            faces.append((a, b, b + m, a + m))
    ti = len(verts)
    verts.append(tip[0])
    uv.append((1.0, 0.5))
    for j in range(m):
        a = (k - 1) * m + j
        faces.append((a, (k - 1) * m + (j + 1) % m, ti))
    # close the base (buried inside the face)
    ci = len(verts)
    verts.append(np.mean(sections[0][0], axis=0))
    uv.append((0.0, 0.5))
    for j in range(m):
        faces.append(((j + 1) % m, j, ci))
    return np.array(verts), faces, np.array(uv)


def build_bill():
    p = np.radians(BILL_PITCH)
    fwd = np.array([0, np.cos(p), np.sin(p)])
    up = np.array([0, -np.sin(p), np.cos(p)])
    lat = np.array([1.0, 0, 0])
    T = np.linspace(0, 1, 34)

    def width(t):
        return 0.0060 * (1 - t) ** 0.75 + 0.0006

    def gape(t):  # gape line drops slightly toward the rictus
        return -0.0010 * (1 - t) ** 3

    def culmen_drop(t):  # decurved culmen + hook
        hook = np.clip((t - 0.80) / 0.20, 0, 1) ** 2
        return -0.0016 * t ** 2 - 0.0026 * hook

    upper, lower = [], []
    for t in T:
        c = BILL_BASE + fwd * (t * BILL_LEN)
        w = width(t)
        # upper mandible: deep, rounded culmen, lateral compression near tip
        hu = 0.0070 * (1 - t) ** 0.85 + 0.0007
        x, z, u = _ring(w * (1 - 0.25 * t), hu, 0.0005 + 0.0012 * (1 - t))
        g = gape(t)
        cu = c + up * (g + culmen_drop(t) * (t > 0) * 0.0)
        # bend the whole section down toward the hook
        drop = culmen_drop(t)
        pts = [cu + lat * xi + up * (zi + drop * (0.4 + 0.6 * (zi > 0))) for xi, zi in zip(x, z)]
        upper.append((pts, u, t))
        tl = t / 0.94
        if tl <= 1.0:
            hl = 0.0050 * (1 - tl) ** 1.05 + 0.0004
            x, z, u = _ring(width(t) * 0.92 * (1 - 0.3 * tl), 0.0006, hl)
            cl = c + up * (g - 0.0004 - 0.0012 * tl ** 2)
            pts = [cl + lat * xi + up * zi for xi, zi in zip(x, z)]
            lower.append((pts, u, t))
    tip_u = BILL_BASE + fwd * (BILL_LEN + 0.0007) + up * (culmen_drop(1.0) * 0.80)
    tip_l = BILL_BASE + fwd * (BILL_LEN * 0.95) + up * (gape(0.95) - 0.0020)
    U = _loft(upper, [tip_u])
    L = _loft(lower, [tip_l])
    return U, L
