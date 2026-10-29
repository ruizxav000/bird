"""Feet (syndactyl: toes 3 and 4 joined at the base, as in all rollers)
and the dead, weathered perch."""
import numpy as np
from util import tube

RNG = np.random.default_rng(11)

# perch tip: the twig rises toward -X, away from camera, and the feet
# wrap it at the origin (its top surface)
PERCH_R = 0.0042
PERCH_AXIS = np.array([-0.85, 0.10, 0.50])
PERCH_AXIS /= np.linalg.norm(PERCH_AXIS)
PERCH_C = np.array([0.0, 0.0, -PERCH_R])  # axis point under the grip


def _perch_frame():
    a = PERCH_AXIS
    fwd = np.array([0, 1.0, 0])
    fwd = fwd - a * (fwd @ a)
    fwd /= np.linalg.norm(fwd)
    top = np.cross(a, fwd)
    if top[2] < 0:
        top = -top
    return a, fwd, top


def toe_path(foot_x, start_ang, sweep, length, slide, r_toe, n=26):
    """Arc around the perch axis. Angle 0 = top, positive = forward/down."""
    a, fwd, top = _perch_frame()
    rad = PERCH_R + r_toe * 0.95
    ang = start_ang + np.linspace(0, 1, n) * sweep
    # don't wrap more than the length allows
    s = np.linspace(0, 1, n)
    base = PERCH_C + a * (foot_x / a[0])
    pts = (base[None]
           + (np.cos(ang)[:, None] * top + np.sin(ang)[:, None] * fwd) * rad
           + a[None] * (slide * s)[:, None])
    # resample to requested arc length
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.linspace(0, min(length, seg[-1]), n)
    return np.stack([np.interp(t, seg, pts[:, i]) for i in range(3)], axis=1)


def build_foot(sx):
    """Returns list of (verts, faces, uv, kind) with kind 'toe' or 'claw'."""
    parts = []
    fx = 0.0085 * sx
    toes = [  # start angle, sweep, length, slide along axis, root radius
        (0.20, 1.9, 0.0150, -0.0050 * sx, 0.00150),   # inner (2)
        (0.15, 2.1, 0.0175, 0.0002 * sx, 0.00165),    # middle (3)
        (0.18, 2.0, 0.0165, 0.0045 * sx, 0.00160),    # outer (4), fused to 3 at base
        (-0.30, -1.7, 0.0110, 0.0012 * sx, 0.00150),  # hind (1)
    ]
    for (a0, sw, ln, sl, r0) in toes:
        P = toe_path(fx, a0, sw, ln, sl, r0)
        n = len(P)
        s = np.linspace(0, 1, n)
        # phalangeal pads: slight bulges along the toe
        r = r0 * (1 - 0.35 * s) * (1 + 0.08 * np.sin(s * np.pi * 6) ** 2)
        v, f, uv = tube(P, r, sides=14, cap=False)
        parts.append((v, f, uv, "toe"))
        # claw: continues the toe direction, curving round the twig
        d = P[-1] - P[-2]
        d /= np.linalg.norm(d)
        a, fwd, top = _perch_frame()
        out = P[-1] - (PERCH_C + a * ((P[-1] - PERCH_C) @ a))
        out /= np.linalg.norm(out)
        m = 10
        t = np.linspace(0, 1, m)
        clen = 0.0052 if sw > 0 else 0.0062
        C = P[-1][None] + np.outer(t * clen, d) - np.outer((t ** 2) * clen * 0.45, out)
        cr = r[-1] * 0.85 * (1 - t) ** 0.9 + 0.00005
        v, f, uv = tube(C, cr, sides=10, cap=True)
        parts.append((v, f, uv, "claw"))
    # tarsus: short, mostly hidden in belly feathers
    a, fwd, top = _perch_frame()
    fb = PERCH_C + a * (fx / a[0]) + top * (PERCH_R + 0.0012)
    T = np.array([[fx * 0.9, 0.003, 0.022], 0.5 * (np.array([fx * 0.95, 0.003, 0.022]) + fb) + (0, 0.001, 0), fb])
    T = np.stack([np.interp(np.linspace(0, 2, 12), [0, 1, 2], T[:, i]) for i in range(3)], axis=1)
    v, f, uv = tube(T, np.linspace(0.0021, 0.0018, len(T)), sides=14, cap=False)
    parts.append((v, f, uv, "toe"))
    return parts


# ------------------------------------------------------------------ branch
def _spline(ctrl, n):
    ctrl = np.asarray(ctrl, float)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))]
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(seg, ctrl, bc_type="natural")
    return cs(np.linspace(0, seg[-1], n))


def _wobble(P, amp, freq, seed):
    r = np.random.default_rng(seed)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    off = np.zeros_like(P)
    for i in (0, 1, 2):
        for k in range(3):
            f = freq * (k + 1) * r.uniform(0.7, 1.3)
            off[:, i] += amp / (k + 1) * np.sin(s * f + r.uniform(0, 6.28))
    return P + off


def branch_parts():
    """Dead twig system. Returns list of (verts, faces, uv)."""
    out = []
    tip = PERCH_C + PERCH_AXIS * 0.012
    main_ctrl = [(0.10, 0.06, -0.85), (0.07, 0.03, -0.45), (0.045, 0.012, -0.16),
                 (0.026, 0.003, -0.040), PERCH_C - PERCH_AXIS * 0.012, tip]
    P = _spline(main_ctrl, 160)
    P[:-12] = _wobble(P[:-12], 0.0025, 40, 1)
    s = np.linspace(0, 1, len(P))
    R = 0.0125 * (1 - s) ** 0.9 + PERCH_R * 0.95
    R[-6:] = np.linspace(R[-7], PERCH_R * 0.7, 6)
    out.append(tube(P, R, sides=20, cap=True))

    def side(i0, ctrl, r0, r1, seed, n=80):
        base = P[i0]
        Q = _spline([base] + [base + np.asarray(c) for c in ctrl], n)
        Q = _wobble(Q, 0.0015, 60, seed)
        Q[0] = base
        t = np.linspace(0, 1, n)
        out.append(tube(Q, r0 * (1 - t) + r1 * t, sides=12, cap=True))
        return Q

    # forks like the reference: a long limb up-right with twigs, short spurs
    i1 = 62
    B1 = side(i1, [(0.0, 0.05, 0.10), (-0.01, 0.12, 0.22), (-0.03, 0.16, 0.33)], 0.0075, 0.0018, 2)
    side(i1 + 10, [(0.01, 0.03, 0.05), (0.015, 0.06, 0.09)], 0.004, 0.0012, 3, 30)
    j = 45
    base = B1[j]
    Q = _spline([base, base + (0.0, 0.04, 0.03), base + (0.01, 0.07, 0.08)], 40)
    t = np.linspace(0, 1, 40)
    out.append(tube(_wobble(Q, 0.001, 80, 4), 0.003 * (1 - t) + 0.001 * t, sides=10, cap=True))
    side(100, [(0.0, -0.03, 0.04), (0.0, -0.05, 0.07)], 0.0035, 0.0010, 5, 30)
    side(30, [(0.02, -0.06, 0.05), (0.03, -0.15, 0.11), (0.03, -0.24, 0.20)], 0.0070, 0.0016, 6)
    side(130, [(-0.01, 0.02, 0.03), (-0.012, 0.035, 0.065)], 0.0025, 0.0009, 7, 30)
    return out
