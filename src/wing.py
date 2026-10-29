"""Folded wings and tail, built feather-by-feather.

Folded-wing colour layout (field-guide plate + reference frames):
  scapulars, tertials        cinnamon-brown, covering the top/rear edge
  lesser coverts             deep ultramarine-violet at the bend of the wing
                             and as a band under the scapulars
  median / greater coverts   light turquoise-azure, scalloped
  secondaries                turquoise base, deep blue distally
  primaries                  azure base, deep violet-blue, blackish tips,
                             projecting beyond the tertials
Tail: rectrices pale azure with darker central pair; outer pair elongated
into streamers whose projecting part is black.
"""
import numpy as np
from scipy.spatial import cKDTree

import sdf
from plumage import Drape, RNG, contour_outline, ss
from util import srgb_np

# wing frame in the side profile (bird Y, Z)
W_O = np.array([0.004, 0.091])
W_TIP = np.array([-0.114, -0.008])
W_A = (W_TIP - W_O) / np.linalg.norm(W_TIP - W_O)
W_D = np.array([W_A[1], -W_A[0]]) * -1  # dorsal: up/back
if W_D[1] < 0:
    W_D = -W_D
W_LEN = np.linalg.norm(W_TIP - W_O)

COL = {k: np.array(v) for k, v in dict(
    brown=(0.60, 0.39, 0.20), brown_edge=(0.70, 0.50, 0.30),
    ultra=(0.14, 0.17, 0.62), violet=(0.20, 0.15, 0.55), azure=(0.30, 0.64, 0.82),
    turq=(0.26, 0.66, 0.73), turq_edge=(0.50, 0.80, 0.82), dark=(0.04, 0.045, 0.09),
    tail_c=(0.16, 0.28, 0.34), tail_az=(0.30, 0.54, 0.72), black=(0.012, 0.012, 0.018),
).items()}


def flight_outline(u):
    w = np.minimum(1, (u + 0.02) / 0.10) ** 0.5
    tip = np.sqrt(np.clip(1 - ((u - 0.78) / 0.24) ** 2, 0, 1))
    return np.maximum(np.where(u > 0.78, tip, w), 0.05)


def covert_outline(u):
    w = 0.55 + 0.45 * np.sin(0.5 * np.pi * np.minimum(u / 0.5, 1))
    tip = np.sqrt(np.clip(1 - ((u - 0.55) / 0.47) ** 2, 0, 1))
    return np.maximum(np.where(u > 0.55, tip, w), 0.06)


class SideSurface:
    """Outer surface of the body plumage, queried from the side."""

    def __init__(self, offset):
        v, f = sdf.polygonize(lambda P: sdf.body_sdf(P) - offset,
                              (-0.06, -0.12, -0.04), (0.06, 0.09, 0.14), 0.0015)
        t = v[f]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        vn = np.zeros_like(v)
        for i in range(3):
            np.add.at(vn, f[:, i], n)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        self.V, self.N = v, vn
        self.drape = Drape(v, vn)
        self.kd_yz = cKDTree(v[:, 1:])

    def side_point(self, yz, sx):
        idx = self.kd_yz.query_ball_point(yz, 0.003)
        cand = [i for i in idx if self.V[i, 0] * sx > 0]
        if not cand:
            d, i = self.kd_yz.query(yz)
            p = self.V[i].copy()
            p[0] = sx * max(abs(p[0]), 0.006)
            return p, np.array([sx, 0, 0.0])
        i = max(cand, key=lambda j: self.V[j, 0] * sx)
        return self.V[i].copy(), self.N[i].copy()


R_SHIFT = -0.012  # push the folded wing down the flank


def profile(s, r):
    return W_O + W_A * s + W_D * (r + R_SHIFT)


def cards(specs, surf, sx, nu=16, nv=7):
    """specs: list of dict(root=(s,r), tip=(s,r), width, layer, colfn, shaft,
    outline, curl, xmin).  Returns verts, faces, attrs."""
    u = np.linspace(0, 1, nu)
    v = np.linspace(-1, 1, nv)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    U = U.ravel(); Vv = Vv.ravel()
    verts, faces, att = [], [], {"uvn": [], "uvm": [], "fcol": [], "scol": [], "rnd": []}
    k = nu * nv
    qi = np.arange(nu - 1)[:, None] * nv + np.arange(nv - 1)[None, :]
    q = np.stack([qi, qi + nv, qi + nv + 1, qi + 1], axis=-1).reshape(-1, 4)
    for n, sp in enumerate(specs):
        r0 = profile(*sp["root"])
        r1 = profile(*sp["tip"])
        p0, n0 = surf.side_point(r0, sx)
        L = np.linalg.norm(r1 - r0)
        d2 = (r1 - r0) / L
        t = np.array([0, d2[0], d2[1]])
        t = t - n0 * (t @ n0)
        t /= np.linalg.norm(t)
        b = np.cross(n0, t) * sx
        wo = sp.get("outline", flight_outline)(U)
        Wd = sp["width"]
        X = p0 + np.outer(U * L, t) + np.outer(Vv * wo * 0.5 * Wd, b)
        S, Nn = surf.drape.project(X)
        lift = sp["layer"] * 0.0009 + 0.0006 + sp.get("lift", 0.0) * U
        h = lift - sp.get("curl", 0.10) * (Vv ** 2) * wo * 0.5 * Wd
        P = S + Nn * h[:, None]
        xmin = sp.get("xmin", 0.0045) + sp["layer"] * 0.0009
        P[:, 0] = sx * np.maximum(P[:, 0] * sx, xmin)
        verts.append(P)
        faces.append(q + n * k)
        att["uvn"].append(np.c_[U, Vv * sx])
        att["uvm"].append(np.c_[U * L * 1000, Vv * sx * wo * Wd * 500])
        att["fcol"].append(srgb_np(sp["colfn"](U, Vv * sx)))
        att["scol"].append(np.repeat(srgb_np(sp["shaft"])[None], k, axis=0))
        att["rnd"].append(np.full(k, RNG.random()))
    return (np.concatenate(verts), np.concatenate(faces),
            {kk: np.concatenate(vv) for kk, vv in att.items()})


def _grad(stops):
    """stops: list of (u, colour). Returns colfn(U, V)."""
    us = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops])

    def f(U, V):
        return np.stack([np.interp(U, us, cs[:, i]) for i in range(3)], axis=-1)
    return f


def _edged(fn, edge_col, amt=0.5, start=0.55):
    """Paler fringe near the vane margin (scalloped covert look)."""
    def f(U, V):
        c = fn(U, V)
        e = ss(0.55, 0.95, np.abs(V)) * ss(start - 0.2, start + 0.25, U) * amt
        return c * (1 - e[:, None]) + edge_col * e[:, None]
    return f


def wing_specs():
    S = []
    jit = lambda a: a + RNG.normal(0, 0.0006)
    # primaries: innermost layer, tips form the wing point
    prim = _grad([(0, COL["azure"]), (0.30, COL["azure"]), (0.48, COL["violet"]),
                  (0.86, COL["violet"] * 0.8), (0.97, COL["dark"])])
    for i in range(9):
        f = i / 8
        S.append(dict(root=(0.030 + 0.020 * f, -0.010 + 0.004 * f),
                      tip=(W_LEN + 0.004 - 0.030 * f ** 1.3, -0.004 + 0.010 * f),
                      width=0.017, layer=0 + 0.15 * (8 - i) / 8, colfn=prim,
                      shaft=(0.06, 0.06, 0.10), curl=0.12))
    # secondaries
    sec = _grad([(0, COL["turq"]), (0.40, COL["azure"]), (0.62, COL["ultra"]),
                 (0.92, COL["violet"] * 0.75), (1.0, COL["dark"])])
    for i in range(8):
        f = i / 7
        S.append(dict(root=(0.008 + 0.040 * f, -0.013 + 0.022 * f),
                      tip=(0.098 + 0.010 * f, -0.014 + 0.020 * f),
                      width=0.019, layer=1.2 + 0.1 * f, colfn=sec,
                      shaft=(0.08, 0.10, 0.18), curl=0.14))
    # tertials: brown, over the secondaries at the top/rear
    ter = _edged(_grad([(0, COL["brown"]), (1, COL["brown"] * 0.92)]), COL["brown_edge"], 0.35)
    for i in range(3):
        f = i / 2
        S.append(dict(root=(0.028 + 0.010 * f, 0.010 + 0.004 * f),
                      tip=(0.104 - 0.010 * f, 0.008 + 0.004 * f),
                      width=0.021, layer=2.4 + 0.2 * i, colfn=ter,
                      shaft=(0.50, 0.33, 0.18), curl=0.12))
    # greater coverts: turquoise, scalloped paler fringes
    gc = _edged(_grad([(0, COL["turq"] * 0.85), (1, COL["turq"])]), COL["turq_edge"], 0.45)
    for i in range(10):
        f = i / 9
        rs = -0.010 + 0.017 * f
        S.append(dict(root=(jit(-0.002 + 0.050 * f), rs),
                      tip=(jit(0.030 + 0.050 * f), rs - 0.006),
                      width=0.013, layer=3.3 + 0.08 * i, colfn=gc, outline=covert_outline,
                      shaft=(0.20, 0.45, 0.50), curl=0.10))
    # median coverts
    for i in range(9):
        f = i / 8
        rs = -0.008 + 0.018 * f
        S.append(dict(root=(jit(-0.008 + 0.040 * f), rs + 0.002),
                      tip=(jit(0.014 + 0.040 * f), rs - 0.003),
                      width=0.011, layer=4.2 + 0.06 * i, colfn=gc, outline=covert_outline,
                      shaft=(0.20, 0.45, 0.50), curl=0.10))
    # lesser coverts: ultramarine at the bend of the wing and a band under
    # the scapulars
    lc = _edged(_grad([(0, COL["ultra"] * 0.85), (1, COL["ultra"])]), COL["violet"] * 1.1, 0.08)
    for i in range(80):
        if i < 36:
            s0 = RNG.uniform(-0.012, 0.012); r0 = RNG.uniform(-0.012, 0.016)
        else:
            s0 = RNG.uniform(0.0, 0.070); r0 = RNG.uniform(0.007, 0.014)
        S.append(dict(root=(s0, r0), tip=(s0 + RNG.uniform(0.012, 0.016), r0 - 0.002),
                      width=0.010, layer=5.0 + 0.4 * RNG.random(), colfn=lc,
                      outline=covert_outline, shaft=(0.10, 0.12, 0.40), curl=0.10))
    # scapulars: brown, tucked over the dorsal edge
    sc = _edged(_grad([(0, COL["brown"] * 0.9), (1, COL["brown"])]), COL["brown_edge"], 0.3)
    for i in range(9):
        f = i / 8
        S.append(dict(root=(jit(-0.010 + 0.060 * f), 0.020 + RNG.uniform(-0.002, 0.003)),
                      tip=(jit(0.026 + 0.060 * f), 0.015),
                      width=0.019, layer=6.0 + 0.1 * i, colfn=sc, outline=covert_outline,
                      shaft=(0.50, 0.33, 0.18), curl=0.10))
    return S


# ------------------------------------------------------------------- tail
TAIL_BASE = np.array([0.0, -0.058, 0.014])
TAIL_ANGLE = 47.0  # degrees below horizontal, pointing back


def tail_frame():
    a = np.radians(TAIL_ANGLE)
    d = np.array([0, -np.cos(a), -np.sin(a)])
    up = np.array([0, -np.sin(a), np.cos(a)])
    return d, up


def streamer_outline(u):
    base = flight_outline(np.minimum(u / 0.6, 1.0))
    narrow = 0.26 * (1 - ss(0.80, 1.0, u)) + 0.04
    return np.where(u < 0.50, np.maximum(base, 0.05),
                    np.maximum(narrow + (1 - narrow) * ss(0.58, 0.50, u), 0.04))


def rigid_cards(specs, nu=24, nv=7):
    """Cards in a fixed frame: dict(root, dir, up, L, width, outline, colfn,
    shaft, droop, curl, splay(3,) lateral drift at tip)."""
    u = np.linspace(0, 1, nu)
    v = np.linspace(-1, 1, nv)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    U = U.ravel(); Vv = Vv.ravel()
    k = nu * nv
    qi = np.arange(nu - 1)[:, None] * nv + np.arange(nv - 1)[None, :]
    q = np.stack([qi, qi + nv, qi + nv + 1, qi + 1], axis=-1).reshape(-1, 4)
    verts, faces, att = [], [], {"uvn": [], "uvm": [], "fcol": [], "scol": [], "rnd": []}
    for n, sp in enumerate(specs):
        d, up = np.asarray(sp["dir"]), np.asarray(sp["up"])
        lat = np.cross(d, up)
        roll = np.radians(sp.get("roll", 0.0))
        up, lat = up * np.cos(roll) + lat * np.sin(roll), lat * np.cos(roll) - up * np.sin(roll)
        L, Wd = sp["L"], sp["width"]
        wo = sp.get("outline", flight_outline)(U)
        P = (sp["root"] + np.outer(U * L, d) + np.outer(Vv * wo * 0.5 * Wd, lat)
             + np.outer(-sp.get("droop", 0.0) * U ** 2 - sp.get("curl", 0.12) * Vv ** 2 * wo * 0.5 * Wd, up)
             + np.outer(U ** 1.5, sp.get("splay", np.zeros(3))))
        verts.append(P)
        faces.append(q + n * k)
        att["uvn"].append(np.c_[U, Vv])
        att["uvm"].append(np.c_[U * L * 1000, Vv * wo * Wd * 500])
        att["fcol"].append(srgb_np(sp["colfn"](U, Vv)))
        att["scol"].append(np.repeat(srgb_np(sp["shaft"])[None], k, axis=0))
        att["rnd"].append(np.full(k, RNG.random()))
    return (np.concatenate(verts), np.concatenate(faces),
            {kk: np.concatenate(vv) for kk, vv in att.items()})


def tail_specs():
    d, up = tail_frame()
    X = np.array([1.0, 0, 0])
    S = []
    central = _grad([(0, COL["tail_c"] * 0.8), (0.7, COL["tail_c"]), (1, COL["tail_c"] * 0.6)])
    side = _grad([(0, COL["tail_az"] * 0.8), (0.75, COL["tail_az"]), (0.92, COL["violet"]), (1, COL["dark"])])
    stream = _grad([(0, COL["tail_az"] * 0.8), (0.42, COL["tail_az"]), (0.52, COL["violet"] * 0.5),
                    (0.58, COL["black"]), (1, COL["black"])])
    for sx in (1, -1):
        for i in range(6):
            outer = i == 5
            S.append(dict(
                root=TAIL_BASE + X * sx * (0.0012 + 0.0014 * i) - up * (0.0009 * i),
                dir=d, up=up, L=0.215 if outer else 0.122 - 0.002 * i,
                width=0.021 if not outer else 0.019,
                outline=streamer_outline if outer else flight_outline,
                colfn=central if i == 0 else (stream if outer else side),
                shaft=(0.05, 0.06, 0.08), droop=0.010 if outer else 0.004,
                curl=0.10, roll=-sx * (6 + 7 * i),
                splay=X * sx * (0.0035 + 0.0020 * i + (0.016 if outer else 0)) - up * (0.0008 * i + (0.010 * sx + 0.002 if outer else 0))))
    # upper-tail coverts (violet-blue rump) and under-tail coverts (pale turquoise)
    utc = _edged(_grad([(0, COL["violet"]), (1, COL["ultra"] * 1.1)]), COL["ultra"] * 1.3, 0.3)
    ltc = _grad([(0, COL["turq"]), (1, COL["turq_edge"])])
    for j in range(7):
        sx = (j - 3) / 3
        S.append(dict(root=TAIL_BASE + d * -0.016 + up * 0.0045 + X * sx * 0.008,
                      dir=d, up=up, L=0.040, width=0.016, outline=covert_outline,
                      colfn=utc, shaft=(0.12, 0.10, 0.35), droop=0.002, curl=0.15,
                      splay=X * sx * 0.003))
        S.append(dict(root=TAIL_BASE + d * -0.018 - up * 0.008 + X * sx * 0.009,
                      dir=d, up=up, L=0.045, width=0.018, outline=covert_outline,
                      colfn=ltc, shaft=(0.4, 0.7, 0.7), droop=-0.003, curl=-0.12,
                      splay=X * sx * 0.004))
    return S


def overlap_roots(surf, sx, n_front=0, n_flank=0):
    """Breast feathers spilling over the bend of the wing and flank
    feathers tucked over its lower edge."""
    pts = []
    for _ in range(n_front):
        pts.append(profile(RNG.uniform(-0.026, -0.012), RNG.uniform(-0.016, 0.020)))
    for _ in range(n_flank):
        pts.append(profile(RNG.uniform(0.0, 0.085), RNG.uniform(-0.030, -0.022)))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    roots, normals = zip(*[surf.side_point(p, sx) for p in pts])
    return np.array(roots), np.array(normals)
