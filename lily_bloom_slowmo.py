"""
Fantasy Lily Bloom - slow motion, cinematic (black + pink)
--------------------
A stem grows, a bud forms, then a six-petal lily slowly opens with
curled petal tips, spots, stamens, glowing light and floating pollen.

Requirements:  pip install numpy matplotlib pillow

Run:
    python lily_bloom_slowmo.py            # saves lily_bloom_slowmo.gif
    python lily_bloom_slowmo.py --show     # also opens a live window
    python lily_bloom_slowmo.py --mp4      # saves lily_bloom_slowmo.mp4 (needs ffmpeg)
"""
import sys
import numpy as np
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ settings
FPS = 30
FRAMES = 480                     # 16 s of slow-motion
TILT = 0.55                      # camera looks slightly down on the flower
rng = np.random.default_rng(7)


# ------------------------------------------------------------------ helpers
def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)       # smootherstep


def lerp(a, b, t):
    return a + (b - a) * t


def project(X, Y, Z):
    """3D -> 2D with a slightly top-down camera. Returns x, y, depth."""
    x = X
    y = Z * np.cos(TILT) + Y * np.sin(TILT)
    d = Y * np.cos(TILT) - Z * np.sin(TILT)
    return x, y, d


def mix(c1, c2, t):
    c1, c2 = np.array(matplotlib.colors.to_rgb(c1)), np.array(matplotlib.colors.to_rgb(c2))
    return tuple(c1 * (1 - t) + c2 * t)


# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(6, 8), dpi=80)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-5.2, 5.2)
ax.axis("off")

# gradient night-garden background
grad = np.linspace(0, 1, 256).reshape(-1, 1)
bg = LinearSegmentedColormap.from_list(
    "bg", ["#000000", "#010003", "#07010d", "#1a0426"])
ax.imshow(grad[::-1], extent=[-4.5, 4.5, -6.5, 7.5], cmap=bg,
          aspect="auto", zorder=0)

# floating bokeh lights
N_BOKEH = 55
bk = np.column_stack([rng.uniform(-4.5, 4.5, N_BOKEH),
                      rng.uniform(-6.5, 7.0, N_BOKEH)])
bk_size = rng.uniform(30, 420, N_BOKEH)
bk_speed = rng.uniform(0.002, 0.008, N_BOKEH)
bk_phase = rng.uniform(0, 6.28, N_BOKEH)
bokeh = ax.scatter(bk[:, 0], bk[:, 1], s=bk_size, c="#ff7ad0",
                   alpha=0.10, linewidths=0, zorder=1)

# twinkling stars
N_STAR = 130
st = np.column_stack([rng.uniform(-4.5, 4.5, N_STAR), rng.uniform(-2.0, 7.2, N_STAR)])
st_ph = rng.uniform(0, 6.28, N_STAR)
st_sz = rng.uniform(2, 14, N_STAR)
stars = ax.scatter(st[:, 0], st[:, 1], s=st_sz, c="#ffd0ee", alpha=0.6,
                   linewidths=0, zorder=1)

# glowing mist on the ground
for rad, al in [(3.2, 0.05), (2.4, 0.06), (1.6, 0.08)]:
    ax.add_patch(Circle((0, -5.2), rad, color="#ff3fa4", alpha=al, lw=0, zorder=1.5))

# pollen / sparkles released at full bloom
N_SPARK = 70
sp_ang = rng.uniform(0, 2 * np.pi, N_SPARK)
sp_spd = rng.uniform(0.6, 2.4, N_SPARK)
sp_size = rng.uniform(4, 22, N_SPARK)
sparks = ax.scatter(np.zeros(N_SPARK), np.zeros(N_SPARK), s=sp_size,
                    c="#ffb8ea", alpha=0, linewidths=0, zorder=50)

dynamic = []          # artists redrawn every frame


def add(artist):
    dynamic.append(artist)
    return artist


# ------------------------------------------------------------------ stem
def stem_point(u, sway):
    """Centre line of the stem, u in [0, 1]."""
    x = 0.55 * np.sin(1.5 * u) - 0.12 + sway * u ** 2
    y = -4.9 + 5.5 * u
    return x, y


# leaves: (position on stem, side, angle, length)
LEAVES = [(0.12, -1, 0.5, 1.9), (0.22, 1, 0.45, 2.1), (0.36, -1, 0.40, 1.7),
          (0.48, 1, 0.38, 1.5), (0.60, -1, 0.35, 1.2)]


def leaf_polygon(px, py, side, ang, length, grow, sway_t):
    s = np.linspace(0, 1, 28)
    a = ang + 0.9 * s ** 1.6 + 0.04 * np.sin(sway_t + s * 3)  # droops with tip
    dx = np.cumsum(np.cos(a) * side) * length / 28 * grow
    dy = np.cumsum(np.sin(a)) * length / 28 * grow
    cx = px + np.insert(dx, 0, 0)[:-1]
    cy = py + np.insert(dy, 0, 0)[:-1] - 0.9 * grow * s ** 2 * 0.0
    w = 0.23 * grow * np.sin(np.pi * s ** 0.7) ** 0.9
    nx, ny = -np.sin(a) * side * -1, np.cos(a)
    left = np.column_stack([cx + nx * w, cy + ny * w])
    right = np.column_stack([cx - nx * w, cy - ny * w])
    return np.vstack([left, right[::-1]]), np.column_stack([cx, cy])


# ------------------------------------------------------------------ petal
PETALS = []
for k in range(6):
    outer = (k % 2 == 0)
    PETALS.append(dict(phi=np.radians(90 + k * 60 + (6 if outer else -6)),
                       outer=outer,
                       spots=rng.uniform(0, 1, (11, 2))))


def petal_geometry(p, b, scale, rot=0.0, n=34):
    outer = p["outer"]
    phi = p["phi"] + rot
    theta0 = lerp(0.10 if outer else 0.05, 0.62 if outer else 0.50, b)
    curl = lerp(0.0, 1.9 if outer else 1.6, b)
    length = lerp(0.85, 1.0, b) * (2.7 if outer else 2.5) * scale
    width = lerp(0.55, 1.0, b) * (0.52 if outer else 0.62) * scale
    s = np.linspace(0, 1, n)
    theta = theta0 + curl * s ** 2.2
    step = length / n
    r = np.insert(np.cumsum(np.sin(theta)) * step, 0, 0)[:-1] + 0.03 * scale
    z = np.insert(np.cumsum(np.cos(theta)) * step, 0, 0)[:-1]
    X, Y, Z = r * np.cos(phi), r * np.sin(phi), z
    w = width * np.sin(np.pi * s ** 0.72) ** 0.85 + 0.05 * width * (1 - s)
    w *= 1 + 0.10 * b * np.sin(13 * s + phi * 3) * s       # ruffled edge
    sx, sy = -np.sin(phi), np.cos(phi)                    # sideways vector
    return s, (X, Y, Z), (sx, sy), w


def to_poly(c, side, w, base):
    X, Y, Z = c
    sx, sy = side
    L = project(X + sx * w, Y + sy * w, Z)
    R = project(X - sx * w, Y - sy * w, Z)
    pts = np.vstack([np.column_stack(L[:2]), np.column_stack(R[:2])[::-1]])
    return pts + base, np.concatenate([L[2], R[2]]).mean()


# ------------------------------------------------------------------ frame
def draw(frame):
    for a in dynamic:
        a.remove()
    dynamic.clear()

    t = frame / (FRAMES - 1)
    g = ease(t / 0.20)                       # stem growth
    b = ease((t - 0.20) / 0.62)              # bloom amount (slow)
    b = float(np.clip(b + 0.015 * np.sin(frame * 0.05) * ease((t - 0.8) / 0.1), 0, 1))
    scale = lerp(0.30, 1.05, ease((t - 0.14) / 0.70))
    sway = 0.07 * np.sin(2 * np.pi * t * 1.3) * ease((t - 0.25) / 0.3)
    rot = lerp(-0.5, 0.75, ease(t))          # slow turntable rotation

    # cinematic push-in
    z = ease((t - 0.12) / 0.75)
    hw = lerp(3.6, 3.15, z)
    hh = hw * 10.4 / 7.2
    cy = lerp(0.0, 1.3, z)
    ax.set_xlim(-hw, hw)
    ax.set_ylim(cy - hh, cy + hh)

    # --- light rays fanning out from the flower
    top = np.array(stem_point(max(g, 1e-3), sway))
    ctr = top + np.array([0, 1.2 * scale])
    for i, a0 in enumerate(np.linspace(35, 145, 9)):
        a0 = np.radians(a0 + 6 * np.sin(frame * 0.02 + i))
        wd = np.radians(2.2 + 1.2 * np.sin(i * 1.7) ** 2)
        Lr = 9.0
        tri = [ctr, ctr + Lr * np.array([np.cos(a0 - wd), np.sin(a0 - wd)]),
               ctr + Lr * np.array([np.cos(a0 + wd), np.sin(a0 + wd)])]
        add(ax.add_patch(Polygon(tri, closed=True, fc="#ff6ac6", ec="none",
                                 alpha=0.05 * b, zorder=1.8)))

    # --- soft glow behind the flower
    top = np.array(stem_point(max(g, 1e-3), sway))
    for rad, al in [(4.2, 0.03), (3.4, 0.04), (2.6, 0.05), (1.8, 0.06), (1.1, 0.08)]:
        add(ax.add_patch(Circle(top + [0, 0.9 * scale], rad * (0.4 + 0.6 * b + 0.3 * scale),
                                color="#ff4fb0", alpha=al * 1.1 * (0.4 + b), zorder=2,
                                lw=0)))

    # --- stem
    u = np.linspace(0, g, 40)
    sx_, sy_ = stem_point(u, sway)
    for lw, col in [(11, "#052a22"), (7, "#0f7a5c"), (2.5, "#5ff5c8")]:
        add(ax.plot(sx_, sy_, color=col, lw=lw, solid_capstyle="round", zorder=3)[0])

    # --- leaves
    for i, (pos, side, ang, ln) in enumerate(LEAVES):
        lg = ease((g - pos) / 0.25) * (1 if g >= pos else 0)
        if lg <= 0.01:
            continue
        px, py = stem_point(pos, sway)
        poly, mid = leaf_polygon(px, py, side, ang, ln, lg, t * 6 + i)
        add(ax.add_patch(Polygon(poly, closed=True, fc="#0c5f4a", ec="#2ee6b0",
                                 lw=1, zorder=3.5)))
        add(ax.plot(mid[:, 0], mid[:, 1], color="#8dfbd8", lw=1.0, alpha=0.8,
                    zorder=3.6)[0])

    # --- flower
    base = np.array(stem_point(g, sway))
    items = []

    # petals
    for p in PETALS:
        s, c, side, w = petal_geometry(p, b, scale, rot)
        poly, depth = to_poly(c, side, w, base)
        items.append((depth, "petal", p, s, c, side, w, poly))

    # stamens
    for k in range(6):
        phi = np.radians(k * 60 + 30) + rot
        th = lerp(0.12, 0.62, b) + 0.05 * np.sin(k)
        ln = (1.9 + 0.2 * np.cos(k * 2)) * scale * lerp(0.75, 1.0, b)
        ss = np.linspace(0, 1, 16)
        thk = th + 0.5 * b * ss ** 2
        r = np.insert(np.cumsum(np.sin(thk)) * ln / 16, 0, 0)[:-1]
        z = np.insert(np.cumsum(np.cos(thk)) * ln / 16, 0, 0)[:-1]
        x, y, d = project(r * np.cos(phi), r * np.sin(phi), z)
        items.append((d[-1], "stamen", (x + base[0], y + base[1]), thk[-1], phi))

    # pistil
    thp = lerp(0.04, 0.30, b)
    lp = 2.25 * scale * lerp(0.8, 1.0, b)
    ss = np.linspace(0, 1, 16)
    r = np.insert(np.cumsum(np.sin(thp + 0.4 * b * ss ** 2)) * lp / 16, 0, 0)[:-1]
    z = np.insert(np.cumsum(np.cos(thp + 0.4 * b * ss ** 2)) * lp / 16, 0, 0)[:-1]
    x, y, d = project(r * np.cos(np.radians(270) + rot), r * np.sin(np.radians(270) + rot), z)
    items.append((d[-1] - 0.3, "pistil", (x + base[0], y + base[1])))

    items.sort(key=lambda it: -it[0])      # far to near
    zc = 10
    for it in items:
        zc += 1
        kind = it[1]
        if kind == "petal":
            _, _, p, s, c, side, w, poly = it
            outer = p["outer"]
            dark = 0.0 if outer else 0.06
            fc = mix("#ffc2e0", "#ff7fbd", 0.35 + dark)
            add(ax.add_patch(Polygon(poly, closed=True, fc="none", ec="#ff4fb0",
                                     lw=9, alpha=0.12 + 0.10 * b, zorder=zc - 0.02)))
            add(ax.add_patch(Polygon(poly, closed=True, fc="none", ec="#ff7ac8",
                                     lw=4, alpha=0.25 + 0.15 * b, zorder=zc - 0.01)))
            X_, Y_, Z_ = c
            Lx, Ly, _ = project(X_ + side[0] * w, Y_ + side[1] * w, Z_)
            Rx, Ry, _ = project(X_ - side[0] * w, Y_ - side[1] * w, Z_)
            nseg = 10
            edges = np.linspace(0, len(s) - 1, nseg + 1).astype(int)
            for i in range(nseg):
                a_, b_ = edges[i], edges[i + 1]
                quad = np.vstack([np.column_stack([Lx[a_:b_ + 1], Ly[a_:b_ + 1]]),
                                  np.column_stack([Rx[a_:b_ + 1], Ry[a_:b_ + 1]])[::-1]]) + base
                sm = ((a_ + b_) / 2) / (len(s) - 1)
                col = mix("#ff2f94", "#ffd6ec", min(1, sm ** 0.7 * 0.92 + (0.0 if outer else 0.06)))
                add(ax.add_patch(Polygon(quad, closed=True, fc=col, ec=col,
                                         lw=0.7, zorder=zc)))
            add(ax.add_patch(Polygon(poly, closed=True, fc="none", ec="#ffc2e6",
                                     lw=1.3, zorder=zc + 0.005)))
            # pink centre stripe (broad, fading toward the tip)
            cut = int(len(s) * 0.82)
            pol2, _ = to_poly(tuple(a[:cut] for a in c), side, w[:cut] * 0.42,
                              np.array(base))
            add(ax.add_patch(Polygon(pol2, closed=True, fc="#d4147f", ec="none",
                                     alpha=0.38 * (0.3 + b), zorder=zc + 0.01)))
            # yellow-green throat near the base
            cut2 = int(len(s) * 0.34)
            pol3, _ = to_poly(tuple(a[:cut2] for a in c), side, w[:cut2] * 0.55,
                              np.array(base))
            add(ax.add_patch(Polygon(pol3, closed=True, fc="#ffe58a", ec="none",
                                     alpha=0.65, zorder=zc + 0.02)))
            # midrib
            mx, my, _ = project(c[0], c[1], c[2])
            add(ax.plot(mx + base[0], my + base[1], color="#c4126d", lw=1.0,
                        alpha=0.55, zorder=zc + 0.03)[0])
            # freckles
            if b > 0.25:
                idx = (p["spots"][:, 0] * 0.45 + 0.22) * (len(s) - 1)
                idx = idx.astype(int)
                off = (p["spots"][:, 1] - 0.5) * 1.1 * w[idx]
                X = c[0][idx] + side[0] * off
                Y = c[1][idx] + side[1] * off
                Z = c[2][idx]
                spx, spy, _ = project(X, Y, Z)
                add(ax.scatter(spx + base[0], spy + base[1], s=5 + 9 * scale,
                               c="#7a0a47", alpha=0.75 * ease((b - 0.25) / 0.4),
                               linewidths=0, zorder=zc + 0.04))
        elif kind == "stamen":
            _, _, (x, y), th, phi = it
            add(ax.plot(x, y, color="#ffe6f4", lw=1.8, zorder=zc,
                        solid_capstyle="round")[0])
            add(ax.plot(x[-3:], y[-3:], color="#6b0f3f", lw=5.5, zorder=zc + 0.01,
                        solid_capstyle="round")[0])
            add(ax.plot(x[-2:], y[-2:], color="#ff5db1", lw=3, zorder=zc + 0.02,
                        solid_capstyle="round")[0])
        else:
            _, _, (x, y) = it
            add(ax.plot(x, y, color="#ffd0ee", lw=2.6, zorder=zc,
                        solid_capstyle="round")[0])
            add(ax.scatter(x[-1], y[-1], s=38 * (0.4 + scale), c="#ff9ad8",
                           edgecolors="#ffffff", linewidths=1, zorder=zc + 0.1))

    # --- shimmer rings at full bloom
    rt = (t - 0.78) / 0.22
    if rt > 0:
        for k in range(3):
            rr = rt - 0.14 * k
            if 0 < rr < 1:
                add(ax.add_patch(Circle(base + [0, 1.3 * scale], 0.8 + 3.6 * rr,
                                        fc="none", ec="#ff8ad2", lw=2.2 * (1 - rr) + 0.3,
                                        alpha=0.55 * (1 - rr), zorder=2.2)))

    # --- bokeh drift
    bk[:, 1] += bk_speed
    bk[bk[:, 1] > 7.2, 1] = -6.5
    off = bk.copy()
    off[:, 0] += 0.12 * np.sin(frame * 0.015 + bk_phase)
    bokeh.set_offsets(off)
    bokeh.set_alpha(0.08 + 0.10 * b)
    stars.set_alpha(0.35 + 0.35 * np.abs(np.sin(frame * 0.08 + st_ph)).mean())
    stars.set_sizes(st_sz * (0.5 + 0.8 * np.abs(np.sin(frame * 0.1 + st_ph))))

    # --- pollen burst once the flower is open
    pt = (t - 0.78) / 0.22
    if pt > 0:
        d = pt * sp_spd * 1.6
        cx, cy = base[0], base[1] + 1.2 * scale
        px = cx + np.cos(sp_ang) * d + 0.25 * np.sin(frame * 0.07 + sp_ang * 5)
        py = cy + np.sin(sp_ang) * d * 0.8 - 0.35 * pt ** 2 * 2
        sparks.set_offsets(np.column_stack([px, py]))
        sparks.set_alpha(float(np.clip(0.9 * np.sin(np.pi * min(pt, 1)) , 0, 0.9)))
    else:
        sparks.set_alpha(0)

    return dynamic


# ------------------------------------------------------------------ run
anim = FuncAnimation(fig, draw, frames=FRAMES, interval=1000 / FPS, blit=False)

if "--mp4" in sys.argv:
    anim.save("lily_bloom_slowmo.mp4", fps=FPS, dpi=100)
    print("Saved lily_bloom_slowmo.mp4")
else:
    anim.save("lily_bloom_slowmo.gif", writer=PillowWriter(fps=FPS))
    print("Saved lily_bloom_slowmo.gif")

if "--show" in sys.argv:
    plt.show()
