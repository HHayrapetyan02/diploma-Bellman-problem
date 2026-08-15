import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.geometry import from_pq

OUT = "figures"
Q_SS = -2.0 * np.sqrt(6.0) / 9.0          # инварианты самоподобной орбиты
P_SS = 2.0 / 3.0
EXACT_SS = (np.sqrt(6.0) ** 5) / 540.0    # точная стоимость на ней при |y| = 1

STYLE = {
    "square":  ("квадрат (описанный)",        "#1f77b4", "-",  "low"),
    "cuts":    ("аффинные отсечения",         "#17becf", "--", "low"),
    "rect":    ("прямоугольник (вписанный)",  "#d62728", "-",  "up"),
    "timeopt": ("быстродействие",             "#ff7f0e", ":",  "up"),
    "polimp":  ("policy improvement",         "#2ca02c", "-.", "up"),
    "jensen":  ("выпуклая оболочка (Йенсен)", "#9467bd", "--", "up"),
}


# ------------------------------------------------------- доступные методы
def build_methods(light=False):
    """{имя: f(x, y) -> J}; отсутствующие модули пропускаются."""
    m = {}

    from bounds.lower.square import LowerBoundBellmanFunction
    from bounds.upper.rectangle import UpperBoundBellmanFunction
    sq, rc = LowerBoundBellmanFunction(), UpperBoundBellmanFunction()
    m["square"] = lambda x, y: -sq.lowerBoundBellman2D(x, y)
    m["rect"] = lambda x, y: -rc.upperBoundBellman2DRectangle(x, y, n_points=24)

    try:
        from bounds.upper.time_optimal import TimeOptimalBound
        to = TimeOptimalBound()
        def _to(x, y):
            v = to.upper_bound_time_optimal(x, y)
            return -v if np.isfinite(v) else None
        m["timeopt"] = _to
    except Exception:
        pass

    try:
        from bounds.upper.policy_improvement import PolicyImprovementBound
        pi = PolicyImprovementBound(
            n_controls=8 if light else 12,
            h_factors=(0.1, 0.3) if light else (0.1, 0.2, 0.4),
            rect_points=20 if light else 24)
        m["polimp"] = lambda x, y: -pi.upper_bound_policy_improvement(x, y)
    except Exception:
        pass

    try:
        from bounds.convex.affine import AffineCutLowerBound
        cu = AffineCutLowerBound(n_restarts=6 if light else 10)
        m["cuts"] = lambda x, y: cu.cost(np.concatenate([x, y]))
    except Exception:
        pass

    try:
        from bounds.convex.jensen import JensenUpperBound
        je = JensenUpperBound(n_state=7 if light else 9, n_dir=16 if light else 24,
                              n_rot=24 if light else 32, n_lambda=5)
        m["jensen"] = lambda x, y: je.cost(np.concatenate([x, y]))
    except Exception:
        pass

    return m


def evaluate(states, cache, methods):
    """states: [(meta, x, y)]; результат кэшируется по индексу точки."""
    rows = json.load(open(cache)) if os.path.exists(cache) else []
    for k in range(len(rows), len(states)):
        meta, x, y = states[k]
        for name, fn in methods.items():
            try:
                meta[name] = fn(x, y)
            except Exception:
                meta[name] = None
        rows.append(meta)
        json.dump(rows, open(cache, "w"))
        print("  %d/%d" % (len(rows), len(states)), end="\r", flush=True)
    return rows


def column(rows, key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in rows],
                    dtype=float)


def sides(rows):
    have = [k for k in STYLE if k in rows[0] and np.isfinite(column(rows, k)).any()]
    return ([k for k in have if STYLE[k][3] == "low"],
            [k for k in have if STYLE[k][3] == "up"])


# ============================================ рис. 1: срез q = q_самоподоб.
def figure_slice(n_points, methods):
    ps = np.sort(np.unique(np.concatenate([
        np.linspace(Q_SS ** 2 + 0.02, 1.4, n_points), [P_SS]])))
    states = [(dict(p=float(p)), *from_pq(float(p), Q_SS, ny=1.0)) for p in ps]
    rows = evaluate(states, f"{OUT}/cache_slice.json", methods)

    p = column(rows, "p")
    data = {k: column(rows, k) for k in STYLE if k in rows[0]}
    lows, ups = sides(rows)
    best_low = np.nanmax(np.vstack([data[k] for k in lows]), axis=0)
    best_up = np.nanmin(np.vstack([data[k] for k in ups]), axis=0)

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    a.fill_between(p, data["square"], data["rect"], color="0.85",
                   label="коридор до наших методов")
    a.fill_between(p, best_low, best_up, color="#ffe4a8", label="коридор сейчас")
    for k in data:
        lab, col, ls, _ = STYLE[k]
        a.plot(p, data[k], ls, color=col, lw=1.8, label=lab)
    a.plot([P_SS], [EXACT_SS], "k*", ms=16, zorder=5, label="точное значение")
    a.annotate("самоподобная орбита\n$J=\\lambda^5/540$", (P_SS, EXACT_SS),
               textcoords="offset points", xytext=(14, -36), fontsize=9)
    a.set_xlabel("$p=\\|x\\|^2/\\|y\\|^4$   (срез $q=-2\\sqrt{6}/9$, $\\|y\\|=1$)")
    a.set_ylabel("стоимость $J=-\\omega$")
    a.set_title("Все оценки на срезе через самоподобную орбиту")
    a.legend(fontsize=8.5, loc="upper left")
    a.grid(alpha=0.25)

    # тот же срез, но в долях старого коридора: 0 = квадрат, 1 = прямоугольник
    width = data["rect"] - data["square"]
    b.axhspan(0, 1, color="0.9")
    b.axhline(0, color=STYLE["square"][1], lw=2)
    b.axhline(1, color=STYLE["rect"][1], lw=2)
    for k in data:
        if k in ("square", "rect"):
            continue
        lab, col, ls, _ = STYLE[k]
        b.plot(p, (data[k] - data["square"]) / width, ls, color=col, lw=1.9,
               label=lab)
    i_ss = int(np.argmin(abs(p - P_SS)))
    b.plot([P_SS], [(EXACT_SS - data["square"][i_ss]) / width[i_ss]], "k*",
           ms=16, zorder=5, label="точное значение")
    b.set_ylim(-0.6, 1.7)
    b.set_xlabel("$p$")
    b.set_ylabel("доля старого коридора")
    b.set_title("То же в нормировке: 0 — квадрат, 1 — прямоугольник")
    b.legend(fontsize=8.5, loc="lower right", ncol=2)
    b.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_slice.png", dpi=160)
    plt.close(fig)
    return rows


# ================================== рис. 2: зазоры в самоподобной точке
def figure_gaps(rows):
    row = min(rows, key=lambda r: abs(r["p"] - P_SS))
    items = [(k, 100 * abs(row[k] - EXACT_SS) / EXACT_SS)
             for k in STYLE if row.get(k) is not None]
    items.sort(key=lambda t: -t[1])

    fig, ax = plt.subplots(figsize=(9.5, 0.75 * len(items) + 2.0))
    bars = ax.barh(range(len(items)), [g for _, g in items],
                   color=[STYLE[k][1] for k, _ in items], alpha=0.85)
    for bar, (k, _) in zip(bars, items):
        if STYLE[k][3] == "low":
            bar.set_hatch("//")
    for i, (_, g) in enumerate(items):
        ax.text(g + 0.25, i, "%.2f %%" % g, va="center", fontsize=10)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([STYLE[k][0] for k, _ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, max(g for _, g in items) * 1.2)
    ax.set_xlabel("отклонение от точного значения, %")
    ax.set_title("Самоподобная точка — единственная внутренняя точка с точным "
                 "ответом\n(штриховка — оценки снизу, заливка — сверху)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_gaps.png", dpi=160)
    plt.close(fig)
    return items


# ==================================== рис. 3 (по флагу): карта по (r, phi)
def figure_grid(n_r, n_f, methods):
    rs, fs = np.linspace(0.0, 0.95, n_r), np.linspace(-np.pi, np.pi, n_f)
    states = []
    for i, r in enumerate(rs):
        for j, f in enumerate(fs):
            G = np.array([[1 + r * np.cos(f), r * np.sin(f)],
                          [r * np.sin(f), 1 - r * np.cos(f)]])
            w, V = np.linalg.eigh(G)
            M = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
            states.append((dict(i=i, j=j), M[:, 0], M[:, 1]))
    rows = evaluate(states, f"{OUT}/cache_grid.json", methods)

    D = {k: np.full((n_r, n_f), np.nan) for k in STYLE if k in rows[0]}
    for r in rows:
        for k in D:
            if r.get(k) is not None:
                D[k][r["i"], r["j"]] = r[k]
    lows, ups = sides(rows)
    old = (D["rect"] - D["square"]) / D["rect"]
    new = ((np.fmin.reduce([D[k] for k in ups])
            - np.fmax.reduce([D[k] for k in lows]))
           / np.fmin.reduce([D[k] for k in ups]))

    F, R = np.meshgrid(fs, rs)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    vmax = 100 * np.nanmax(old)
    for a, W, t in ((ax[0], old, "было: прямоугольник / квадрат"),
                    (ax[1], new, "стало: лучшее из всех методов")):
        im = a.pcolormesh(F, R, 100 * W, cmap="viridis", vmin=0, vmax=vmax,
                          shading="auto")
        a.set_xlabel("$\\varphi$"); a.set_ylabel("$r$")
        a.set_title(t + "\nширина коридора, %")
        fig.colorbar(im, ax=a)

    who = np.argmin(np.stack([D[k] for k in ups]), axis=0).astype(float)
    cmap = matplotlib.colors.ListedColormap([STYLE[k][1] for k in ups])
    im = ax[2].pcolormesh(F, R, who, cmap=cmap, vmin=-0.5, vmax=len(ups) - 0.5,
                          shading="auto")
    cb = fig.colorbar(im, ax=ax[2], ticks=range(len(ups)))
    cb.ax.set_yticklabels([STYLE[k][0] for k in ups], fontsize=8)
    ax[2].set_xlabel("$\\varphi$"); ax[2].set_ylabel("$r$")
    ax[2].set_title("кто даёт верхнюю оценку\nв каждой точке")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_map.png", dpi=160)
    plt.close(fig)
    print("коридор: было среднее %.1f%%, стало %.1f%%"
          % (100 * np.nanmean(old), 100 * np.nanmean(new)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22, help="точек на срезе")
    ap.add_argument("--grid", action="store_true", help="считать ещё и рис. 3")
    ap.add_argument("--grid-size", type=int, nargs=2, default=(12, 15))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("методы:", ", ".join(build_methods()))
    rows = figure_slice(args.n, build_methods())
    gaps = figure_gaps(rows)
    print("\nзазоры в самоподобной точке:")
    for k, g in gaps:
        print("  %-28s %6.2f %%" % (STYLE[k][0], g))
    if args.grid:
        figure_grid(*args.grid_size, build_methods(light=True))
    print("\nготово:", ", ".join(sorted(os.listdir(OUT))))
