"""
visualization.py
All plots.  Every function saves a PNG (150 dpi) and closes the figure.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

DRONE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


# map helpers

def draw_map(ax, env, show_targets=True, hazard_mode="max"):
    #Priority background + hazard + no-fly zones + targets.
    n = env.size
    ax.imshow(env.priority, origin="lower", cmap="Greys", vmin=0, vmax=4.5, alpha=0.55,
              extent=(-0.5, n - 0.5, -0.5, n - 0.5))
    hz = env.hazard.max(axis=0) if hazard_mode == "max" else env.hazard[hazard_mode]
    ax.imshow(np.ma.masked_where(hz <= 0.01, hz), origin="lower", cmap="Reds", vmin=0, vmax=1.3,
              alpha=0.45, extent=(-0.5, n - 0.5, -0.5, n - 0.5))
    for z in env.nofly_zones:
        ax.add_patch(Rectangle((z.x0 - 0.5, z.y0 - 0.5), z.x1 - z.x0 + 1, z.y1 - z.y0 + 1,
                               facecolor="none", edgecolor="black", hatch="////", lw=1.2))
    ax.add_patch(Rectangle((-0.5, -0.5), n, env.cfg.deploy_band, facecolor="none",
                           edgecolor="#555", ls=":", lw=1))
    if show_targets:
        ax.scatter(env.targets[:, 0], env.targets[:, 1], s=14, marker="*", c="#FFD400",
                   edgecolors="black", linewidths=0.4, zorder=4)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, n + 1, 20))
    ax.set_yticks(range(0, n + 1, 20))


def draw_paths(ax, ch, lw_scale=1.0):
    #Drone routes: thick = high sensor power, dotted = sensor OFF on that leg.
    for d in range(ch.n_drones):
        c = DRONE_COLORS[d % len(DRONE_COLORS)]
        P = ch.pos[d]
        for k in range(ch.n_legs):
            on = ch.on[d, k]
            lw = (1.0 + 2.2 * ch.power[d, k]) * lw_scale if on else 1.0 * lw_scale
            ax.plot(P[k:k + 2, 0], P[k:k + 2, 1], color=c, lw=lw, ls="-" if on else (0, (1, 2)),
                    alpha=0.95, zorder=5, solid_capstyle="round")
        ax.scatter(P[1:, 0], P[1:, 1], s=9, color=c, zorder=6)
        ax.scatter(P[0, 0], P[0, 1], s=55, marker="s", color=c, edgecolors="white", zorder=7)


def _legend_handles():
    h = [Line2D([0], [0], marker="s", color="w", markerfacecolor=DRONE_COLORS[i], markersize=8,
                label=f"Drone {i + 1} (square = launch)") for i in range(5)]
    h += [Patch(facecolor="#d9d9d9", label="Disaster priority (darker = higher)"),
          Patch(facecolor="#f4a7a0", label="Hazard zone (max over time)"),
          Patch(facecolor="white", edgecolor="black", hatch="////", label="No-fly zone"),
          Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD400", markeredgecolor="black",
                 markersize=9, label="Hidden target"),
          Line2D([0], [0], color="#555", lw=1, ls=(0, (1, 2)), label="Sensor OFF (dotted leg)")]
    return h


# figures

def plot_environment(env, path):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    draw_map(ax, env)
    for i, z in enumerate(env.disaster_zones):
        cx, cy = (z.params[0], z.params[1]) if z.kind == "circle" else \
                 ((z.params[0] + z.params[2]) / 2, (z.params[1] + z.params[3]) / 2)
        ax.text(cx, cy, f"Z{i + 1}\nP={z.priority:g}", ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=8)
    for z in env.nofly_zones:
        ax.text((z.x0 + z.x1) / 2, (z.y0 + z.y1) / 2, z.name[:2], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none"), zorder=8)
    ax.text(50, 4.5, "launch staging strip (rows 0-9)", ha="center", fontsize=8, color="#333", zorder=8)
    ax.set_title("Mission grid (100 x 100): disaster zones, hazards, no-fly zones, targets")
    ax.set_xlabel("x (cell)"); ax.set_ylabel("y (cell)")
    ax.legend(handles=_legend_handles()[5:], loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    _save(fig, path)


def plot_convergence(hist, path, title="Fitness convergence: best vs average"):
    g = np.arange(len(hist.best))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(g, hist.best, color="#0072B2", lw=2.2, label="Best fitness")
    ax.plot(g, hist.avg, color="#D55E00", lw=2.0, label="Average fitness")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness (higher is better)")
    ax.set_ylim(min(np.min(hist.avg), np.min(hist.best)) - 0.05, max(hist.best) + 0.05)
    ax.set_title(title)
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_diversity(hist, path):
    g = np.arange(len(hist.diversity))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(g, hist.diversity, color="#009E73", lw=2, label="Population diversity (mean waypoint std, cells)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Diversity (cells)")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(g, hist.feasible, color="#CC79A7", lw=2, ls="--", label="Fraction with no no-fly violation")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Feasible fraction")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8)
    ax.set_title("Diversity and constraint satisfaction over generations")
    ax.grid(alpha=0.25)
    _save(fig, path)


def _snapshot_gens(hist, n=6):
    last = len(hist.best_chrom) - 1
    cand = [0, 3, 10, 30, 80, last]
    gens = sorted(set(min(c, last) for c in cand))
    while len(gens) < n and len(gens) < last + 1:
        gens.append(max(set(range(last + 1)) - set(gens)))
        gens = sorted(set(gens))
    return gens[:n]


def plot_paths_evolution(env, hist, path, gens=None):
    gens = gens or _snapshot_gens(hist)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, g in zip(axes.ravel(), gens):
        draw_map(ax, env, show_targets=True)
        draw_paths(ax, hist.best_chrom[g], lw_scale=0.9)
        m = hist.best_metrics[g]
        ax.set_title(f"Generation {g}   (fitness {hist.best[g]:.3f})", fontsize=10)
        ax.text(1, 96, f"cov {m['coverage']:.2f}  det {m['detection']:.2f}", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.85), zorder=9)
    fig.suptitle("Evolution of the best UAV flight paths across generations", fontweight="bold", fontsize=13)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=2.5)
    _save(fig, path)


def plot_coverage_evolution(env, hist, evaluator, path, gens=None):
    gens = gens or _snapshot_gens(hist)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    n = env.size
    for ax, g in zip(axes.ravel(), gens):
        m = evaluator.simulate(hist.best_chrom[g], want_maps=True)
        lv = m["levelmap"].astype(float)
        ax.imshow(env.priority, origin="lower", cmap="Greys", vmin=0, vmax=6, alpha=0.35,
                  extent=(-0.5, n - 0.5, -0.5, n - 0.5))
        im = ax.imshow(np.ma.masked_where(lv == 0, lv), origin="lower", cmap="YlOrRd", vmin=1,
                       vmax=evaluator.levels, extent=(-0.5, n - 0.5, -0.5, n - 0.5), alpha=0.9)
        for z in env.nofly_zones:
            ax.add_patch(Rectangle((z.x0 - 0.5, z.y0 - 0.5), z.x1 - z.x0 + 1, z.y1 - z.y0 + 1,
                                   fill=False, edgecolor="black", hatch="////", lw=1))
        for k, z in enumerate(env.disaster_zones):                  # outline disaster zones
            if z.kind == "circle":
                ax.add_patch(plt.Circle((z.params[0], z.params[1]), z.params[2], fill=False,
                                        ec="#0072B2", lw=1.4, ls="--"))
            else:
                x0, y0, x1, y1 = z.params
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#0072B2", lw=1.4, ls="--"))
        ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5); ax.set_aspect("equal")
        ax.set_title(f"Generation {g}   coverage {m['coverage']:.2f}", fontsize=10)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02, ticks=range(1, evaluator.levels + 1))
    cb.set_label("Best sensor level that saw the cell (1 = weak ... 4 = strong)")
    fig.suptitle("Coverage heatmaps of the best plan across generations (blue dashed = disaster zones)",
                 fontweight="bold", fontsize=13)
    _save(fig, path)


def plot_final_solution(env, ch, metrics, path):
    fig, ax = plt.subplots(figsize=(8.5, 8))
    draw_map(ax, env)
    draw_paths(ax, ch, lw_scale=1.2)
    ax.set_title("Final evolved UAV mission plan (best individual)")
    ax.set_xlabel("x (cell)"); ax.set_ylabel("y (cell)")
    txt = (f"fitness {metrics['raw']:.3f}\ncoverage {metrics['coverage']:.1%}   detection {metrics['detection']:.1%}\n"
           f"energy {metrics['energy_n']:.2f}  time {metrics['time_n']:.2f}  hazard {metrics['hazard_n']:.2f}\n"
           f"no-fly cells {metrics['nfz_cells']}   max range violation {metrics['range_excess']:.2f}")
    ax.text(1.5, 97.5, txt, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9), zorder=9)
    ax.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
    _save(fig, path)


def plot_objectives(hist, path):
    # --- How each objective of the best individual changes over the generations.
    g = np.arange(len(hist.best_metrics))
    keys = [("coverage", "Coverage (max)", "#0072B2"), ("detection", "Detection rate (max)", "#009E73"),
            ("energy_n", "Energy (min)", "#D55E00"), ("time_n", "Time (min)", "#CC79A7"),
            ("hazard_n", "Hazard exposure (min)", "#222222")]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for k, lab, c in keys:
        ax.plot(g, [m[k] for m in hist.best_metrics], lw=2, label=lab, color=c)
    ax.set_xlabel("Generation"); ax.set_ylabel("Normalised value")
    ax.set_title("Objectives of the best plan: the trade-off being negotiated")
    ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(alpha=0.25)
    _save(fig, path)


def plot_tradeoff(random_metrics, final_pop, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([m["energy_n"] for m in random_metrics], [m["coverage"] for m in random_metrics],
               s=10, c="#bbbbbb", label="random plans")
    sc = ax.scatter([c.metrics["energy_n"] for c in final_pop], [c.metrics["coverage"] for c in final_pop],
                    s=26, c=[c.metrics["hazard_n"] for c in final_pop], cmap="Reds", edgecolors="black",
                    linewidths=0.4, label="final GA population")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Hazard exposure (normalised)")
    ax.set_xlabel("Energy used (normalised)"); ax.set_ylabel("Priority-weighted coverage")
    ax.set_title("Trade-off: coverage vs energy (final GA population vs random plans)")
    ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.25)
    _save(fig, path)


# experiment figures

def plot_ga_vs_random(res, path):
    ga = np.array([r["curve_best"] for r in res["ga"]])
    gavg = np.array([r["curve_avg"] for r in res["ga"]])
    rs = np.array([r["curve_best"] for r in res["rs"]])
    g = np.arange(ga.shape[1])
    gr = np.arange(rs.shape[1])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for arr, c, lab, x in [(ga, "#0072B2", "GA - best", g), (rs, "#D55E00", "Random search - best so far", gr)]:
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(x, mu, color=c, lw=2.2, label=lab)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.plot(g, gavg.mean(0), color="#0072B2", lw=1.4, ls="--", label="GA - population average")
    ax.set_ylim(-0.3, ga.max() + 0.06)
    ax.set_xlabel("Generation  (= one population-size batch of fitness evaluations)")
    ax.set_ylabel("Fitness"); ax.grid(alpha=0.25); ax.legend(frameon=False, loc="lower right")
    n = len(res["ga"])
    ax.set_title(f"GA vs random search, equal evaluation budget (mean +/- std, {n} seeds)")
    _save(fig, path)


def plot_random_distribution(rand_fit, ga_best, rs_best, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.hist(rand_fit, bins=60, color="#bbbbbb", edgecolor="white", label=f"{len(rand_fit):,} random plans")
    ax.axvline(np.max(rand_fit), color="#D55E00", lw=2, label=f"best random plan ({np.max(rand_fit):.3f})")
    ax.axvline(ga_best, color="#0072B2", lw=2.4, label=f"best GA plan ({ga_best:.3f})")
    ax.set_xlabel("Fitness"); ax.set_ylabel("Number of plans")
    ax.set_title("Where the evolved plan sits in the distribution of random plans")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    _save(fig, path)


def plot_sensitivity(sens, path):
    # sens = {'pop_size': {value: [runs...]}, 'mutation_rate': {...}, 'tournament_size': {...}}"""
    names = {"pop_size": "Population size", "mutation_rate": "Mutation rate",
             "tournament_size": "Selection pressure (tournament size)"}
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for j, (p, label) in enumerate(names.items()):
        vals = list(sens[p].keys())
        ax = axes[0, j]
        for i, v in enumerate(vals):
            curves = np.array([r["curve_best"] for r in sens[p][v]])
            ax.plot(curves.mean(0), color=cols[i % 4], lw=1.8, label=f"{v}")
        ax.set_title(label, fontsize=10); ax.set_xlabel("Generation")
        if j == 0: ax.set_ylabel("Mean best fitness")
        ax.set_ylim(-0.1, None)
        ax.legend(frameon=False, fontsize=8, title="value", title_fontsize=8); ax.grid(alpha=0.25)
        ax = axes[1, j]
        data = [[r["final"] for r in sens[p][v]] for v in vals]
        ax.boxplot(data, tick_labels=[str(v) for v in vals], patch_artist=True,
                   boxprops=dict(facecolor="#cfe3f4"), medianprops=dict(color="#D55E00", lw=2))
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-0.08, 0.08, len(d)), d,
                       s=14, color="#222", zorder=3)
        ax.set_xlabel(label.split(" (")[0]);
        if j == 0: ax.set_ylabel("Final best fitness (per seed)")
        ax.grid(alpha=0.25)
    fig.suptitle("Hyper-parameter sensitivity (same 100 generations, 5 seeds per setting)", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_weight_profiles(prof, path):
    # Mean objective values of the best plan under different fitness weight profiles.
    keys = [("coverage", "Coverage"), ("detection", "Detection"), ("energy_n", "Energy"),
            ("time_n", "Time"), ("hazard_n", "Hazard")]
    names = list(prof.keys())
    x = np.arange(len(keys))
    wdt = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cols = ["#0072B2", "#D55E00", "#009E73"]
    for i, nme in enumerate(names):
        mu = [np.mean([r["metrics"][k] for r in prof[nme]]) for k, _ in keys]
        sd = [np.std([r["metrics"][k] for r in prof[nme]]) for k, _ in keys]
        ax.bar(x + i * wdt - 0.4 + wdt / 2, mu, wdt, yerr=sd, capsize=2, color=cols[i % 3], label=nme)
    ax.set_xticks(x); ax.set_xticklabels([k for _, k in keys])
    ax.set_ylabel("Normalised value of the evolved best plan")
    ax.set_title("Effect of the fitness weighting on the evolved solution")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, axis="y")
    _save(fig, path)