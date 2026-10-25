"""
experiments.py
Runs every experiment needed for the technical report and saves the plots plus
a JSON file with all raw numbers.

Experiments
  A. GA vs random search      (same number of fitness evaluations, many seeds)
  B. Hyper-parameter study    population size, mutation rate, selection pressure
  C. Fitness-weight profiles  how the weights change the evolved plan
  D. Constraint handling      repair + dynamic penalty vs penalty-only vs weak static penalty
"""
import argparse
import dataclasses as dc
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from config import Config
from environment import Environment
from genetic_algorithm import GeneticAlgorithm
from random_search import RandomSearch
import visualization as viz

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

SCALARS = ["raw", "coverage", "detection", "energy_n", "time_n", "hazard_n", "nfz_cells", "range_excess"]


def _scalars(m):
    return {k: float(m[k]) for k in SCALARS}


def run_ga(args):
    # One GA run.  ``overrides`` = dict of {'ga': {...}, 'weights': {...}, 'constraints': {...}}.
    overrides, seed, generations = args
    cfg = Config()
    cfg.ga = dc.replace(cfg.ga, seed=seed, generations=generations, patience=None, **overrides.get("ga", {}))
    cfg.weights = dc.replace(cfg.weights, **overrides.get("weights", {}))
    cfg.constraints = dc.replace(cfg.constraints, **overrides.get("constraints", {}))
    env = Environment(cfg.env)
    ga = GeneticAlgorithm(env, cfg)
    h = ga.run(verbose=False)
    best = ga.best_of(h)
    # re-score the final plan with the DEFAULT weights so profiles can be compared on equal terms
    base = Config()
    final_m = best.metrics
    return dict(seed=seed, final=float(max(h.best)), curve_best=h.best, curve_avg=h.avg,
                evaluations=h.evaluations, metrics=_scalars(final_m), runtime=h.runtime,
                feasible=bool(final_m["nfz_cells"] == 0 and final_m["range_excess"] <= 1e-9),
                pos=best.pos.tolist())


def run_rs(args):
    n_evals, batch, seed = args
    cfg = Config()
    env = Environment(cfg.env)
    rs = RandomSearch(env, cfg, rng=np.random.default_rng(seed + 5000))
    r = rs.run(n_evals, batch)
    return dict(seed=seed, final=float(r["best"]), curve_best=r["curve"], curve_avg=r["avg_curve"],
                metrics=_scalars(r["best_chrom"].metrics), pos=r["best_chrom"].pos.tolist())


def main(quick=False):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    n_seeds_cmp, n_seeds_sens = (3, 2) if quick else (10, 5)
    gens_cmp, gens_sens = (40, 30) if quick else (150, 100)
    cfg = Config()
    pop = cfg.ga.pop_size
    results = {}

    with Pool(2) as pool:

        #  GA vs random
        print("A. GA vs random search ...")
        ga_runs = pool.map(run_ga, [({}, s, gens_cmp) for s in range(1, n_seeds_cmp + 1)])
        n_evals = ga_runs[0]["evaluations"]
        rs_runs = pool.map(run_rs, [(n_evals, pop, s) for s in range(1, n_seeds_cmp + 1)])
        results["ga_vs_rs"] = dict(ga=ga_runs, rs=rs_runs, n_evals=n_evals)
        viz.plot_ga_vs_random(results["ga_vs_rs"], f"{OUT}/09_ga_vs_random_search.png")

        # long random search -> distribution of random plans
        rs_long = RandomSearch(Environment(cfg.env), cfg, rng=np.random.default_rng(777)).run(
            n_evals * (2 if not quick else 1), pop, keep_all=True)
        results["random_distribution"] = dict(n=len(rs_long["all_fit"]), best=float(rs_long["best"]),
                                              mean=float(np.mean(rs_long["all_fit"])),
                                              p99=float(np.percentile(rs_long["all_fit"], 99)))
        viz.plot_random_distribution(rs_long["all_fit"], max(r["final"] for r in ga_runs), rs_long["best"],
                                     f"{OUT}/10_random_plan_distribution.png")

        #  sensitivity
        print("B. hyper-parameter sensitivity ...")
        grid = {"pop_size": [20, 50, 80, 150],
                "mutation_rate": [0.01, 0.05, 0.10, 0.30],
                "tournament_size": [2, 3, 6, 12]}
        if quick:
            grid = {k: v[:2] for k, v in grid.items()}
        sens = {}
        for p, values in grid.items():
            sens[p] = {}
            for v in values:
                sens[p][v] = pool.map(run_ga, [({"ga": {p: v}}, s, gens_sens) for s in range(1, n_seeds_sens + 1)])
                print(f"   {p}={v}: mean final {np.mean([r['final'] for r in sens[p][v]]):.4f}")
        results["sensitivity"] = {p: {str(v): r for v, r in d.items()} for p, d in sens.items()}
        viz.plot_sensitivity(sens, f"{OUT}/11_hyperparameter_sensitivity.png")

        #  weight profiles
        print("C. fitness weight profiles ...")
        profiles = {
            "balanced (default)": {},
            "coverage-first": dict(coverage=0.50, detection=0.35, energy=0.05, time=0.03, hazard=0.05),
            "cost-averse": dict(coverage=0.25, detection=0.20, energy=0.20, time=0.12, hazard=0.23),
        }
        prof = {}
        for name, w in profiles.items():
            prof[name] = pool.map(run_ga, [({"weights": w}, s, gens_sens) for s in range(1, n_seeds_sens + 1)])
        results["weight_profiles"] = prof
        viz.plot_weight_profiles(prof, f"{OUT}/12_weight_profiles.png")

        #  constraint handling
        print("D. constraint handling ...")
        modes = {
            "repair + dynamic penalty (default)": {},
            "penalty only (no repair of range)": {"constraints": dict(range_mode="penalty")},
            "weak static penalty": {"constraints": dict(lambda_start=0.2, lambda_end=0.2)},
        }
        con = {}
        for name, o in modes.items():
            con[name] = pool.map(run_ga, [(o, s, gens_sens) for s in range(1, n_seeds_sens + 1)])
        results["constraint_modes"] = con

    with open(f"{OUT}/experiment_results.json", "w") as f:
        json.dump(results, f)
    print(f"\nAll experiments finished in {(time.time() - t0) / 60:.1f} min. Results: {OUT}/experiment_results.json")

    # short console summary
    g = [r["final"] for r in results["ga_vs_rs"]["ga"]]
    r_ = [r["final"] for r in results["ga_vs_rs"]["rs"]]
    print(f"GA  final fitness: mean {np.mean(g):.4f} +/- {np.std(g):.4f}")
    print(f"RS  final fitness: mean {np.mean(r_):.4f} +/- {np.std(r_):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny version for testing")
    main(ap.parse_args().quick)
