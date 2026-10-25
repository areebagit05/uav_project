"""
main.py
Run full GA simulation with the default settings and produce  convergence curves, evolved flight paths, coverage heatmaps

"""
import json
import os
import numpy as np

from config import Config
from environment import Environment
from genetic_algorithm import GeneticAlgorithm
from random_search import RandomSearch
import visualization as viz

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    cfg = Config()
    env = Environment(cfg.env)
    print(f"Grid {env.size}x{env.size} | drones {cfg.env.n_drones} | waypoints/drone {cfg.env.n_waypoints} "
          f"| targets {len(env.targets)} | max range {cfg.env.max_range:g}")
    print(f"GA: population {cfg.ga.pop_size}, generations {cfg.ga.generations}, "
          f"mutation {cfg.ga.mutation_rate}, tournament {cfg.ga.tournament_size}, seed {cfg.ga.seed}\n")

    #  run GA
    ga = GeneticAlgorithm(env, cfg)
    hist = ga.run(verbose=True)
    best = ga.best_of(hist)
    m = best.metrics
    print(f"\nStopped: {hist.stop_reason}  |  evaluations {hist.evaluations}  |  {hist.runtime:.1f} s")
    print("Best plan -> "
          f"fitness {hist.best[int(np.argmax(hist.best))]:.4f}, coverage {m['coverage']:.1%}, "
          f"detection {m['detection']:.1%}, energy {m['energy_n']:.2f}, time {m['time_n']:.2f}, "
          f"hazard {m['hazard_n']:.2f}, no-fly cells {m['nfz_cells']}")

    # random-search comparison
    rs = RandomSearch(env, cfg)
    rres = rs.run(hist.evaluations, cfg.ga.pop_size)
    print(f"Random search with the same budget ({hist.evaluations} evaluations): best fitness {rres['best']:.4f}")

    #  plots
    final_m = ga.evaluator.simulate(best, want_maps=True)
    viz.plot_environment(env, f"{OUT}/01_environment.png")
    viz.plot_convergence(hist, f"{OUT}/02_fitness_convergence.png")
    viz.plot_paths_evolution(env, hist, f"{OUT}/03_flight_paths_evolution.png")
    viz.plot_coverage_evolution(env, hist, ga.evaluator, f"{OUT}/04_coverage_heatmaps_evolution.png")
    viz.plot_final_solution(env, best, final_m, f"{OUT}/05_final_solution.png")
    viz.plot_diversity(hist, f"{OUT}/06_diversity_and_feasibility.png")
    viz.plot_objectives(hist, f"{OUT}/07_objectives_over_generations.png")
    rnd = [ga._prepare(c) for c in ga.factory.random_population(np.random.default_rng(99), 400)]
    viz.plot_tradeoff([c.metrics for c in rnd], hist.final_population, f"{OUT}/08_tradeoff_coverage_vs_energy.png")

    #  save raw results
    summary = dict(
        best_fitness=float(max(hist.best)), stop_reason=hist.stop_reason, evaluations=hist.evaluations,
        runtime_s=hist.runtime, random_search_best=float(rres["best"]),
        metrics={k: (float(v) if np.isscalar(v) else np.asarray(v).tolist())
                 for k, v in m.items() if k != "levelmap"},
        best_curve=hist.best, avg_curve=hist.avg, diversity=hist.diversity)
    with open(f"{OUT}/main_run_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nDone. Figures and summary are in: {OUT}")


if __name__ == "__main__":
    main()

