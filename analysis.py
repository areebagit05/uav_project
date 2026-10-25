"""
analysis.py
Structural comparison of evolved plans vs the best random plan
rites outputs/solution_analysis.json.
"""
import dataclasses as dc, json, os
import numpy as np
from config import Config
from environment import Environment
from genetic_algorithm import GeneticAlgorithm
from random_search import RandomSearch
from fitness import FitnessEvaluator

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def describe(env, ev, ch):
    m = ev.simulate(ch, want_maps=True)
    cov = m["levelmap"] > 0
    zones = {}
    for i, z in enumerate(env.disaster_zones):
        mask = env.zone_id == i
        zones[f"Z{i+1}"] = float(cov[mask].mean())
    hi = np.isin(env.zone_id, [0, 1])
    return dict(fitness=float(m["raw"]), coverage=m["coverage"], detection=m["detection"],
                energy_n=m["energy_n"], time_n=m["time_n"], hazard_n=m["hazard_n"],
                zone_cov=zones, high_priority_cov=float(cov[hi].mean()),
                background_cov=float(cov[env.zone_id < 0].mean()),
                path_len=[float(x) for x in m["drone_len"]], cov_mask=cov)


def jaccard(a, b):
    return float((a & b).sum() / (a | b).sum())


def main():
    cfg = Config(); env = Environment(cfg.env); ev = FitnessEvaluator(env, cfg)
    sols = {}
    for s in (1, 2, 3, 4):
        c = dc.replace(cfg.ga, seed=s, patience=None)
        cfg_s = Config(); cfg_s.ga = c
        ga = GeneticAlgorithm(env, cfg_s); h = ga.run(verbose=False)
        sols[f"GA seed {s}"] = describe(env, ev, ga.best_of(h))
    evals = 11780
    for s in (1, 2, 3, 4):
        rs = RandomSearch(env, cfg, rng=np.random.default_rng(s + 5000)).run(evals, 80)
        sols[f"Random seed {s}"] = describe(env, ev, rs["best_chrom"])
    names = list(sols)
    gaN = [n for n in names if n.startswith("GA")]; rnN = [n for n in names if n.startswith("Random")]
    pair = lambda grp: float(np.mean([jaccard(sols[a]["cov_mask"], sols[b]["cov_mask"])
                                      for i, a in enumerate(grp) for b in grp[i + 1:]]))
    out = dict(plans={n: {k: v for k, v in d.items() if k != "cov_mask"} for n, d in sols.items()},
               jaccard_ga_vs_ga=pair(gaN), jaccard_random_vs_random=pair(rnN),
               jaccard_ga_vs_random=float(np.mean([jaccard(sols[a]["cov_mask"], sols[b]["cov_mask"]) for a in gaN for b in rnN])))
    json.dump(out, open(f"{OUT}/solution_analysis.json", "w"), indent=1)
    for n, d in out["plans"].items():
        print(n, round(d["fitness"], 3), "cov", round(d["coverage"], 2), "high", round(d["high_priority_cov"], 2),
              "bg", round(d["background_cov"], 2), "haz", round(d["hazard_n"], 2),
              {k: round(v, 2) for k, v in d["zone_cov"].items()})
    print({k: round(v, 3) for k, v in out.items() if k.startswith("jac")})


if __name__ == "__main__":
    main()