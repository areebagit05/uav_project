"""
random_search.py
Baseline for the comparison: pure random search.  It uses exactly the same
random plan generator, the same repair operator and the same fitness function
as the GA, and the same number of fitness evaluations - the only difference
is that nothing is *evolved*.
"""
import numpy as np

from config import Config
from environment import Environment
from chromosome import ChromosomeFactory
from fitness import FitnessEvaluator
from constraints import ConstraintHandler


class RandomSearch:
    def __init__(self, env: Environment, cfg: Config, rng=None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(cfg.ga.seed + 1000)
        self.factory = ChromosomeFactory(env, cfg)
        self.evaluator = FitnessEvaluator(env, cfg)
        self.constraints = ConstraintHandler(env, cfg)

    def run(self, n_evals: int, batch: int, keep_all: bool = False):
        """
        Evaluate ``n_evals`` random plans.  ``best_curve`` holds the best-so-far fitness
        after every ``batch`` evaluations (batch = GA population size, so one batch
        corresponds to one GA generation).
        """
        best, best_ch, curve, avg_curve, all_fit = -np.inf, None, [], [], []
        batch_fit = []
        for i in range(1, n_evals + 1):
            ch = self.factory.random(self.rng, biased=bool(i % 2))
            self.constraints.repair(ch)
            m = self.evaluator.evaluate(ch)
            f = self.constraints.reference_fitness(m)
            batch_fit.append(f)
            if keep_all:
                all_fit.append(f)
            if f > best:
                best, best_ch = f, ch
            if i % batch == 0:
                curve.append(best)
                avg_curve.append(float(np.mean(batch_fit)))
                batch_fit = []
        return dict(best=best, best_chrom=best_ch, curve=curve, avg_curve=avg_curve, all_fit=all_fit)
