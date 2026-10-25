"""
genetic_algorithm.py

The GA engine: population initialization, selection, crossover, mutation,
elitism, diversity maintenance and termination criteria.
"""
from dataclasses import dataclass, field
import time
import numpy as np

from config import Config
from environment import Environment
from chromosome import Chromosome, ChromosomeFactory
from fitness import FitnessEvaluator
from constraints import ConstraintHandler
from operators import SpatialCrossover, SpatialMutation, make_selection


@dataclass
class History:
    # Everything recorded during a run (used for plots and the report)
    best: list = field(default_factory=list)  # best reference fitness per generation
    avg: list = field(default_factory=list)  # mean reference fitness per generation
    worst: list = field(default_factory=list)
    diversity: list = field(default_factory=list)  # mean std of waypoint positions (cells)
    feasible: list = field(default_factory=list)  # fraction with zero no-fly cells
    best_metrics: list = field(default_factory=list)  # metrics dict of the best individual
    best_chrom: list = field(default_factory=list)  # copy of the best chromosome per generation
    boosted: list = field(default_factory=list)  # was the adaptive mutation boost active?
    evaluations: int = 0
    stop_reason: str = ""
    runtime: float = 0.0
    final_population: list = field(default_factory=list)


class GeneticAlgorithm:
    def __init__(self, env: Environment, cfg: Config, rng: np.random.Generator = None):
        self.env = env
        self.cfg = cfg
        self.g = cfg.ga
        self.rng = rng if rng is not None else np.random.default_rng(cfg.ga.seed)
        self.factory = ChromosomeFactory(env, cfg)
        self.evaluator = FitnessEvaluator(env, cfg)
        self.constraints = ConstraintHandler(env, cfg)
        self.selection = make_selection(cfg)
        self.crossover = SpatialCrossover()
        self.mutation = SpatialMutation(env, cfg)

    def _prepare(self, ch: Chromosome) -> Chromosome:
        # repair -> evaluate
        self.constraints.repair(ch)
        self.evaluator.evaluate(ch)
        return ch

    @staticmethod
    def diversity(pop) -> float:
        # Mean (over genes) std-dev of waypoint coordinates across the population, in cells.
        arr = np.stack([c.pos for c in pop]).astype(float)
        return float(arr.std(axis=0).mean())

    def run(self, verbose: bool = True) -> History:
        g, con = self.g, self.constraints
        hist = History()
        t0 = time.time()

        #   population initialization
        pop = [self._prepare(c) for c in self.factory.random_population(self.rng, g.pop_size)]
        hist.evaluations += g.pop_size
        best_ref, stagnant = -np.inf, 0

        for gen in range(g.generations + 1):
            lam = con.lam(gen, g.generations)
            sel_fit = np.array([con.fitness(c.metrics, lam) for c in pop])  # used for selection
            ref_fit = np.array([con.reference_fitness(c.metrics) for c in pop])  # used for plots
            ib = int(np.argmax(ref_fit))

            #  record statistics
            div = self.diversity(pop)
            hist.best.append(float(ref_fit[ib]))
            hist.avg.append(float(ref_fit.mean()))
            hist.worst.append(float(ref_fit.min()))
            hist.diversity.append(div)
            hist.feasible.append(float(np.mean([c.metrics["nfz_cells"] == 0 for c in pop])))
            hist.best_metrics.append(pop[ib].metrics)
            bc = pop[ib].copy()
            bc.metrics = pop[ib].metrics
            hist.best_chrom.append(bc)

            #  termination criteria
            if ref_fit[ib] > best_ref + g.min_improvement:
                best_ref, stagnant = float(ref_fit[ib]), 0
            else:
                stagnant += 1
            if verbose and (gen % 10 == 0 or gen == g.generations):
                m = pop[ib].metrics
                print(f"gen {gen:3d} | best {ref_fit[ib]:7.4f} | avg {ref_fit.mean():7.4f} | "
                      f"cov {m['coverage']:.3f} det {m['detection']:.3f} | "
                      f"diversity {div:5.1f} | feasible {hist.feasible[-1]:.2f}")
            if gen == g.generations:
                hist.stop_reason = f"reached max generations ({g.generations})"
                break
            if g.patience is not None and stagnant >= g.patience:
                hist.stop_reason = f"no improvement for {g.patience} generations"
                break

            #  adaptive mutation: boost when diversity collapses
            boost = g.mutation_boost if (g.adaptive_mutation and div < g.diversity_threshold) else 1.0
            hist.boosted.append(boost > 1.0)

            #  build the next generation
            order = np.argsort(-sel_fit)
            new_pop = []
            for i in order[:g.elite_count]:  # elitism
                ec = pop[i].copy()
                ec.metrics = pop[i].metrics  # already evaluated
                new_pop.append(ec)

            n_imm = 0
            if g.immigrant_fraction > 0 and stagnant >= g.stagnation_trigger:  # random immigrants
                n_imm = max(1, int(g.immigrant_fraction * g.pop_size))
            n_children = g.pop_size - len(new_pop) - n_imm
            for _ in range(n_children):
                ia, ib2 = self.selection.select(sel_fit, self.rng, 2)
                if self.rng.random() < g.crossover_rate:
                    child = self.crossover.cross(pop[ia], pop[ib2], self.rng)
                else:
                    child = pop[ia].copy()
                self.mutation.mutate(child, self.rng, boost=boost)
                new_pop.append(self._prepare(child))
            for _ in range(n_imm):
                new_pop.append(self._prepare(self.factory.random(self.rng, biased=bool(self.rng.random() < 0.5))))
            hist.evaluations += n_children + n_imm
            pop = new_pop

        hist.runtime = time.time() - t0
        hist.final_population = pop
        return hist

    #     convenience
    def best_of(self, hist: History) -> Chromosome:
        return hist.best_chrom[int(np.argmax(hist.best))]
