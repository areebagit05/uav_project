"""
operators.py
Genetic operators: selection, crossover and mutation.

Crossover and mutation are spatial: they act on waypoint coordinates (geometry)
instead of blindly flipping bits.
"""
import numpy as np

from config import Config
from environment import Environment
from chromosome import Chromosome


# SELECTION

class TournamentSelection:
    # Pick k random individuals, the best one wins.  Larger k = higher selection pressure.

    def __init__(self, k: int):
        self.k = k

    def select(self, fit: np.ndarray, rng, n: int = 2):
        idx = rng.integers(0, len(fit), size=(n, self.k))
        return idx[np.arange(n), np.argmax(fit[idx], axis=1)]


class RankSelection:
    # Probability proportional to rank (robust to wildly different fitness scales).

    def select(self, fit: np.ndarray, rng, n: int = 2):
        ranks = np.empty(len(fit))
        ranks[np.argsort(fit)] = np.arange(1, len(fit) + 1)
        return rng.choice(len(fit), size=n, p=ranks / ranks.sum())


class RouletteSelection:
    # Fitness-proportional ('roulette wheel') selection on shifted fitness.

    def select(self, fit: np.ndarray, rng, n: int = 2):
        shifted = fit - fit.min() + 1e-9
        return rng.choice(len(fit), size=n, p=shifted / shifted.sum())


def make_selection(cfg: Config):
    name = cfg.ga.selection
    if name == "tournament":
        return TournamentSelection(cfg.ga.tournament_size)
    if name == "rank":
        return RankSelection()
    if name == "roulette":
        return RouletteSelection()
    raise ValueError(f"unknown selection '{name}'")


# CROSSOVER

class SpatialCrossover:
    """
    For every drone the child inherits its plan through ONE of four rules:

      swap     (25 %)  take the whole drone plan of parent B            (route-level swap)
      blend    (35 %)  GEOMETRIC crossover: on a random waypoint segment each waypoint is
                       alpha*A + (1-alpha)*B  (a point on the line between the parents'
                       waypoints), speed / power are blended the same way
      segment  (25 %)  SEGMENT SWAP: a contiguous block of waypoints and their
                       speed/power/sensor genes is copied from B
      keep     (15 %)  keep parent A's plan

    Because the rules work on coordinates, the child's route stays spatially
    sensible (no random jumps across the map).
    """

    def __init__(self, p_swap=0.25, p_blend=0.35, p_segment=0.25):
        self.cuts = np.cumsum([p_swap, p_blend, p_segment])

    def cross(self, a: Chromosome, b: Chromosome, rng) -> Chromosome:
        c = a.copy()
        D, K = a.speed.shape
        for d in range(D):
            r = rng.random()
            if r < self.cuts[0]:  # whole-drone swap
                c.pos[d], c.speed[d] = b.pos[d], b.speed[d]
                c.power[d], c.on[d] = b.power[d], b.on[d]
            elif r < self.cuts[1]:  # geometric blend
                i, j = sorted(rng.integers(0, K + 1, 2))
                alpha = rng.random()
                c.pos[d, i:j + 1] = np.rint(alpha * a.pos[d, i:j + 1] + (1 - alpha) * b.pos[d, i:j + 1])
                lo = max(i - 1, 0)
                c.speed[d, lo:j] = alpha * a.speed[d, lo:j] + (1 - alpha) * b.speed[d, lo:j]
                c.power[d, lo:j] = alpha * a.power[d, lo:j] + (1 - alpha) * b.power[d, lo:j]
                pick_b = rng.random(j - lo) < 0.5
                c.on[d, lo:j] = np.where(pick_b, b.on[d, lo:j], a.on[d, lo:j])
            elif r < self.cuts[2]:  # segment swap
                i, j = sorted(rng.integers(0, K + 1, 2))
                lo = max(i - 1, 0)
                c.pos[d, i:j + 1] = b.pos[d, i:j + 1]
                c.speed[d, lo:j] = b.speed[d, lo:j]
                c.power[d, lo:j] = b.power[d, lo:j]
                c.on[d, lo:j] = b.on[d, lo:j]
        return c


# MUTATION

class SpatialMutation:
    """
    Five mutation operators, all driven by one rate ``p`` (per-gene probability):

      1. Gaussian shift     every waypoint (and launch point) moves by N(0, sigma) cells  [prob p]
      2. Priority relocate  a waypoint jumps to a priority-weighted random cell near the
                            previous waypoint - steers drones to disaster zones          [prob p/2 per drone]
      3. Segment reversal   a block of waypoints is reversed (2-opt style route repair)  [prob p/2 per drone]
      4. Parameter jitter   speed and sensor power get Gaussian noise                    [prob p per leg]
      5. Sensor flip        ON/OFF bit flips                                              [prob p/2 per leg]

    The adaptive boost multiplies ``p`` when population diversity collapses.
    """

    def __init__(self, env: Environment, cfg: Config):
        self.env = env
        self.cfg = cfg
        self.sigma = cfg.ga.mutation_sigma
        self.base_rate = cfg.ga.mutation_rate

    def mutate(self, ch: Chromosome, rng, boost: float = 1.0, rate=None) -> Chromosome:
        p = min(0.9, (self.base_rate if rate is None else rate) * boost)
        dcfg = self.cfg.drone
        D, K = ch.speed.shape
        n = self.cfg.env.grid_size

        # 1. Gaussian waypoint shift
        mask = rng.random((D, K + 1)) < p
        noise = rng.normal(0.0, self.sigma, (D, K + 1, 2))
        ch.pos = np.where(mask[..., None], np.rint(ch.pos + noise).astype(int), ch.pos)
        ch.pos = np.clip(ch.pos, 0, n - 1)

        for d in range(D):

            # 2. priority-biased relocation
            if rng.random() < p / 2:
                k = int(rng.integers(1, K + 1))
                cand = self.env.sample_priority_cells(rng, 6)
                dist = np.hypot(*(cand - ch.pos[d, k - 1]).T)
                ch.pos[d, k] = cand[int(np.argmin(dist))]

            # 3. segment reversal
            if rng.random() < p / 2:
                i, j = sorted(rng.integers(1, K + 1, 2))
                if j > i:
                    ch.pos[d, i:j + 1] = ch.pos[d, i:j + 1][::-1].copy()

        # 4. speed / power jitter
        m_leg = rng.random((D, K)) < p
        ch.speed = np.where(m_leg, ch.speed + rng.normal(0, 0.25, (D, K)), ch.speed)
        ch.speed = np.clip(ch.speed, dcfg.speed_min, dcfg.speed_max)
        m_pow = rng.random((D, K)) < p
        ch.power = np.clip(np.where(m_pow, ch.power + rng.normal(0, 0.2, (D, K)), ch.power), 0.0, 1.0)

        # 5. sensor ON/OFF flip
        flip = rng.random((D, K)) < p / 2
        ch.on = np.where(flip, ~ch.on, ch.on)
        return ch
