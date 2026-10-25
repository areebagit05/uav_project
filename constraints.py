"""
constraints.py
Hard-constraint handling

  1. REPAIR (applied to every chromosome before it is evaluated)
       - a waypoint that lies inside a no-fly zone is moved to the nearest free cell
       - a route longer than the drone's maximum range is truncated: the leg that
         crosses the limit is cut at the limit and the remaining waypoints collapse
         onto that point (the drone simply stops flying)

  2. DYNAMIC PENALTY (applied to the fitness)
       - a leg may still cross a no-fly zone between two legal waypoints.  Every
         cell flown inside a no-fly zone is penalised with a weight lambda(g) that
         grows linearly with the generation g.  Early on the search may cross
         no-fly zones to explore; later such plans are crushed.
       - when ``range_mode == 'penalty'`` range excess is penalised instead of repaired
         (used in the constraint-handling ablation experiment).
"""
import numpy as np

from config import Config
from environment import Environment
from chromosome import Chromosome


class ConstraintHandler:
    def __init__(self, env: Environment, cfg: Config):
        self.env = env
        self.cfg = cfg
        self.c = cfg.constraints
        self.R = cfg.env.max_range
        self.band = cfg.env.deploy_band
        self.n = cfg.env.grid_size

    #  REPAIR
    def repair(self, ch: Chromosome) -> Chromosome:
        env = self.env
        pos = ch.pos
        pos[...] = np.clip(pos, 0, self.n - 1)
        pos[:, 0, 1] = np.clip(pos[:, 0, 1], 0, self.band - 1)  # launch from the staging strip
        # (a) waypoints inside a no-fly zone -> nearest free cell
        bad = env.nofly[pos[..., 1], pos[..., 0]]
        if bad.any():
            pos[bad] = env.nearest_free[pos[bad][:, 1], pos[bad][:, 0]]
        # (b) range limit
        if self.c.range_mode == "repair":
            self._truncate_range(ch)
        return ch

    def _truncate_range(self, ch: Chromosome):
        pos = ch.pos
        seg = np.hypot(*(pos[:, 1:] - pos[:, :-1]).transpose(2, 0, 1))  # (D, K)
        cum = np.cumsum(seg, axis=1)
        limit = self.R - 1.0  # safety margin for rounding
        for dr in np.where(cum[:, -1] > limit)[0]:
            k = int(np.argmax(cum[dr] > limit))  # first leg that crosses
            before = cum[dr, k] - seg[dr, k]
            frac = (limit - before) / seg[dr, k]
            a, b = pos[dr, k].astype(float), pos[dr, k + 1].astype(float)
            while True:  # step back out of no-fly cells
                pt = np.rint(a + frac * (b - a)).astype(int)
                if not self.env.nofly[pt[1], pt[0]] or frac <= 0:
                    break
                frac -= 0.05
            pos[dr, k + 1:] = pt

    # PENALTY
    def lam(self, gen: int, max_gen: int) -> float:

        # Dynamic penalty weight: grows linearly from lambda_start to lambda_end.
        frac = 0.0 if max_gen <= 0 else min(gen / max_gen, 1.0)
        return self.c.lambda_start + (self.c.lambda_end - self.c.lambda_start) * frac

    def penalty(self, m: dict, lam: float) -> float:
        p = lam * m["nfz_cells"] / self.c.nfz_scale
        if self.c.range_mode == "penalty":
            p += self.c.range_penalty * m["range_excess"]
        return p

    def fitness(self, m: dict, lam: float) -> float:
        return m["raw"] - self.penalty(m, lam)

    def reference_fitness(self, m: dict) -> float:

        # Fitness with the FINAL penalty weight - the fixed yardstick used in all plots.
        return self.fitness(m, self.c.lambda_end)

    def is_feasible(self, m: dict) -> bool:
        return m["nfz_cells"] == 0 and m["range_excess"] <= 1e-9