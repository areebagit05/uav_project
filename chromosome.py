"""
chromosome.py
Chromosome encoding (hybrid integer / real / binary, one block per drone).

For every drone d (D drones) and K legs:
    pos    [d, 0..K, (x, y)]   INTEGER   waypoint 0 is the DEPLOYMENT (launch) point,
                                         waypoints 1..K are the route
    speed  [d, 0..K-1]         REAL      speed on the leg that ends at waypoint k+1
    power  [d, 0..K-1]         REAL      sensor power 0..1 on that leg
    on     [d, 0..K-1]         BINARY    sensor ON / OFF on that leg (activation schedule)

Genes per drone = 2 (launch) + 2K (waypoints) + K (speed) + K (power) + K (on/off)
                = 2 + 5K ... = 52 for K = 10  ->  260 genes for 5 drones.
"""
import numpy as np

from config import Config
from environment import Environment


class Chromosome:
      # One complete mission plan for the whole fleet.

    __slots__ = ("pos", "speed", "power", "on", "metrics")

    def __init__(self, pos, speed, power, on):
        self.pos = pos            # (D, K+1, 2) int
        self.speed = speed        # (D, K) float
        self.power = power        # (D, K) float in [0, 1]
        self.on = on              # (D, K) bool
        self.metrics = None       # filled in by FitnessEvaluator

    def copy(self) -> "Chromosome":
        return Chromosome(self.pos.copy(), self.speed.copy(), self.power.copy(), self.on.copy())

    @property
    def n_drones(self) -> int:
        return self.pos.shape[0]

    @property
    def n_legs(self) -> int:
        return self.speed.shape[1]

    def to_vector(self) -> np.ndarray:
        #Flat gene vector (useful for printing / distance measures)
        return np.concatenate([self.pos.ravel(), self.speed.ravel(),
                               self.power.ravel(), self.on.ravel().astype(float)])


class ChromosomeFactory:
    # Creates random chromosomes (used for the initial population and for random search)

    def __init__(self, env: Environment, cfg: Config):
        self.env = env
        self.cfg = cfg

    def random(self, rng: np.random.Generator, biased: bool = False) -> Chromosome:
        """
        Random plan.  Routes are bounded random walks (step 5..28 cells) so that a
        random route is not absurdly longer than the drone range.  If ``biased`` is
        True, half of the steps jump towards a priority-weighted random cell instead.
        """
        e, d = self.cfg.env, self.cfg.drone
        D, K, n = e.n_drones, e.n_waypoints, e.grid_size
        pos = np.zeros((D, K + 1, 2), dtype=int)
        pos[:, 0, 0] = rng.integers(0, n, D)                     # launch x anywhere
        pos[:, 0, 1] = rng.integers(0, e.deploy_band, D)         # launch y inside the staging strip
        for dr in range(D):
            cur = pos[dr, 0].astype(float)
            for k in range(1, K + 1):
                if biased and rng.random() < 0.5:
                    cand = self.env.sample_priority_cells(rng, 6)
                    dist = np.hypot(*(cand - cur).T)
                    nxt = cand[int(np.argmin(dist))].astype(float)
                else:
                    ang = rng.uniform(0, 2 * np.pi)
                    step = rng.uniform(5, 28)
                    nxt = cur + step * np.array([np.cos(ang), np.sin(ang)])
                nxt = np.clip(np.rint(nxt), 0, n - 1)
                pos[dr, k] = nxt
                cur = nxt
        speed = rng.uniform(d.speed_min, d.speed_max, (D, K))
        power = rng.uniform(0.0, 1.0, (D, K))
        on = rng.random((D, K)) < 0.7
        return Chromosome(pos, speed, power, on)

    def random_population(self, rng, size: int):
        # Half unbiased, half priority-biased random plans.
        return [self.random(rng, biased=(i % 2 == 1)) for i in range(size)]
