"""
environment.py
==============
The mission space: a discretised 100 x 100 grid.

Layers stored on the grid (all indexed as ``[y, x]``):
  * priority  - value of every cell (disaster zones are worth more)
  * hazard    - danger level 0..1, *time dependent* (volatile zones drift and grow)
  * nofly     - boolean mask of forbidden cells
  * targets   - hidden survivors / points of interest inside disaster zones
"""
from dataclasses import dataclass
import numpy as np

from config import EnvConfig
# Zone descriptions (a fixed, hand-designed scenario so results are repeatable)

@dataclass(frozen=True)
class DisasterZone:
    name: str
    kind: str          # 'circle' -> (cx, cy, r) ; 'rect' -> (x0, y0, x1, y1)
    params: tuple
    priority: float    # 3 = high, 2 = medium, 1 = low
    n_targets: int


@dataclass(frozen=True)
class HazardZone:
    name: str
    kind: str          # 'wind' or 'radiation'
    center0: tuple     # (x, y) at mission start
    center1: tuple     # (x, y) at mission end (volatile zones drift)
    radius0: float
    radius1: float     # (volatile zones grow / shrink)
    level: float       # wind = 0.5, radiation = 1.0
    volatile: bool


@dataclass(frozen=True)
class NoFlyZone:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int


DISASTER_ZONES = [
    DisasterZone("Z1 earthquake centre", "circle", (62, 72, 10), 3.0, 12),
    DisasterZone("Z2 collapsed district", "rect", (15, 50, 30, 65), 3.0, 12),
    DisasterZone("Z3 flooded suburb", "circle", (75, 35, 9), 2.0, 7),
    DisasterZone("Z4 industrial fire", "rect", (40, 20, 55, 32), 2.0, 7),
    DisasterZone("Z5 remote village", "circle", (85, 85, 8), 1.0, 7),
]

HAZARD_ZONES = [
    HazardZone("H1 storm front (static)", "wind", (30, 60), (30, 60), 14, 14, 0.5, False),
    HazardZone("H2 gust corridor (drifting)", "wind", (70, 30), (60, 45), 12, 16, 0.5, True),
    HazardZone("H3 radiation pocket (static)", "radiation", (55, 70), (55, 70), 8, 8, 1.0, False),
    HazardZone("H4 radiation leak (growing)", "radiation", (30, 30), (32, 32), 5, 12, 1.0, True),
]

NOFLY_ZONES = [
    NoFlyZone("N1 airport", 44, 40, 56, 52),
    NoFlyZone("N2 military site", 5, 75, 15, 90),
    NoFlyZone("N3 government area", 78, 55, 92, 65),
]


class Environment:
    """Builds and stores all grid layers."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.size = cfg.grid_size
        self.rng = np.random.default_rng(cfg.seed)
        n = self.size
        self.yy, self.xx = np.mgrid[0:n, 0:n]

        self.disaster_zones = DISASTER_ZONES
        self.hazard_zones = HAZARD_ZONES
        self.nofly_zones = NOFLY_ZONES

        self.nofly = self._build_nofly()
        self.zone_id = np.full((n, n), -1, dtype=int)        # which disaster zone a cell belongs to
        self.priority = self._build_priority()
        self.hazard = self._build_hazard()                   # (snapshots, n, n)
        self.targets, self.target_w = self._build_targets()
        self._build_nearest_free()
        self._build_priority_sampler()

    # ------------------------------------------------------------------ #
    def _zone_mask(self, z: DisasterZone) -> np.ndarray:
        if z.kind == "circle":
            cx, cy, r = z.params
            return (self.xx - cx) ** 2 + (self.yy - cy) ** 2 <= r ** 2
        x0, y0, x1, y1 = z.params
        return (self.xx >= x0) & (self.xx <= x1) & (self.yy >= y0) & (self.yy <= y1)

    def _build_nofly(self) -> np.ndarray:
        m = np.zeros((self.size, self.size), dtype=bool)
        for z in self.nofly_zones:
            m[z.y0:z.y1 + 1, z.x0:z.x1 + 1] = True
        return m

    def _build_priority(self) -> np.ndarray:
        p = np.full((self.size, self.size), self.cfg.background_priority, dtype=float)
        for i, z in enumerate(self.disaster_zones):
            m = self._zone_mask(z)
            higher = m & (z.priority > p)
            p[higher] = z.priority
            self.zone_id[higher] = i
        return p

    def hazard_snapshot(self, frac: float) -> np.ndarray:
        """Hazard map at mission-time fraction ``frac`` (0 = start, 1 = end)."""
        h = np.zeros((self.size, self.size))
        for z in self.hazard_zones:
            cx = z.center0[0] + frac * (z.center1[0] - z.center0[0])
            cy = z.center0[1] + frac * (z.center1[1] - z.center0[1])
            r = z.radius0 + frac * (z.radius1 - z.radius0)
            d = np.hypot(self.xx - cx, self.yy - cy)
            # full strength inside 75 % of the radius, fading to 0 at the edge
            v = z.level * np.clip((r - d) / (0.25 * r), 0.0, 1.0)
            h = np.maximum(h, v)
        return h

    def _build_hazard(self) -> np.ndarray:
        s = self.cfg.n_snapshots
        return np.stack([self.hazard_snapshot(i / (s - 1)) for i in range(s)])

    def _build_targets(self):
        pts, w = [], []
        for z in self.disaster_zones:
            m = self._zone_mask(z) & ~self.nofly
            cells = np.argwhere(m)                           # rows of (y, x)
            pick = self.rng.choice(len(cells), size=min(z.n_targets, len(cells)), replace=False)
            for y, x in cells[pick]:
                pts.append((x, y))
                w.append(z.priority)
        return np.array(pts, dtype=int), np.array(w, dtype=float)

    def _build_nearest_free(self):
        """For every no-fly cell, the closest free cell (used by the repair operator)."""
        n = self.size
        self.nearest_free = np.stack([self.xx, self.yy], axis=-1).copy()   # (n, n, 2) as (x, y)
        free = np.argwhere(~self.nofly)                       # (y, x)
        blocked = np.argwhere(self.nofly)
        for y, x in blocked:
            d = (free[:, 0] - y) ** 2 + (free[:, 1] - x) ** 2
            j = int(np.argmin(d))
            self.nearest_free[y, x] = (free[j, 1], free[j, 0])

    def _build_priority_sampler(self):
        """Probability of drawing each cell proportional to its priority (no-fly excluded)."""
        w = np.where(self.nofly, 0.0, self.priority).ravel()
        self._cell_p = w / w.sum()

    # ------------------------------------------------------------------ #
    # helpers used by the GA
    # ------------------------------------------------------------------ #
    def sample_priority_cells(self, rng, k: int) -> np.ndarray:
        """Draw ``k`` cells (x, y) with probability proportional to priority."""
        idx = rng.choice(self._cell_p.size, size=k, p=self._cell_p)
        return np.stack([idx % self.size, idx // self.size], axis=1)

    def clip_xy(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, 0, self.size - 1)

    @property
    def total_priority(self) -> float:
        return float(self.priority.sum())
