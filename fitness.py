"""
fitness.py

Multi-objective evaluation of a chromosome

  MAXIMISE   coverage   priority-weighted fraction of the map seen by a sensor
             detection  expected (priority-weighted) fraction of hidden targets found
  MINIMISE   energy     flight energy + sensor energy
             time       mission time of the slowest drone
             hazard     time spent inside hazard cells (the hazard map changes with time)
the constraint measures
             nfz_cells     route cells that lie inside a no-fly zone
             range_excess  how far a drone flies beyond its maximum range

Scalar fitness (higher is better):
  raw = w_cov * coverage + w_det * detection
        - w_en * energy_n - w_t * time_n - w_haz * hazard_n

  fitness(gen) = raw - lambda(gen) * nfz_cells / nfz_scale - range penalty
"""
import numpy as np

from config import Config
from environment import Environment
from chromosome import Chromosome


class FitnessEvaluator:
    def __init__(self, env: Environment, cfg: Config):
        self.env = env
        self.cfg = cfg
        self.d = cfg.drone
        self.w = cfg.weights
        self.R = cfg.env.max_range
        self.levels = self.d.sensor_levels
        # sensing radius of level L is L + 1 cells -> pre-compute the disc offsets
        self.pad = self.levels + 1
        self.discs = {}
        for lv in range(1, self.levels + 1):
            r = lv + 1
            oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
            keep = ox ** 2 + oy ** 2 <= r * r
            self.discs[lv] = (oy[keep], ox[keep])
        # reference values used to normalise the cost terms to roughly 0..1
        nominal = self.d.move_c0 / self.d.nominal_speed + self.d.move_c1 * self.d.nominal_speed ** 2
        mid_lvl = (self.levels + 1) // 2 + 1  # a "medium" sensor level
        sensor = (self.d.sensor_s0 + self.d.sensor_s1 * mid_lvl ** 2) / self.d.nominal_speed
        self.energy_ref = cfg.env.n_drones * self.R * (nominal + sensor)
        self.time_ref = self.R / self.d.nominal_speed
        self.time_horizon = 1.5 * self.time_ref  # hazard clock runs 0..horizon
        self.n_snap = cfg.env.n_snapshots

    def level_of(self, power: np.ndarray) -> np.ndarray:
        # Quantise sensor power 0..1 into level 1..L.
        return np.clip(1 + np.floor(power * self.levels).astype(int), 1, self.levels)

    def simulate(self, ch: Chromosome, want_maps: bool = False):
        """
        Run the mission and return a metrics dict.  With ``want_maps`` the sensor
        level map (H x W, 0 = unseen, 1..L = best sensor level that saw the cell)
        is returned as well.
        """
        env, d, w = self.env, self.d, self.w
        D, K = ch.speed.shape
        p = ch.pos

        #  legs (flattened over drones) #
        x0 = p[:, :-1, 0].ravel();
        y0 = p[:, :-1, 1].ravel()
        x1 = p[:, 1:, 0].ravel();
        y1 = p[:, 1:, 1].ravel()
        dx = x1 - x0;
        dy = y1 - y0
        n_s = np.maximum(np.abs(dx), np.abs(dy)) + 1  # samples (cells) per leg
        seg = np.hypot(dx, dy)  # Euclidean leg length
        v = ch.speed.ravel()
        on = ch.on.ravel()
        lvl = self.level_of(ch.power.ravel())
        L = x0.size

        #  rasterise every leg into grid cells  #
        leg_id = np.repeat(np.arange(L), n_s)
        starts = np.cumsum(n_s) - n_s
        offs = np.arange(n_s.sum()) - np.repeat(starts, n_s)
        t = offs / np.maximum(n_s[leg_id] - 1, 1)
        xs = np.rint(x0[leg_id] + t * dx[leg_id]).astype(int)
        ys = np.rint(y0[leg_id] + t * dy[leg_id]).astype(int)
        v_s = v[leg_id]

        # time, energy
        leg_time = seg / v  # (L,)
        lt = leg_time.reshape(D, K)
        drone_time = lt.sum(axis=1)
        makespan = float(drone_time.max())
        flight_e = float(np.sum(seg * (d.move_c0 / v + d.move_c1 * v ** 2)))
        sensor_e = float(np.sum(on[leg_id] * (d.sensor_s0 + d.sensor_s1 * lvl[leg_id] ** 2) / v_s))
        energy = flight_e + sensor_e

        # ---- hazard exposure (hazard map depends on the mission clock)
        leg_start = (np.cumsum(lt, axis=1) - lt).ravel()
        clock = leg_start[leg_id] + t * leg_time[leg_id]
        snap = np.minimum((clock / self.time_horizon * self.n_snap).astype(int), self.n_snap - 1)
        exposure = float(np.sum(env.hazard[snap, ys, xs] / v_s))

        # - constraints
        nfz_cells = int(env.nofly[ys, xs].sum())
        drone_len = seg.reshape(D, K).sum(axis=1)
        range_excess = float(np.maximum(drone_len - self.R, 0.0).sum() / self.R)

        # --- sensor coverage
        pad, n = self.pad, env.size
        lvmap = np.zeros((n + 2 * pad, n + 2 * pad), dtype=np.int8)
        on_s = on[leg_id]
        lvl_s = lvl[leg_id]
        for lv in range(1, self.levels + 1):  # ascending: highest level wins
            m = on_s & (lvl_s == lv)
            if m.any():
                oy, ox = self.discs[lv]
                lvmap[ys[m][:, None] + pad + oy[None, :], xs[m][:, None] + pad + ox[None, :]] = lv
        lvmap = lvmap[pad:-pad, pad:-pad]

        covered = lvmap > 0
        coverage = float(env.priority[covered].sum() / env.total_priority)
        tl = lvmap[env.targets[:, 1], env.targets[:, 0]]
        det_prob = np.where(tl > 0, 0.5 + 0.1 * tl, 0.0)  # stronger sensor -> better detection
        detection = float((env.target_w * det_prob).sum() / env.target_w.sum())

        # normalised cost terms and raw score
        energy_n = energy / self.energy_ref
        time_n = makespan / self.time_ref
        hazard_n = exposure / w.hazard_ref
        raw = (w.coverage * coverage + w.detection * detection
               - w.energy * energy_n - w.time * time_n - w.hazard * hazard_n)

        m = dict(raw=raw, coverage=coverage, detection=detection, energy=energy,
                 energy_n=energy_n, time=makespan, time_n=time_n, hazard=exposure,
                 hazard_n=hazard_n, nfz_cells=nfz_cells, range_excess=range_excess,
                 drone_len=drone_len, drone_time=drone_time,
                 covered_cells=int(covered.sum()))
        if want_maps:
            m["levelmap"] = lvmap
        return m

    def evaluate(self, ch: Chromosome) -> dict:
        ch.metrics = self.simulate(ch)
        return ch.metrics
