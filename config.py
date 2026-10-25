"""
config.py
All tunable settings of the UAV Strategy Engine live .  Every group of settings is a small
dataclass,  experiments create variants with ``dataclasses.replace``.
"""
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Mission space (the 'grid') and fleet size."""
    grid_size: int = 100          # 100 x 100 cells
    n_drones: int = 5             # fleet size
    n_waypoints: int = 10         # waypoints (legs) per drone, after the launch point
    deploy_band: int = 10         # drones may launch only from rows 0..9 (staging strip)
    max_range: float = 220.0      # hard limit: max path length per drone (cells)
    n_targets: int = 45           # hidden targets (e.g. trapped survivors)
    n_snapshots: int = 6          # time slices used for the volatile hazard zones
    background_priority: float = 0.2   # value of ordinary (non-disaster) cells
    seed: int = 7                 # fixed seed -> the same map on every run


@dataclass
class DroneConfig:
    # Physical model of one UAV
    speed_min: float = 0.5        # cells per time unit
    speed_max: float = 2.0
    nominal_speed: float = 1.0
    sensor_levels: int = 4        # sensor power is quantised into 4 levels
    # flight energy per cell = move_c0 / v + move_c1 * v^2  (hover + drag trade-off)
    move_c0: float = 0.5
    move_c1: float = 0.5
    # sensor energy per scanned cell = (sensor_s0 + sensor_s1 * level^2) / v
    sensor_s0: float = 0.15
    sensor_s1: float = 0.05


@dataclass
class FitnessWeights:
    # Weights of the multi-objective fitness (positives first, then costs)
    coverage: float = 0.40        # priority-weighted area coverage      (maximise)
    detection: float = 0.30       # expected target detection rate       (maximise)
    energy: float = 0.12          # total battery used                    (minimise)
    time: float = 0.08            # mission time of the slowest drone     (minimise)
    hazard: float = 0.10          # exposure to hazard zones              (minimise)
    hazard_ref: float = 20.0      # normaliser for hazard exposure


@dataclass
class ConstraintConfig:
    #Hard-constraint handling.
    range_mode: str = "repair"    # 'repair' (truncate route) or 'penalty'
    lambda_start: float = 0.5     # no-fly penalty weight at generation 0
    lambda_end: float = 2.0       # growing linearly to this value (dynamic penalty)
    nfz_scale: float = 40.0       # no-fly cells that cost one penalty unit
    range_penalty: float = 4.0    # weight of range excess when range_mode == 'penalty'


@dataclass
class GAConfig:
    #Genetic-algorithm hyper-parameters.
    pop_size: int = 80
    generations: int = 150
    crossover_rate: float = 0.90
    mutation_rate: float = 0.10   # per-gene probability (operators.py)
    tournament_size: int = 3      # selection pressure
    selection: str = "tournament" # 'tournament' | 'rank' | 'roulette'
    elite_count: int = 2
    patience: int = 60            # stop if no improvement for this many generations (None = off)
    min_improvement: float = 1e-4
    immigrant_fraction: float = 0.10   # random immigrants injected when stagnating
    stagnation_trigger: int = 10       # generations without improvement before immigrants
    adaptive_mutation: bool = True     # boost mutation when diversity collapses
    diversity_threshold: float = 6.0   # mean waypoint std (cells) below which we boost
    mutation_boost: float = 2.5
    mutation_sigma: float = 6.0   # std of Gaussian waypoint shift (cells)
    seed: int = 1


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    weights: FitnessWeights = field(default_factory=FitnessWeights)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    ga: GAConfig = field(default_factory=GAConfig)
