# UAV Strategy Engine - Genetic Algorithm for Disaster-Zone Surveillance

Course: Artificial Intelligence (BSCS 5), Bahria University, Lahore Campus - Assignment 2 (CLO4)

A fleet of **5 UAVs** must survey a volatile **100 x 100 hazard grid**. A **Genetic Algorithm (GA)**
evolves the deployment (launch) positions, waypoint routes, speeds and sensor schedules so that the
fleet balances four competing goals:

| Goal | Direction |
|---|---|
| Priority-weighted area coverage and target detection rate | maximise |
| Battery/energy use, mission time, exposure to hazard zones | minimise |
| Maximum flight range, no-fly zones | hard constraints (repair + dynamic penalty) |

There is no single "correct" answer: the GA returns a good compromise, and the experiments show how
the compromise moves when the weights and hyper-parameters change.

---

## 1. Quick start (PyCharm )



```bash
# 1. (optional but recommended) create a virtual environment - PyCharm does this for you
python -m venv venv
venv\Scripts\activate          # Windows        |   source venv/bin/activate   (Linux / macOS)

# 2. install the two libraries
pip install -r requirements.txt

# 3. run ONE full simulation + all main figures  (about 15 seconds)
python main.py

# 4. run the full experimental study for the report  (about 10-15 minutes)
python experiments.py            # add  --quick  for a 1-minute test version

# 5. (optional) structural comparison used in the report's 'uniqueness' section
python analysis.py

# 6. (optional) run the small test-suite
python tests/test_basic.py
```

All figures are written to `outputs/` (PNG files) together with JSON files holding the raw numbers.

## 2. Project structure

```
uav_ga/
|-- main.py                 run one GA simulation, make the main figures
|-- experiments.py          GA vs random search, sensitivity study, weight profiles, constraint study
|-- config.py               ALL settings (grid, drones, weights, GA hyper-parameters)
|-- environment.py          100x100 grid: priority, time-varying hazards, no-fly zones, targets
|-- chromosome.py           Chromosome class + random plan factory
|-- fitness.py              mission simulation and multi-objective fitness
|-- constraints.py          repair operators + dynamic penalty
|-- operators.py            selection (tournament / rank / roulette), spatial crossover, spatial mutation
|-- genetic_algorithm.py    GA engine: initialisation, evolution loop, elitism, diversity, termination
|-- random_search.py        baseline for the comparison
|-- analysis.py             structural comparison of GA vs random plans (zone coverage, overlap)
|-- visualization.py        all plots
|-- tests/test_basic.py     smoke tests
|-- outputs/                generated figures and results
`-- requirements.txt
```

## 3. How the design maps 

**Chromosome (hybrid integer / real / binary).** For each drone: launch point (x, y) and K = 10
route waypoints (integers, cells of the grid); for each of the 10 legs a speed (real, 0.5-2.0), a
sensor power (real, 0-1, quantised into 4 levels) and a sensor ON/OFF bit (binary activation
schedule). 52 genes per drone, 260 genes in total.

**Fitness (multi-objective, weighted sum of normalised terms).**

```
fitness = 0.40*coverage + 0.30*detection - 0.12*energy - 0.08*time - 0.10*hazard - penalty
```

The mission is simulated: every leg is rasterised into grid cells, the sensor footprint is stamped
on the map (stronger sensor = larger radius, higher detection probability, more energy), and the
hazard map is looked up at the moment the drone is in the cell (volatile zones drift and grow).

**Genetic operators (spatial).**
* Selection: tournament (default, size 3), rank, roulette.
* Crossover: whole-drone swap, *geometric* blend of waypoints (point on the line between the
  parents' waypoints), contiguous segment swap.
* Mutation: Gaussian waypoint shift, priority-biased relocation, segment reversal (2-opt style),
  speed/power jitter, sensor flip.
* Diversity: adaptive mutation boost when the population collapses, random immigrants during
  stagnation, elitism (2 best survive).
* Termination: maximum generations (150) **or** no improvement for 60 generations.

**Constraint handling.**
* Repair: waypoints inside no-fly zones move to the nearest free cell; routes longer than the
  maximum range (220 cells) are truncated.
* Dynamic penalty: every flown cell inside a no-fly zone is penalised with a weight that grows
  from 0.5 to 2.0 over the generations.

## 4. Outputs

| File | Content |
|---|---|
| `01_environment.png` | the scenario map |
| `02_fitness_convergence.png` | **best vs average fitness over generations** |
| `03_flight_paths_evolution.png` | **evolved UAV flight paths at generations 0, 3, 10, 30, 80, 150** |
| `04_coverage_heatmaps_evolution.png` | **coverage heatmaps at the same generations** |
| `05_final_solution.png` | the final evolved plan |
| `06_diversity_and_feasibility.png` | population diversity and constraint satisfaction |
| `07_objectives_over_generations.png` | how each objective of the best plan evolves |
| `08_tradeoff_coverage_vs_energy.png` | multi-objective trade-off (GA population vs random plans) |
| `09_ga_vs_random_search.png` | GA vs random-search baseline (10 seeds) |
| `10_random_plan_distribution.png` | best GA plan vs distribution of random plans |
| `11_hyperparameter_sensitivity.png` | population size, mutation rate, selection pressure |
| `12_weight_profiles.png` | effect of the fitness weights |
| `Technical_Report.docx` | the 5-page technical report |
| `*.json` | raw numbers used in the report |

## 5. Reproducibility

The map is fixed by `EnvConfig.seed` (7) and the GA by `GAConfig.seed`, so every run gives the same
numbers. Change `config.py` to try other fleet sizes, ranges, weights or hyper-parameters.
