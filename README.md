## Ant Colony Optimization for the Traveling Salesman Problem

An implementation of the Ant Colony Optimization to approximate solutions for the Traveling Salesman Problem. 
The solver is implemented in Python, fully written in NumPy, and accelerated using Numba.

## Usage

```bash
python main.py [options]
```

**Options:**

* `--size`, `-n`: Number of cities in the TSP instance
* `--pheromone_influence`: Exponent for pheromone strength
* `--heuristic_influence`: Exponent for heuristic information
* `--evaporation_rate`: Rate at which pheromone evaporates each iteration
* `--initial_pheromone_factor`: Scaling factor for initial pheromone level
* `--ants`: Number of ants per iteration (defaults to number of cities)
* `--iters`: Number of optimization iterations
* `--seed`: Random seed for reproducibility

## Algorithm Details

1. **Initialization**

   * Compute the nearest-neighbor tour length.
   * Set all pheromone values to `initial_pheromone_factor * num_cities / initial_length`.

2. **Iteration Loop**

   * Each ant constructs a tour probabilistically, influenced by pheromone and heuristic matrices.
   * Compute the cost of each tour.
   * Update the global best tour if a better solution is found.
   * Evaporate existing pheromone by factor `(1 - evaporation_rate)`.
   * Deposit pheromone along each ant's tour, proportional to `1 / tour_cost`.

3. **Termination**

   * Return the best tour and its cost.
