"""Ant Colony Optimization for the Traveling Salesman Problem."""

import argparse
import numpy as np
from solver import aco_solver, ACOConfig


def main():
    """Main function to run the ACO TSP solver with command line arguments."""
    parser = argparse.ArgumentParser(
        prog="aco-tsp",
        description="Run the ACO TSP solver with configurable parameters.",
    )
    parser.add_argument(
        "--size",
        "-n",
        type=int,
        default=50,
        help="Number of cities.",
    )
    parser.add_argument(
        "--pheromone_influence",
        type=float,
        default=1.0,
        help="Pheromone influence exponent.",
    )
    parser.add_argument(
        "--heuristic_influence",
        type=float,
        default=3.5,
        help="Heuristic influence exponent.",
    )
    parser.add_argument(
        "--evaporation_rate",
        type=float,
        default=0.5,
        help="Pheromone evaporation rate.",
    )
    parser.add_argument(
        "--initial_pheromone_factor",
        type=float,
        default=1.0,
        help="Initial pheromone scaling factor.",
    )
    parser.add_argument(
        "--ants",
        type=int,
        default=None,
        help="Number of ants per iteration (defaults to one per city).",
    )
    parser.add_argument(
        "--iters", type=int, default=200, help="Number of optimization iterations."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    cfg = ACOConfig(
        size=args.size,
        pheromone_influence=args.pheromone_influence,
        heuristic_influence=args.heuristic_influence,
        evaporation_rate=args.evaporation_rate,
        initial_pheromone_factor=args.initial_pheromone_factor,
        ants=args.ants,
        iters=args.iters,
        seed=args.seed,
    )

    coords = np.random.rand(cfg.size, 2)
    distance_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    best_tour, best_cost = aco_solver(distance_matrix, cfg)
    print("Best cost:", best_cost)
    print("Best tour:", best_tour)


if __name__ == "__main__":
    main()
