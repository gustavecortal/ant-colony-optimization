"""Ant Colony Optimization for the Traveling Salesman Problem"""

from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
from numba import njit, prange


@dataclass
class ACOConfig:
    """Configuration for the Ant Colony Optimization algorithm."""

    size: int = 50
    pheromone_influence: float = 1.0
    heuristic_influence: float = 3.5
    evaporation_rate: float = 0.5
    initial_pheromone_factor: float = 1.0
    ants: Optional[int] = None
    iters: int = 200
    seed: Optional[int] = None


@njit(fastmath=True)
def nearest_neighbor_length(distance_matrix: np.ndarray) -> float:
    """Compute tour length using nearest neighbor heuristic starting at city 0."""
    num_cities = distance_matrix.shape[0]
    current_city = 0
    visited = np.zeros(num_cities, np.uint8)
    visited[current_city] = 1
    total_length = 0.0
    for _ in range(num_cities - 1):
        shortest_distance = 1e18
        next_city = 0
        for city in range(num_cities):
            if visited[city] == 0:
                dist = distance_matrix[current_city, city]
                if dist < shortest_distance:
                    shortest_distance = dist
                    next_city = city
        total_length += shortest_distance
        visited[next_city] = 1
        current_city = next_city
    return total_length + distance_matrix[current_city, 0]


@njit(fastmath=True, parallel=True)
def update_pheromone(
    pheromone_matrix: np.ndarray,
    tours: np.ndarray,
    tour_costs: np.ndarray,
    evaporation_rate: float,
) -> None:
    """Apply pheromone evaporation and deposit based on ant tours and their costs."""
    num_cities = pheromone_matrix.shape[0]
    num_ants = tours.shape[0]
    # Evaporation
    for i in prange(num_cities):
        for j in range(num_cities):
            pheromone_matrix[i, j] *= 1.0 - evaporation_rate

    # Deposit
    for ant_idx in prange(num_ants):
        deposit_amount = 1.0 / tour_costs[ant_idx]
        tour = tours[ant_idx]
        for step in range(num_cities):
            u = tour[step]
            v = tour[step + 1]
            pheromone_matrix[u, v] += deposit_amount
            pheromone_matrix[v, u] += deposit_amount  # symmetric update


@njit(fastmath=True)
def choose_next_city(weights: np.ndarray, total_weight: float) -> int:
    """Select next city index probabilistically based on cumulative weight distribution."""
    threshold = np.random.random() * total_weight
    cumulative = 0.0
    for idx in range(weights.shape[0]):
        cumulative += weights[idx]
        if cumulative >= threshold:
            return idx
    # fallback to any feasible city
    for idx in range(weights.shape[0]):
        if weights[idx] > 0:
            return idx
    return 0


@njit(fastmath=True, parallel=True)
def construct_tour(
    distance_matrix: np.ndarray,
    pheromone_matrix: np.ndarray,
    heuristic_matrix: np.ndarray,
    start_city: int,
    weight_buffer: np.ndarray,
    pheromone_influence: float,
    heuristic_influence: float,
) -> np.ndarray:
    """Construct a full TSP tour for one ant using pheromone and heuristic values."""
    num_cities = distance_matrix.shape[0]
    tour = np.empty(num_cities + 1, np.int64)
    tour[0] = start_city
    visited = np.zeros(num_cities, np.uint8)
    visited[start_city] = 1
    current_city = start_city
    for step in range(1, num_cities):
        total_weight = 0.0
        for city in range(num_cities):
            if visited[city] == 0:
                weight = (
                    pheromone_matrix[current_city, city] ** pheromone_influence
                ) * (heuristic_matrix[current_city, city] ** heuristic_influence)
                weight_buffer[city] = weight
                total_weight += weight
            else:
                weight_buffer[city] = 0.0
        next_city = choose_next_city(weight_buffer, total_weight)
        tour[step] = next_city
        visited[next_city] = 1
        current_city = next_city
    tour[num_cities] = start_city
    return tour


@njit(fastmath=True)
def calculate_tour_cost(distance_matrix: np.ndarray, tour: np.ndarray) -> float:
    """Calculate total distance of a given TSP tour based on the distance matrix."""
    cost = 0.0
    for idx in range(tour.shape[0] - 1):
        cost += distance_matrix[tour[idx], tour[idx + 1]]
    return float(cost)


def aco_solver(
    distance_matrix: np.ndarray, config: ACOConfig
) -> Tuple[np.ndarray, float]:
    """Perform ant colony optimization for specified iterations and return best tour and its cost."""
    if config.seed is not None:
        np.random.seed(config.seed)

    matrix = distance_matrix.astype(np.float64)
    num_cities = matrix.shape[0]
    num_ants = config.ants if config.ants is not None else num_cities

    # Build heuristic matrix
    heuristic_matrix = np.zeros_like(matrix)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j and matrix[i, j] > 0:
                heuristic_matrix[i, j] = 1.0 / matrix[i, j]

    # Initial pheromone
    initial_length = nearest_neighbor_length(matrix)
    initial_pheromone = config.initial_pheromone_factor * num_cities / initial_length
    pheromone_matrix = np.full((num_cities, num_cities), initial_pheromone, np.float64)

    # Buffers
    weight_buffer = np.empty(num_cities, np.float64)
    tours = np.empty((num_ants, num_cities + 1), np.int64)
    tour_costs = np.empty(num_ants, np.float64)

    # Initialize best
    best_tour = np.arange(num_cities + 1, dtype=np.int64)
    best_tour[:-1] = np.arange(num_cities)
    best_tour[-1] = 0
    best_cost = np.inf

    # Main loop
    for _ in range(config.iters):
        for ant in range(num_ants):
            start_city = np.random.randint(0, num_cities)
            tours[ant] = construct_tour(
                matrix,
                pheromone_matrix,
                heuristic_matrix,
                start_city,
                weight_buffer,
                config.pheromone_influence,
                config.heuristic_influence,
            )
            tour_costs[ant] = calculate_tour_cost(matrix, tours[ant])

        # Update best
        best_idx = np.argmin(tour_costs)
        if tour_costs[best_idx] < best_cost:
            best_cost = tour_costs[best_idx]
            best_tour = tours[best_idx].copy()

        # Pheromone update
        update_pheromone(pheromone_matrix, tours, tour_costs, config.evaporation_rate)

    return best_tour, best_cost
