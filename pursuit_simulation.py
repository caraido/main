"""Mutual pursuit simulation for four cars.

This module simulates four cars starting at the corners of a square and each pursuing
the next car in a clockwise order. The traces converge to the center, forming
logarithmic spirals. Run this module as a script to visualize the pursuit curves.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PursuitResult:
    traces: np.ndarray  # shape: (steps, 4, 2)
    elapsed_hours: float


def simulate_pursuit(
    side_length_miles: float = 1.0,
    speed_mph: float = 60.0,
    time_step_seconds: float = 0.1,
    max_seconds: float = 600.0,
    tolerance_miles: float = 1e-5,
) -> PursuitResult:
    """Simulate four cars in mutual pursuit around a square.

    Args:
        side_length_miles: Length of a side of the initial square.
        speed_mph: Constant speed of each car.
        time_step_seconds: Integration step size in seconds.
        max_seconds: Hard cutoff on simulation time to avoid infinite loops.
        tolerance_miles: Stop once all cars are this close (or closer) to one another.

    Returns:
        PursuitResult containing a record of all positions and elapsed time in hours.
    """
    if side_length_miles <= 0:
        raise ValueError("side_length_miles must be positive")
    if speed_mph <= 0:
        raise ValueError("speed_mph must be positive")
    if time_step_seconds <= 0:
        raise ValueError("time_step_seconds must be positive")

    # Clockwise corner ordering: (0, 0), (L, 0), (L, L), (0, L)
    positions = np.array(
        [
            [0.0, 0.0],
            [side_length_miles, 0.0],
            [side_length_miles, side_length_miles],
            [0.0, side_length_miles],
        ],
        dtype=float,
    )

    dt_hours = time_step_seconds / 3600.0
    max_steps = int(np.ceil(max_seconds / time_step_seconds))
    history = [positions.copy()]

    for _ in range(max_steps):
        targets = np.roll(positions, -1, axis=0)
        pursuit_vectors = targets - positions
        distances = np.linalg.norm(pursuit_vectors, axis=1, keepdims=True)
        directions = np.divide(
            pursuit_vectors,
            distances,
            out=np.zeros_like(pursuit_vectors),
            where=distances > 1e-12,
        )

        positions = positions + speed_mph * dt_hours * directions
        history.append(positions.copy())

        next_distances = np.linalg.norm(np.roll(positions, -1, axis=0) - positions, axis=1)
        if np.max(next_distances) < tolerance_miles:
            break

    traces = np.stack(history)
    elapsed_hours = (traces.shape[0] - 1) * dt_hours
    return PursuitResult(traces=traces, elapsed_hours=elapsed_hours)


def plot_traces(result: PursuitResult) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the pursuit curves for the four cars."""
    traces = result.traces
    fig, ax = plt.subplots(figsize=(6, 6))

    for idx in range(4):
        car_trace = traces[:, idx, :]
        ax.plot(car_trace[:, 0], car_trace[:, 1], label=f"Car {idx + 1}")
        ax.scatter(car_trace[0, 0], car_trace[0, 1], marker="o", s=40, color=ax.lines[-1].get_color())
        ax.scatter(car_trace[-1, 0], car_trace[-1, 1], marker="x", s=60, color=ax.lines[-1].get_color())

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Miles (east)")
    ax.set_ylabel("Miles (north)")
    ax.set_title("Mutual Pursuit of Four Cars")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate four cars in mutual pursuit around a square.")
    parser.add_argument("--side", type=float, default=1.0, help="Side length of the square in miles (default: 1.0)")
    parser.add_argument("--speed", type=float, default=60.0, help="Speed of each car in mph (default: 60.0)")
    parser.add_argument(  
        "--dt",
        type=float,
        default=0.1,
        help="Time step for the integrator in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=600.0,
        help="Maximum simulated time in seconds (default: 600)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Stop when cars are within this many miles of one another (default: 1e-5)",
    )

    args = parser.parse_args()
    result = simulate_pursuit(
        side_length_miles=args.side,
        speed_mph=args.speed,
        time_step_seconds=args.dt,
        max_seconds=args.max_seconds,
        tolerance_miles=args.tolerance,
    )

    minutes = result.elapsed_hours * 60
    print(f"Simulation finished after {minutes:.2f} minutes with {result.traces.shape[0]} steps.")
    plot_traces(result)
    plt.show()


if __name__ == "__main__":
    main()
