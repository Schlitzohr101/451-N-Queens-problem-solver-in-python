#!/usr/bin/env python3
"""
PA2: N-Queens with Simulated Annealing (SA) & Genetic Algorithm (GA)
Starter Code with TODOs for students to fill in.

State representation: board[c] = r  (column c has a queen at row r), 0-indexed.
Goal: zero attacking pairs (no row or diagonal conflicts).

Usage examples (after completing TODOs):
  python PA2_NQueens_GA_SA_Starter.py --alg sa --n 8 --T0 5.0 --alpha 0.995 --max_steps 20000 --restarts 5 --seed 42
  python PA2_NQueens_GA_SA_Starter.py --alg ga --n 8 --pop 200 --gens 500 --mut 0.12 --tour 3 --elite 5 --seed 123

What you must implement:
  • conflict counting (energy)
  • a neighbor generator for SA
  • SA acceptance and cooling loop
  • GA: selection (tournament), crossover (1-point), mutation, and main evolution loop
  • keep simple stats for both algorithms (iterations, time, success)

You may add helper functions, printouts, and logging as needed. Use only Python stdlib.
"""
from __future__ import annotations
import argparse
import math
import random
import time
from typing import List, Tuple, Dict, Optional

Board = List[int]

# ----------------------------
# Pretty print utility (provided)
# ----------------------------

def print_board(board: Board) -> None:
    """Pretty-print board with 'Q' and '.' (row-major)."""
    n = len(board)
    for r in range(n):
        row = [('Q' if board[c] == r else '.') for c in range(n)]
        print(' '.join(row))
    print()

# ----------------------------
# Core utilities (SOME TODOs)
# ----------------------------

def random_board(n: int) -> Board:
    """Return a random N-Queens board: one queen per column at random rows."""
    return [random.randrange(n) for _ in range(n)]


def conflicts(board: Board) -> int:
    """TODO: Return the number of attacking pairs on the board.

    Two queens attack if they share a row, or a diagonal.
    Hint: count, then sum combinations k choose 2 for any row/diag bucket with k>1.
    Example buckets:
      row_counts[r] increments for each queen in row r
      diag1_counts[r - c]
      diag2_counts[r + c]
    """
    # TODO: implement conflict counting
    raise NotImplementedError("TODO: implement conflicts(board)")


def max_pairs(n: int) -> int:
    """Max number of distinct pairs among N queens (used as a fitness baseline)."""
    return n * (n - 1) // 2


def is_goal(board: Board) -> bool:
    """A goal board has zero conflicts."""
    return conflicts(board) == 0

# ----------------------------
# Simulated Annealing (SA) -- TODOs in neighbor + loop
# ----------------------------

def sa_neighbor(board: Board) -> Board:
    """TODO: Move one random column's queen to a different random row; return new board.
    Requirements:
      • Choose a random column c
      • Choose a new row r' != current row
      • Return a copy of the board with that single change
    """
    # TODO: implement neighbor generator
    raise NotImplementedError("TODO: implement sa_neighbor(board)")


def simulated_annealing(
    n: int = 8,
    T0: float = 5.0,
    alpha: float = 0.995,
    max_steps: int = 20000,
    restarts: int = 3,
    seed: Optional[int] = None
) -> Tuple[Optional[Board], Dict]:
    """Simulated Annealing with geometric cooling. Energy = conflicts(board).

    TODOs:
      • Initialize random seed if provided
      • For each (re)start: start from a random board
      • At each step, sample a neighbor, compute ΔE = E_new - E_cur
      • Accept always if ΔE <= 0; otherwise accept with probability exp(-ΔE / T)
      • Cool temperature T <- alpha * T each step
      • Track best-ever board/energy; stop early if energy==0
      • Return solution (or None) and stats dict
    """
    # Recommended: keep these stats
    # steps_total = 0
    # best_overall, best_energy_overall = None, math.inf

    # TODO: implement SA main loop
    raise NotImplementedError("TODO: implement simulated_annealing(...)")

# ----------------------------
# Genetic Algorithm (GA) -- TODOs in selection, crossover, mutation, loop
# ----------------------------

def fitness(board: Board) -> int:
    """Higher is better: max_pairs(n) - conflicts(board)."""
    return max_pairs(len(board)) - conflicts(board)


def tournament_select(pop: List[Board], k: int) -> Board:
    """TODO: k-way tournament; return the fittest among k random individuals.
    Requirements:
      • Randomly sample k individuals from pop
      • Return a COPY of the fittest (highest fitness)
    """
    # TODO: implement tournament selection
    raise NotImplementedError("TODO: implement tournament_select(pop, k)")


def one_point_crossover(p1: Board, p2: Board) -> Tuple[Board, Board]:
    """TODO: One-point crossover on column index; return two children.
    Requirements:
      • Pick cut in [1, n-1]
      • c1 = p1[:cut] + p2[cut:]
      • c2 = p2[:cut] + p1[cut:]
    """
    # TODO: implement crossover
    raise NotImplementedError("TODO: implement one_point_crossover(p1, p2)")


def mutate(board: Board, pmut: float) -> None:
    """TODO: With probability pmut per column, move the queen to a random row.
    Requirements:
      • For each column c, with prob pmut set board[c] to a random row (0..n-1)
      • In-place mutation (no return)
    """
    # TODO: implement mutation
    raise NotImplementedError("TODO: implement mutate(board, pmut)")


def ga_solve(
    n: int = 8,
    pop_size: int = 200,
    generations: int = 500,
    mutation: float = 0.10,
    tournament_k: int = 3,
    elitism: int = 5,
    seed: Optional[int] = None
) -> Tuple[Optional[Board], Dict]:
    """Genetic Algorithm for N-Queens.

    TODOs:
      • Initialize seed if provided
      • Create initial population (random boards)
      • Sort by fitness descending each generation
      • Elitism: copy top E into new population
      • While new population < pop_size: select parents via tournament, crossover, mutate
      • Track best board; stop early if is_goal(best)
      • Return (solution_or_None, stats)
    """
    # TODO: implement GA main loop
    raise NotImplementedError("TODO: implement ga_solve(...)")

# ----------------------------
# CLI (provided) — you may extend for logging/plots
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="PA2: N-Queens via Simulated Annealing or Genetic Algorithm")
    ap.add_argument("--alg", choices=["sa", "ga"], required=True, help="Algorithm to run")
    ap.add_argument("--n", type=int, default=8, help="Board size N (default 8)")

    # SA params
    ap.add_argument("--T0", type=float, default=5.0, help="Initial temperature (SA)")
    ap.add_argument("--alpha", type=float, default=0.995, help="Cooling factor (SA)")
    ap.add_argument("--max_steps", type=int, default=20000, help="Max steps per run (SA)")
    ap.add_argument("--restarts", type=int, default=3, help="Random restarts (SA)")

    # GA params
    ap.add_argument("--pop", type=int, default=200, help="Population size (GA)")
    ap.add_argument("--gens", type=int, default=500, help="Generations (GA)")
    ap.add_argument("--mut", type=float, default=0.10, help="Mutation probability per gene (GA)")
    ap.add_argument("--tour", type=int, default=3, help="Tournament size (GA)")
    ap.add_argument("--elite", type=int, default=5, help="Elitism count (GA)")

    # Misc
    ap.add_argument("--seed", type=int, default=None, help="Random seed")

    args = ap.parse_args()

    if args.alg == "sa":
        sol, stats = simulated_annealing(
            n=args.n,
            T0=args.T0,
            alpha=args.alpha,
            max_steps=args.max_steps,
            restarts=args.restarts,
            seed=args.seed,
        )
    else:
        sol, stats = ga_solve(
            n=args.n,
            pop_size=args.pop,
            generations=args.gens,
            mutation=args.mut,
            tournament_k=args.tour,
            elitism=args.elite,
            seed=args.seed,
        )

    print("Stats:", stats)
    if sol is not None:
        print("\nSolution board:")
        print_board(sol)
    else:
        print("\nNo solution found (try different params or increase limits).")


if __name__ == "__main__":
    main()
