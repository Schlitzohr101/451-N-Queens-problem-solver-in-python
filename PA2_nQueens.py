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
    
    row_conflicts = 0
    diag_conflicts = 0
    #row check 
    for i in range(len(board)):
        if (i<len(board)-1) and board[i] in board[i+1:]:
            row_conflicts+=1
    
         
    #diag check...
    for c in range(len(board)):
        cur_row = board[c]
        
        for i in range(c+1, len(board)):
            #count up_diag_conflicts
            if (cur_row - (i-c) >= 0) and (board[i] == cur_row-(i-c)):
                diag_conflicts+= 1
            #count down_diag_conflicts
            if (cur_row - (i-c) >= 0) and (board[i] == cur_row+(i-c)):
                diag_conflicts+= 1
        
         
    return row_conflicts + diag_conflicts
    

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
    
    new_board = board.copy()
    
    rand_row = random.randrange(0,len(board))
    
    rand_col = random.randrange(0,len(board))
    
    new_board[rand_col] = rand_row
    
    return new_board
     


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
    
    #gen seed
    if seed is not None:
        random.seed(seed)
        
    steps_total = 0
    best_board_overall = None
    best_energy_overall = math.inf
    
    start_time = time.time()
    
    #count for restarts
    for restart in range(restarts):
        current_board = random_board(n)
        current_energy = conflicts(current_board)
        
        best_board = current_board #this never really gets used...
        best_energy = current_energy
        
        T = T0
        
        for step in range(max_steps): #step cap
            steps_total += 1
            
            neighbor_board = sa_neighbor(current_board)
            neighbor_energy = conflicts(neighbor_board)
            
            delta_E = neighbor_energy - current_energy
            
            if delta_E <= 0:
                current_board = neighbor_board
                current_energy = neighbor_energy
            
            else :
                accept_prob = math.exp(-delta_E / T) #should be something like 0.####
                if (random.random() < accept_prob): # if within that prob...
                    current_board = neighbor_board
                    current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_board = current_board
                
            if current_energy < best_energy_overall:
                best_energy_overall = current_energy
                best_board_overall = current_board
                
            if current_energy == 0: #found sol
                break
            
            T = alpha * T
        
        if best_energy_overall == 0: #found sol
            break
        
        
    
    # Prepare statistics
    stats = {
        'steps_total': steps_total,
        'restarts_used': restart + 1,
        'best_energy': best_energy_overall,
        'elapsed_time': time.time() - start_time,
        'success': best_energy_overall == 0
    }
    
    # Return solution (or None if not found) and stats
    return (best_board_overall if best_energy_overall == 0 else None), stats    
            

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
    
    samples = []
    
    for i in range(k):
        cur_choice = random.choice(pop)
        while cur_choice in samples:
            cur_choice = random.choice(pop)
        samples.append(cur_choice.copy())
        
    most_fit = None
    most_fit_score = 0
    for choice in range(len(samples)):
        cur_fit_score = fitness(samples[choice])
        if cur_fit_score > most_fit_score:
            most_fit = samples[choice]
    
    return most_fit
        


def one_point_crossover(p1: Board, p2: Board) -> Tuple[Board, Board]:
    """TODO: One-point crossover on column index; return two children.
    Requirements:
      • Pick cut in [1, n-1]
      • c1 = p1[:cut] + p2[cut:]
      • c2 = p2[:cut] + p1[cut:]
      
      given two boards, crossover at the random point for every col and give back the new boards...
    """
    
    b1 = p1.copy()
    b2 = p2.copy()
    
    pivot = random.randrange(1, len(p1))
    
    r1 = b1[:pivot] +  b2[pivot:]
    r2 = b2[:pivot] + b1[pivot:]
    
    return (r1,r2)
    

def mutate(board: Board, pmut: float) -> None:
    """TODO: With probability pmut per column, move the queen to a random row.
    Requirements:
      • For each column c, with prob pmut set board[c] to a random row (0..n-1)
      • In-place mutation (no return)
    """
    
    for c in range(len(board)):
        if (random.random() < pmut):
            board[c] = random.randint(0, len(board)-1)
            


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
    
    if seed is not None:
        random.seed(seed)
        
    start_time = time.time()
    
    #init pop
    pop = [random_board(n) for _ in range(pop_size)]
    
    
    best_board = None
    best_fitness = -1
    
    for generation_count in range(generations):
        board_and_fit_score_list = []
        for board in range(len(pop)):
            board_and_fit_score_list.append((pop[board], fitness(pop[board])))
            
            #we set x[1] here to x because what we want to sort with
        board_and_fit_score_list.sort(key=lambda x: x[1], reverse=True)
         
        current_best_board = board_and_fit_score_list[0][0]
        current_best_score = board_and_fit_score_list[0][1]
        
        if current_best_score > best_fitness:
            best_fitness = current_best_score
            best_board = current_best_board
            
        if is_goal(current_best_board):
            best_board = current_best_board
            break
        
        #elitism implement
        new_pop = []
        for i in range(elitism):
            new_pop.append(board_and_fit_score_list[i][0])
            
        while len(new_pop) < pop_size:
            parent_1 = tournament_select(pop, tournament_k)
            parent_2 = tournament_select(pop, tournament_k)
            
            child_1, child_2 = one_point_crossover(parent_1,parent_2)
            
            mutate(child_1, mutation)
            mutate(child_2, mutation)
            
            new_pop.append(child_1)
            if len(new_pop) < pop_size:
                new_pop.append(child_2)
        
        pop = new_pop
        
    stats = {
        'generations': generation_count + 1,
        'best_fitness': best_fitness,
        'max_fitness': max_pairs(n),
        'elapsed_time': time.time() - start_time,
        'success': best_board is not None and is_goal(best_board)
    }
    
    return (best_board if best_board is not None and is_goal(best_board) else None), stats
    
    
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

# ----------------------------
# Testing Framework
# ----------------------------

def run_experiments(n: int, num_runs: int = 10):
    """Run comprehensive experiments for both GA and SA."""
    
    print("="*80)
    print(f"N-QUEENS SOLVER EXPERIMENTS (N={n}, Runs per config={num_runs})")
    print("="*80)
    
    # ========== SIMULATED ANNEALING EXPERIMENTS ==========
    print("\n" + "="*80)
    print("SIMULATED ANNEALING EXPERIMENTS")
    print("="*80)
    
    sa_configs = [
        # Baseline
        {"T0": 5.0, "alpha": 0.995, "max_steps": 20000},
        # Vary T0
        {"T0": 1.0, "alpha": 0.995, "max_steps": 20000},
        {"T0": 10.0, "alpha": 0.995, "max_steps": 20000},
        # Vary alpha (cooling schedule)
        {"T0": 5.0, "alpha": 0.99, "max_steps": 20000},
        {"T0": 5.0, "alpha": 0.999, "max_steps": 20000},
        # Vary max_steps (steps_per_temp implicitly)
        {"T0": 5.0, "alpha": 0.995, "max_steps": 10000},
        {"T0": 5.0, "alpha": 0.995, "max_steps": 40000},
    ]
    
    for config_idx, config in enumerate(sa_configs, 1):
        print(f"\n--- SA Config {config_idx}: T0={config['T0']}, alpha={config['alpha']}, max_steps={config['max_steps']} ---")
        
        results = []
        for run in range(num_runs):
            seed = random.randint(0, 100000)
            sol, stats = simulated_annealing(
                n=n,
                T0=config['T0'],
                alpha=config['alpha'],
                max_steps=config['max_steps'],
                restarts=5,
                seed=seed
            )
            results.append(stats)
        
        # Calculate statistics
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / num_runs * 100
        
        times = [r['elapsed_time'] for r in results]
        avg_time = sum(times) / len(times)
        best_time = min(times)
        
        steps = [r['steps_total'] for r in results]
        avg_steps = sum(steps) / len(steps)
        best_steps = min(steps)
        
        successful_times = [r['elapsed_time'] for r in results if r['success']]
        successful_steps = [r['steps_total'] for r in results if r['success']]
        
        print(f"  Success Rate: {success_rate:.1f}% ({success_count}/{num_runs})")
        print(f"  Avg Time: {avg_time:.4f}s | Best Time: {best_time:.4f}s")
        print(f"  Avg Steps: {avg_steps:.0f} | Best Steps: {best_steps}")
        
        if successful_times:
            print(f"  Successful runs - Avg Time: {sum(successful_times)/len(successful_times):.4f}s | Avg Steps: {sum(successful_steps)/len(successful_steps):.0f}")
    
    # ========== GENETIC ALGORITHM EXPERIMENTS ==========
    print("\n" + "="*80)
    print("GENETIC ALGORITHM EXPERIMENTS")
    print("="*80)
    
    ga_configs = [
        # Baseline
        {"pop": 200, "mut": 0.10, "tour": 3, "elite": 5},
        # Vary population
        {"pop": 100, "mut": 0.10, "tour": 3, "elite": 5},
        {"pop": 400, "mut": 0.10, "tour": 3, "elite": 5},
        # Vary mutation
        {"pop": 200, "mut": 0.05, "tour": 3, "elite": 5},
        {"pop": 200, "mut": 0.20, "tour": 3, "elite": 5},
        # Vary tournament_k
        {"pop": 200, "mut": 0.10, "tour": 2, "elite": 5},
        {"pop": 200, "mut": 0.10, "tour": 5, "elite": 5},
        # Vary elitism
        {"pop": 200, "mut": 0.10, "tour": 3, "elite": 2},
        {"pop": 200, "mut": 0.10, "tour": 3, "elite": 10},
    ]
    
    for config_idx, config in enumerate(ga_configs, 1):
        print(f"\n--- GA Config {config_idx}: pop={config['pop']}, mut={config['mut']}, tour_k={config['tour']}, elite={config['elite']} ---")
        
        results = []
        for run in range(num_runs):
            seed = random.randint(0, 100000)
            sol, stats = ga_solve(
                n=n,
                pop_size=config['pop'],
                generations=500,
                mutation=config['mut'],
                tournament_k=config['tour'],
                elitism=config['elite'],
                seed=seed
            )
            results.append(stats)
        
        # Calculate statistics
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / num_runs * 100
        
        times = [r['elapsed_time'] for r in results]
        avg_time = sum(times) / len(times)
        best_time = min(times)
        
        gens = [r['generations'] for r in results]
        avg_gens = sum(gens) / len(gens)
        best_gens = min(gens)
        
        successful_times = [r['elapsed_time'] for r in results if r['success']]
        successful_gens = [r['generations'] for r in results if r['success']]
        
        print(f"  Success Rate: {success_rate:.1f}% ({success_count}/{num_runs})")
        print(f"  Avg Time: {avg_time:.4f}s | Best Time: {best_time:.4f}s")
        print(f"  Avg Generations: {avg_gens:.0f} | Best Generations: {best_gens}")
        
        if successful_times:
            print(f"  Successful runs - Avg Time: {sum(successful_times)/len(successful_times):.4f}s | Avg Gens: {sum(successful_gens)/len(successful_gens):.0f}")
    
    # ========== COMPARISON: GA vs SA (N=8) ==========
    print("\n" + "="*80)
    print("COMPARISON: GA vs SA (N=8, Baseline Configs)")
    print("="*80)
    
    print("\n--- Running SA (baseline) ---")
    sa_results = []
    for run in range(num_runs):
        seed = random.randint(0, 100000)
        sol, stats = simulated_annealing(n=8, T0=5.0, alpha=0.995, max_steps=20000, restarts=5, seed=seed)
        sa_results.append(stats)
    
    sa_success = sum(1 for r in sa_results if r['success']) / num_runs * 100
    sa_avg_time = sum(r['elapsed_time'] for r in sa_results) / num_runs
    sa_best_time = min(r['elapsed_time'] for r in sa_results)
    
    print(f"SA: Success={sa_success:.1f}%, Avg Time={sa_avg_time:.4f}s, Best Time={sa_best_time:.4f}s")
    
    print("\n--- Running GA (baseline) ---")
    ga_results = []
    for run in range(num_runs):
        seed = random.randint(0, 100000)
        sol, stats = ga_solve(n=8, pop_size=200, generations=500, mutation=0.10, tournament_k=3, elitism=5, seed=seed)
        ga_results.append(stats)
    
    ga_success = sum(1 for r in ga_results if r['success']) / num_runs * 100
    ga_avg_time = sum(r['elapsed_time'] for r in ga_results) / num_runs
    ga_best_time = min(r['elapsed_time'] for r in ga_results)
    
    print(f"GA: Success={ga_success:.1f}%, Avg Time={ga_avg_time:.4f}s, Best Time={ga_best_time:.4f}s")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"For N={n}:")
    print(f"  SA is {'more' if sa_success > ga_success else 'less'} reliable (success rate: {sa_success:.1f}% vs {ga_success:.1f}%)")
    print(f"  SA is {'faster' if sa_avg_time < ga_avg_time else 'slower'} on average ({sa_avg_time:.4f}s vs {ga_avg_time:.4f}s)")


def test_scalability():
    """Test scalability to N=10."""
    print("\n" + "="*80)
    print("SCALABILITY TEST: N=10")
    print("="*80)
    
    num_runs = 10
    n = 10
    
    print("\n--- SA (N=10) ---")
    sa_results = []
    for run in range(num_runs):
        seed = random.randint(0, 100000)
        sol, stats = simulated_annealing(n=n, T0=10.0, alpha=0.995, max_steps=50000, restarts=5, seed=seed)
        sa_results.append(stats)
    
    sa_success = sum(1 for r in sa_results if r['success']) / num_runs * 100
    sa_avg_time = sum(r['elapsed_time'] for r in sa_results) / num_runs
    print(f"Success Rate: {sa_success:.1f}%, Avg Time: {sa_avg_time:.4f}s")
    
    print("\n--- GA (N=10) ---")
    ga_results = []
    for run in range(num_runs):
        seed = random.randint(0, 100000)
        sol, stats = ga_solve(n=n, pop_size=300, generations=1000, mutation=0.10, tournament_k=3, elitism=10, seed=seed)
        ga_results.append(stats)
    
    ga_success = sum(1 for r in ga_results if r['success']) / num_runs * 100
    ga_avg_time = sum(r['elapsed_time'] for r in ga_results) / num_runs
    print(f"Success Rate: {ga_success:.1f}%, Avg Time: {ga_avg_time:.4f}s")
    
    print("\nObservation: As N increases, both runtime and reliability are affected.")
    print("SA may need higher T0 and more steps. GA may need larger population and more generations.")

    
        
if __name__ == "__main__":
    # main()
    # running SA and getting values
    run_experiments(n=8, num_runs=10)
    
    # Test scalability to N=10
    test_scalability()
    
    print("\n" + "="*80)
    print("GUIDING QUESTIONS - ANALYSIS")
    print("="*80)
    print("\n1. How do GA and SA compare on success rate and time for N=8?")
    print("   See the 'COMPARISON' section above for direct comparison.")
    print("\n2. Which GA parameters most affect performance? Why?")
    print("   - Population size: Larger populations explore more but take longer per generation.")
    print("   - Mutation rate: Too low = premature convergence; too high = random search.")
    print("   - Tournament k: Higher k = more selection pressure, faster convergence.")
    print("   - Elitism: Preserves best solutions, improves reliability.")
    print("\n3. How does the cooling schedule (alpha) trade off quality vs. time?")
    print("   - Higher alpha (slower cooling): More exploration, better quality, longer time.")
    print("   - Lower alpha (faster cooling): Less exploration, faster, may miss solutions.")
    print("="*80)
    