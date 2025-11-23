import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import time
from config import config

def build_simple_master():
    """Build simplified Benders master problem - only first stage decisions."""
    m = pyo.ConcreteModel()
    
    # Sets
    m.T = pyo.RangeSet(1, config.T1)  # First stage time periods only
    
    # Variables - first stage decisions
    m.q = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
    m.V = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
    m.s = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, None))  # spillage
    
    # Alpha represents expected future profit from second stage
    m.alpha = pyo.Var(bounds=(-1e6, 5e5))
    
    # Parameters for Benders cuts (initially empty)
    m.Cut = pyo.Set(initialize=[])  # Will be populated with cut indices
    m.Phi = pyo.Param(m.Cut, initialize={}, mutable=True)  # RHS of cuts
    m.Lambda = pyo.Param(m.Cut, initialize={}, mutable=True)  # Coefficient of V24 in cuts
    m.V24_hat = pyo.Param(m.Cut, initialize={}, mutable=True)  # V24 value when cut was generated
    
    # Reservoir balance constraints (first stage only)
    def res_balance(m, t):
        if t == 1:
            return m.V[t] == config.V0 + config.alpha * config.certain_inflow - config.alpha * m.q[t] - m.s[t]
        else:
            return m.V[t] == m.V[t-1] + config.alpha * config.certain_inflow - config.alpha * m.q[t] - m.s[t]
    
    m.res_balance = pyo.Constraint(m.T, rule=res_balance)
    
    # Benders cuts constraint (will be populated during iterations)
    def benders_cuts(m, c):
        # alpha <= Phi[c] + Lambda[c] * (V[T1] - V24_hat[c])
        return m.alpha <= m.Phi[c] + m.Lambda[c] * (m.V[config.T1] - m.V24_hat[c])
    
    m.benders_cuts = pyo.Constraint(m.Cut, rule=benders_cuts)
    
    # Objective: first stage profit + expected future profit
    def objective(m):
        first_stage_profit = sum(config.pi[t] * 3.6 * config.E_conv * m.q[t] for t in m.T) - config.spillage_cost * sum(m.s[t] for t in m.T)
        return first_stage_profit + m.alpha
    
    m.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    
    return m

def build_simple_subproblem():
    """Build simplified Benders subproblem - second stage given first stage decision."""
    m = pyo.ConcreteModel()
    
    # Sets
    m.T = pyo.RangeSet(config.T1 + 1, config.T)  # Second stage time periods
    m.S = pyo.Set(initialize=config.scenarios)
    
    # Variables - second stage decisions per scenario
    m.q = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
    m.V = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
    m.s = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, None))
    
    # Parameter: V24 fixed from master problem
    m.V24_fixed = pyo.Param(initialize=0.0, mutable=True)
    
    # Dual suffix to get shadow prices
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # Linking constraint: connects first stage to second stage
    def linking_constraint(m, s):
        # V25[s] = V24_fixed + inflow - discharge - spillage
        inflow = config.scenario_info[s]  # Second stage inflow for scenario s
        return m.V[s, config.T1 + 1] == m.V24_fixed + config.alpha * inflow - config.alpha * m.q[s, config.T1 + 1] - m.s[s, config.T1 + 1]
    
    m.linking = pyo.Constraint(m.S, rule=linking_constraint)
    
    # Reservoir balance for remaining periods
    def res_balance(m, s, t):
        if t == config.T1 + 1:
            return pyo.Constraint.Skip  # Already handled by linking constraint
        inflow = config.scenario_info[s]
        return m.V[s, t] == m.V[s, t-1] + config.alpha * inflow - config.alpha * m.q[s, t] - m.s[s, t]
    
    m.res_balance = pyo.Constraint(m.S, m.T, rule=res_balance)
    
    # Objective: expected second stage profit
    def objective(m):
        total_profit = 0
        for s in m.S:
            scenario_profit = (
                sum(config.pi[t] * 3.6 * config.E_conv * m.q[s, t] for t in m.T) 
                - config.spillage_cost * sum(m.s[s, t] for t in m.T)
                + config.WV_end * m.V[s, config.T]
            )
            total_profit += config.prob[s] * scenario_profit
        return total_profit
    
    m.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    
    return m

def add_cut_to_master(master, cut_number, phi, lambda_val, V24_hat):
    """Add a new Benders cut to the master problem."""
    # Add the cut index to the set
    master.Cut.add(cut_number)
    
    # Set the cut parameters
    master.Phi[cut_number] = phi
    master.Lambda[cut_number] = lambda_val  
    master.V24_hat[cut_number] = V24_hat
    
    print(f"  Added cut {cut_number}: α ≤ {phi:.2f} + {lambda_val:.4f} * (V24 - {V24_hat:.3f})")

def run_simple_benders(plot=True, summary=True):
    """Run simplified Benders decomposition with clear iteration-by-iteration output."""
    start_time = time.time()
    
    if summary:
        print("\\n" + "="*80)
        print("Simple Benders Decomposition - Step by Step")
        print("="*80)
        print("This algorithm alternates between:")
        print("1. Master Problem: Choose first-stage decisions (q1, V24) + future cost approximation (α)")
        print("2. Subproblem: Given V24, find best second-stage decisions for all scenarios")
        print("3. Generate Cut: Add constraint to master based on subproblem dual information")
        print()
    
    # Build models
    master = build_simple_master()
    subproblem = build_simple_subproblem()
    solver = config.get_solver()
    
    # Initialize alpha to a reasonable lower bound
    master.alpha.set_value(-1e5)
    
    # Algorithm parameters
    tolerance = 1e-2  # Relaxed tolerance for educational purposes
    max_iterations = 8
    iteration = 0
    
    # Tracking
    upper_bounds = []  # From master problem
    lower_bounds = []  # From subproblem evaluation
    
    while iteration < max_iterations:
        iteration += 1
        
        if summary:
            print(f"--- ITERATION {iteration} ---")
        
        # STEP 1: Solve Master Problem
        if summary:
            print("Step 1: Solving master problem...")
        
        master_result = solver.solve(master, tee=False)
        
        if master_result.solver.termination_condition != pyo.TerminationCondition.optimal:
            if summary:
                print(f"  ERROR: Master problem failed: {master_result.solver.termination_condition}")
            break
        
        # Extract master solution
        V24_solution = pyo.value(master.V[config.T1])
        alpha_solution = pyo.value(master.alpha)
        master_obj = pyo.value(master.obj)
        
        if summary:
            print(f"  Master objective: {master_obj:,.2f} NOK")
            print(f"  First-stage ending reservoir level V24: {V24_solution:.3f} Mm³")
            print(f"  Expected future profit α: {alpha_solution:,.2f} NOK")
        
        # STEP 2: Solve Subproblem with fixed V24
        if summary:
            print("Step 2: Solving subproblem with V24 = {:.3f}...".format(V24_solution))
        
        subproblem.V24_fixed.set_value(V24_solution)
        sub_result = solver.solve(subproblem, tee=False)
        
        if sub_result.solver.termination_condition != pyo.TerminationCondition.optimal:
            if summary:
                print(f"  ERROR: Subproblem failed: {sub_result.solver.termination_condition}")
            break
        
        # Extract subproblem solution
        subproblem_obj = pyo.value(subproblem.obj)
        
        # Calculate true master objective (first stage + actual second stage)
        first_stage_obj = sum(config.pi[t] * 3.6 * config.E_conv * pyo.value(master.q[t]) for t in range(1, config.T1 + 1))
        first_stage_obj -= config.spillage_cost * sum(pyo.value(master.s[t]) for t in range(1, config.T1 + 1))
        true_obj = first_stage_obj + subproblem_obj
        
        upper_bounds.append(master_obj)
        lower_bounds.append(true_obj)
        
        if summary:
            print(f"  Subproblem objective: {subproblem_obj:,.2f} NOK")
            print(f"  True total objective: {true_obj:,.2f} NOK")
        
        # STEP 3: Check convergence
        gap = abs(alpha_solution - subproblem_obj)  # Gap between estimate and actual
        
        if summary:
            print(f"Step 3: Convergence check...")
            print(f"  Upper bound (master): {master_obj:,.2f}")
            print(f"  Lower bound (true): {true_obj:,.2f}")
            print(f"  Alpha vs subproblem gap: {gap:,.4f}")
        
        if gap < tolerance:
            if summary:
                print(f"  ✓ CONVERGED! Gap {gap:.4f} < tolerance {tolerance}")
            break
        
        # STEP 4: Generate and add Benders cut
        if summary:
            print("Step 4: Generating Benders cut...")
        
        # Get dual value from linking constraint (expected across scenarios)
        dual_sum = 0
        for s in config.scenarios:
            try:
                dual_val = subproblem.dual[subproblem.linking[s]]
                if dual_val is None:
                    dual_val = 0.0
                dual_sum += config.prob[s] * dual_val
            except:
                pass
        
        # Benders cut: α ≤ φ + λ * (V24 - V24_hat)  
        phi = subproblem_obj  # RHS constant
        lambda_coeff = dual_sum  # Coefficient of V24
        
        if summary:
            print(f"  Cut formula: α ≤ {phi:.2f} + {lambda_coeff:.4f} * (V24 - {V24_solution:.3f})")
        
        # Add cut to master
        add_cut_to_master(master, iteration, phi, lambda_coeff, V24_solution)
        
        if summary:
            print()
    
    solve_time = time.time() - start_time
    
    if summary:
        print("\\n" + "="*80)
        print("Simple Benders Results")
        print("="*80)
        print(f"Solve time: {solve_time:.3f} seconds")
        print(f"Iterations: {iteration}")
        print(f"Final objective: {lower_bounds[-1]:,.2f} NOK")
        print(f"Final V24: {V24_solution:.3f} Mm³")
        print(f"Convergence gap: {gap:.6f}")
        print("="*80)
    
    # Create convergence plot
    if plot and len(lower_bounds) > 1:
        iterations_list = list(range(1, len(lower_bounds) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations_list, upper_bounds, 'b-o', label='Upper bound (Master)', linewidth=2, markersize=8)
        plt.plot(iterations_list, lower_bounds, 'r-s', label='Lower bound (True)', linewidth=2, markersize=8)
        
        # Fill the gap area
        plt.fill_between(iterations_list, lower_bounds, upper_bounds, color='gray', alpha=0.3, label='Gap')
        
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value (NOK)')
        plt.title('Simple Benders Decomposition Convergence')
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.tight_layout()
        
        # Add convergence annotation
        if gap < tolerance:
            plt.annotate(f'Converged\\n(Gap: {gap:.4f})', 
                        xy=(iteration, lower_bounds[-1]), 
                        xytext=(iteration + 0.5, lower_bounds[-1]),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        plt.show()
    
    return {
        'objective': lower_bounds[-1] if lower_bounds else None,
        'V24': V24_solution,
        'iterations': iteration,
        'gap': gap,
        'solve_time': solve_time,
        'upper_bounds': upper_bounds,
        'lower_bounds': lower_bounds
    }