import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import time
from config import config

def build_master_problem():
    """Build Benders decomposition master problem."""
    master = pyo.ConcreteModel()
    
    # Sets
    master.T = pyo.RangeSet(1, config.T1)
    
    # Variables
    master.q = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap), initialize=0.0)
    master.V = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax), initialize=config.V0)
    master.theta = pyo.Var(within=pyo.Reals, bounds=(-1e6, 1e6), initialize=0.0)  # Future cost approximation
    master.s = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, None), initialize=0.0)  # Spillage
    
    # Expected inflow calculation
    I_exp = {}
    for t in range(1, config.T + 1):
        if t <= config.T1:
            I_exp[t] = config.certain_inflow
        else:
            I_exp[t] = sum(config.scenario_inflows) / len(config.scenario_inflows)
    
    # Reservoir balance constraint
    def master_res_rule(m, t):
        if t == 1:
            return m.V[t] == config.V0 + config.alpha * I_exp[t] - config.alpha * m.q[t] - m.s[t]
        else:
            return m.V[t] == m.V[t-1] + config.alpha * I_exp[t] - config.alpha * m.q[t] - m.s[t]
    
    master.res_balance = pyo.Constraint(master.T, rule=master_res_rule)
    
    # Objective function
    def master_obj_rule(m):
        return sum(config.pi[t] * 3.6 * config.E_conv * m.q[t] for t in m.T) + m.theta - config.spillage_cost * sum(m.s[t] for t in m.T)
    
    master.obj = pyo.Objective(rule=master_obj_rule, sense=pyo.maximize)
    
    return master

def build_subproblem():
    """Build Benders decomposition subproblem."""
    sub = pyo.ConcreteModel()
    
    # Sets
    sub.T = pyo.RangeSet(config.T1 + 1, config.T)
    sub.S = pyo.Set(initialize=config.scenarios)
    
    # Variables
    sub.q = pyo.Var(sub.S, sub.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap), initialize=0.0)
    sub.V = pyo.Var(sub.S, sub.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax), initialize=config.V0)
    sub.s = pyo.Var(sub.S, sub.T, within=pyo.NonNegativeReals, bounds=(0, None), initialize=0.0)
    
    # Suffix to capture dual values
    sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # Mutable parameter to receive V24 from master
    sub.V24 = pyo.Param(initialize=0.0, mutable=True)
    
    # Build inflow data
    I = {}
    for s in config.scenarios:
        for t in range(1, config.T + 1):
            if t <= config.T1:
                I[s, t] = config.certain_inflow
            else:
                I[s, t] = config.scenario_info[s]
    
    # Reservoir balance constraint for linking with master problem
    def V25_rule(m, s):
        return m.V[s, config.T1 + 1] == m.V24 + config.alpha * I[s, config.T1 + 1] - config.alpha * m.q[s, config.T1 + 1] - m.s[s, config.T1 + 1]
    sub.V25 = pyo.Constraint(sub.S, rule=V25_rule)
    
    # Reservoir balance constraint for remaining periods
    def sub_res_rule(m, s, t):
        if t == config.T1 + 1:
            return pyo.Constraint.Skip
        return m.V[s, t] == m.V[s, t-1] + config.alpha * I[s, t] - config.alpha * m.q[s, t] - m.s[s, t]
    sub.res_balance = pyo.Constraint(sub.S, sub.T, rule=sub_res_rule)
    
    # Objective function (per scenario)
    def sub_obj_rule(m, s):
        return sum(config.pi[t] * 3.6 * config.E_conv * m.q[s, t] for t in m.T) - config.spillage_cost * sum(m.s[s, t] for t in m.T) + config.WV_end * m.V[s, config.T]
    sub.obj = pyo.Objective(sub.S, rule=sub_obj_rule, sense=pyo.maximize)
    
    return sub, I

def run_benders_decomposition(plot=True, summary=True):
    """Run Benders decomposition algorithm."""
    start_time = time.time()
    
    # Build models
    master = build_master_problem()
    sub, I = build_subproblem()
    
    # Algorithm parameters
    solver = config.get_solver()
    tolerance = 1e-3
    max_iterations = 10
    iteration = 1
    global_LB = -float('inf')
    global_UB = float('inf')
    
    # Storage for convergence tracking
    upper_bound_list = []
    lower_bound_list = []
    
    if summary:
        print("\n" + "="*80)
        print("Benders Decomposition Algorithm")
        print("="*80)
    
    while abs(global_UB - global_LB) > tolerance and iteration <= max_iterations:
        if summary:
            print(f"\n--- Iteration {iteration} ---")
        
        # Solve master problem
        master_result = solver.solve(master)
        
        if master_result.solver.termination_condition != pyo.TerminationCondition.optimal:
            if summary:
                print(f"Master problem failed: {master_result.solver.termination_condition}")
            break
        
        master_V24 = pyo.value(master.V[config.T1])
        master_theta = pyo.value(master.theta)
        master_obj_value = pyo.value(master.obj)
        
        if summary:
            print(f"Master objective: {master_obj_value:.2f}, V24: {master_V24:.3f}")
        
        # Solve subproblems for each scenario
        expected_sub_obj = 0.0
        duals_V25 = []
        w_s_list = []
        any_subproblem_failed = False
        
        for s in config.scenarios:
            # Update V24 in subproblem
            sub.V24.set_value(master_V24)
            
            # Calculate maximum feasible discharge
            max_feasible_discharge = (master_V24 + config.alpha * I[s, config.T1 + 1]) / config.alpha
            sub.q[s, config.T1 + 1].setub(max(0.0, min(config.q_cap, max_feasible_discharge)))
            
            # Activate only the objective for this scenario
            for obj in sub.obj.values():
                try:
                    obj.deactivate()
                except:
                    pass
            sub.obj[s].activate()
            
            sub_result = solver.solve(sub)
            
            if sub_result.solver.termination_condition != pyo.TerminationCondition.optimal:
                if summary:
                    print(f"Subproblem for {s} failed")
                any_subproblem_failed = True
                break
            
            sub_obj_value = pyo.value(sub.obj[s])
            w_s_list.append(sub_obj_value)
            expected_sub_obj += config.prob[s] * sub_obj_value
            
            # Get dual value
            try:
                dual_value = sub.dual[sub.V25[s]]
                if dual_value is None:
                    dual_value = 0.0
            except:
                dual_value = 0.0
            duals_V25.append(dual_value)
        
        # Update bounds
        if not any_subproblem_failed:
            lower_bound = master_obj_value
            upper_bound = master_obj_value - master_theta + expected_sub_obj
            
            global_LB = max(global_LB, lower_bound)
            global_UB = min(global_UB, upper_bound)
            
            lower_bound_list.append(lower_bound)
            upper_bound_list.append(upper_bound)
            
            if summary:
                print(f"Bounds - Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}")
                print(f"Gap: {abs(upper_bound - lower_bound):.4f}")
            
            # Add Benders cut
            cut_name = f"BendersCut_{iteration}"
            
            def benders_cut_rule(m):
                return m.theta <= sum(
                    config.prob[s] * (w_s_list[i] + duals_V25[i] * (m.V[config.T1] - master_V24))
                    for i, s in enumerate(config.scenarios)
                )
            
            setattr(master, cut_name, pyo.Constraint(rule=benders_cut_rule))
        else:
            # Add feasibility cut
            cut_name = f"FeasibilityCut_{iteration}"
            setattr(master, cut_name, pyo.Constraint(expr=master.V[config.T1] <= master_V24 - 0.1))
            if summary:
                print(f"Added feasibility cut: V[{config.T1}] <= {master_V24 - 0.1:.3f}")
        
        iteration += 1
    
    solve_time = time.time() - start_time
    
    if summary:
        print("\n" + "="*80)
        print("Benders Decomposition Results")
        print("="*80)
        print(f"Solve time: {solve_time:.3f} seconds")
        print(f"Iterations: {iteration - 1}")
        
        if global_UB < float('inf'):
            print(f"Optimal objective: {global_UB:.2f} NOK")
            print(f"Optimal V24: {master_V24:.3f} Mm³")
        else:
            print("No feasible solution found")
    
    if plot and len(lower_bound_list) > 0:
        # Create convergence plot
        iters = list(range(1, len(lower_bound_list) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(iters, upper_bound_list, 'b-o', label='Upper bound', linewidth=2)
        plt.plot(iters, lower_bound_list, 'r-s', label='Lower bound', linewidth=2)
        plt.fill_between(iters, lower_bound_list, upper_bound_list, color='gray', alpha=0.2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Objective value (NOK)')
        plt.title('Benders Decomposition Convergence')
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    return master, sub, global_UB, iteration - 1, solve_time