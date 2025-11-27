import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from config import config

def build_benders_master(cuts_data):
    """Build Benders master problem"""
    m = pyo.ConcreteModel()
    
    # Sets
    m.T = pyo.RangeSet(1, config.T1)
    
    # Variables - first stage decisions
    m.q = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
    m.V = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
    m.s = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, None))
    
    # Alpha = approximation of second stage profit
    m.a = pyo.Var(within=pyo.Reals, bounds=(-1e6, 1e6))
    
    # Cut information (from cuts_data)
    m.Cut = pyo.Set(initialize=cuts_data["Set"])
    m.Phi = pyo.Param(m.Cut, initialize=cuts_data["Phi"])
    m.Lambda = pyo.Param(m.Cut, initialize=cuts_data["Lambda"])  
    m.V24_hat = pyo.Param(m.Cut, initialize=cuts_data["V24_hat"])
    
    # Reservoir balance (first stage only)
    def res_balance(m, t):
        if t == 1:
            return m.V[t] == config.V0 + config.alpha * config.certain_inflow - config.alpha * m.q[t] - m.s[t]
        else:
            return m.V[t] == m.V[t-1] + config.alpha * config.certain_inflow - config.alpha * m.q[t] - m.s[t]
    
    m.res_balance = pyo.Constraint(m.T, rule=res_balance)
    
    # Benders cuts
    def benders_cuts(m, c):
        print(f"Cut {c}: α ≤ {m.Phi[c]:.2f} + {m.Lambda[c]:.4f} * (V24 - {m.V24_hat[c]:.3f})")
        return m.a <= m.Phi[c] + m.Lambda[c] * (m.V[config.T1] - m.V24_hat[c])
    
    m.benders_cuts = pyo.Constraint(m.Cut, rule=benders_cuts)
    
    # Objective = first stage profit + expected second stage profit
    def objective(m):
        first_stage = sum(config.pi[t] * 3.6 * config.E_conv * m.q[t] for t in m.T) - config.spillage_cost * sum(m.s[t] for t in m.T)
        return first_stage + m.a
    
    m.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    return m

def build_benders_subproblem(V24_hat):
    """Subproblem with explicit linking variable and equality whose dual gives λ_s."""
    m = pyo.ConcreteModel()

    m.T = pyo.RangeSet(config.T1 + 1, config.T)
    m.S = pyo.Set(initialize=config.scenarios)

    m.q = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
    m.V = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
    m.s = pyo.Var(m.S, m.T, within=pyo.NonNegativeReals, bounds=(0, None))

    m.V24_hat = pyo.Param(initialize=V24_hat)
    m.V_link = pyo.Var(m.S, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def link_rule(m, s):
        return m.V_link[s] == m.V24_hat
    m.link_v24 = pyo.Constraint(m.S, rule=link_rule)

    def first_period(m, s):
        inflow = config.scenario_info[s]
        return m.V[s, config.T1 + 1] == m.V_link[s] + config.alpha * inflow - config.alpha * m.q[s, config.T1 + 1] - m.s[s, config.T1 + 1]
    m.first_period = pyo.Constraint(m.S, rule=first_period)

    def res_balance(m, s, t):
        if t == config.T1 + 1:
            return pyo.Constraint.Skip
        inflow = config.scenario_info[s]
        return m.V[s, t] == m.V[s, t-1] + config.alpha * inflow - config.alpha * m.q[s, t] - m.s[s, t]
    m.res_balance = pyo.Constraint(m.S, m.T, rule=res_balance)

    def objective(m):
        total = 0
        for s in m.S:
            scen_profit = (
                sum(config.pi[t] * 3.6 * config.E_conv * m.q[s, t] for t in m.T)
                - config.spillage_cost * sum(m.s[s, t] for t in m.T)
                + config.WV_end * m.V[s, config.T]
            )
            total += config.prob[s] * scen_profit
        return total
    m.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    return m

def solve_model(model, solver):
    """Solve model and return results."""
    result = solver.solve(model, tee=False)
    return result, model

def manage_cuts(cuts_data, subproblem_model):
    cut = len(cuts_data["Set"])
    cuts_data["Set"].append(cut)
    cuts_data["Phi"][cut] = pyo.value(subproblem_model.obj)
    dual_sum = 0.0
    for s in config.scenarios:
        dual_val = subproblem_model.dual[subproblem_model.link_v24[s]]
        dual_sum += dual_val

    cuts_data["Lambda"][cut] = dual_sum
    cuts_data["V24_hat"][cut] = pyo.value(subproblem_model.V24_hat)
    return cuts_data

def run_benders_decomposition(plot=True, summary=True, binding_tol=1e-6, enforce_min_iterations=True, plot_cuts_overview=False):
    """Run Benders decomposition following a clear step-by-step pattern."""
    start_time = time.time()
    
    if summary:
        print("\n" + "="*80)
        print("Benders Decomposition")
        print("="*80)
        print("Algorithm steps:")
        print("1. Solve master problem → get V24 decision")
        print("2. Fix V24 in subproblem → solve for all scenarios")  
        print("3. Generate cut from dual information")
        print("4. Add cut to master and repeat")
        print()
    
    # Setup for Benders decomposition
    max_iterations = 10
    solver = config.get_solver()
    
    # Initialize cut data structure
    cuts_data = {
        "Set": [],
        "Phi": {},
        "Lambda": {},
        "V24_hat": {}
    }
    
    # Track convergence
    upper_bounds = []
    lower_bounds = []
    lower_bound = -float('inf')
    upper_bound = float('inf')
    
    # Track iteration sequence for cuts overview
    v24_sequence = []
    alpha_sequence = []

    # main benders loop
    for i in range(max_iterations):
        if summary:
            print(f"--- ITERATION {i+1} ---")
        
        # STEP 1: Solve master problem
        master = build_benders_master(cuts_data)
        # Attach dual suffix to inspect reservoir balance duals
        master_result, master = solve_model(master, solver)
        
        if master_result.solver.termination_condition != pyo.TerminationCondition.optimal:
            if summary:
                print(f"Master problem failed: {master_result.solver.termination_condition}")
            break
        
        # Extract master solution
        V24_solution = pyo.value(master.V[config.T1])
        a_solution = pyo.value(master.a)
        master_obj = pyo.value(master.obj)
        
        upper_bounds.append(master_obj)
        upper_bound = min(upper_bound, master_obj)
        
        v24_sequence.append(V24_solution)
        alpha_sequence.append(a_solution)

        if summary:
            print(f"Master problem solved:")
            print(f"  Objective: {master_obj:,.2f} NOK")
            print(f"  V24 decision: {V24_solution:.3f} Mm³")
            print(f"  Alpha estimate: {a_solution:,.2f} NOK")
            
        
        # STEP 2: Solve subproblem with fixed V24
        subproblem = build_benders_subproblem(V24_solution)
       
        sub_result, subproblem = solve_model(subproblem, solver)
        
        if sub_result.solver.termination_condition != pyo.TerminationCondition.optimal:
            if summary:
                print(f"Subproblem failed: {sub_result.solver.termination_condition}")
            break
        
        subproblem_obj = pyo.value(subproblem.obj)
        
        # Calculate true objective (master first stage + subproblem)
        first_stage_profit = sum(config.pi[t] * 3.6 * config.E_conv * pyo.value(master.q[t]) for t in range(1, config.T1 + 1))
        first_stage_profit -= config.spillage_cost * sum(pyo.value(master.s[t]) for t in range(1, config.T1 + 1))
        true_obj = first_stage_profit + subproblem_obj
        
        lower_bounds.append(true_obj)
        lower_bound = max(lower_bound, true_obj)
        
        
        # STEP 3: Generate new cut
        cuts_data = manage_cuts(cuts_data, subproblem)

        gap = upper_bound - lower_bound
        
        if summary:
            print(f"Convergence check:")
            print(f"  Gap (UB - LB): {gap:,.2f}")
            print(f"  Upper bound (UB): {upper_bound:,.2f}")
            print(f"  Lower bound (LB): {lower_bound:,.2f}")
            
            # Check if converged
            tolerance = 1e-8
            if gap < tolerance:
                print(f"  ✓ CONVERGED! Gap {gap:.8f} < tolerance {tolerance}")
                break
            print()
    
    solve_time = time.time() - start_time
    
    if summary:
        print("="*80)
        print("Benders Decomposition Results")
        print("="*80)
        print(f"Total solve time: {solve_time:.3f} seconds")
        if lower_bounds:
            print(f"Final objective (LB): {lower_bound:,.2f} NOK")
            print(f"Final V24: {V24_solution:.3f} Mm³")
            print(f"Final gap: {gap:,.2f}")
        print("="*80)
    

    # Create discharge and reservoir level plot
    if plot and 'master' in locals():
        # Extract solution from final master and subproblem
        hours_q = list(range(1, config.T + 1))
        hours_V = [0] + hours_q
        
        # First stage solution (from master)
        benders_q = []
        benders_V = [config.V0]
        
        for t in range(1, config.T1 + 1):
            benders_q.append(pyo.value(master.q[t]))
            benders_V.append(pyo.value(master.V[t]))
        
        # Scenario-specific second stage solution & expected values
        scenario_q = {s: [] for s in config.scenarios}
        scenario_V = {s: [] for s in config.scenarios}
        for t in range(config.T1 + 1, config.T + 1):
            expected_q = 0.0
            expected_V = 0.0
            for s in config.scenarios:
                val_q = pyo.value(subproblem.q[s, t])
                val_V = pyo.value(subproblem.V[s, t])
                scenario_q[s].append(val_q)
                scenario_V[s].append(val_V)
                expected_q += config.prob[s] * val_q
                expected_V += config.prob[s] * val_V
            benders_q.append(expected_q)
            benders_V.append(expected_V)
        
        plt.figure(figsize=(11, 4.5))
        ax = plt.gca()
        ax2 = ax.twinx()
        
        lq, = ax.plot(hours_q, benders_q, color='purple', linewidth=2, label='Discharge q (expected)')
        lV, = ax2.plot(hours_V, benders_V, color='purple', linestyle='--', linewidth=2, label='Reservoir V (expected)', zorder=3)

        # Scenario thin lines (second stage only) with connection at hour T1
        hours_stage2 = list(range(config.T1 + 1, config.T + 1))
        hours_stage2_with_start = [config.T1] + hours_stage2
        last_q1 = pyo.value(master.q[config.T1])
        last_V1 = pyo.value(master.V[config.T1])
        for s in config.scenarios:
            series_q = [last_q1] + scenario_q[s]
            ax.plot(hours_stage2_with_start, series_q, color='purple', alpha=0.4, linewidth=1.0)
        for s in config.scenarios:
            series_V = [last_V1] + scenario_V[s]
            ax2.plot(hours_stage2_with_start, series_V, color='orange', alpha=0.4, linewidth=1.0)
        
        ax.set_title('Benders Decomposition — first stage exact; expected & scenario lines on second stage', fontweight='bold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Discharge q (m³/s)')
        ax2.set_ylabel('Reservoir Level V (Mm³)')
        ax.grid(True, linewidth=0.5, alpha=0.6)
        ax.set_xlim(0, config.T)
        ax.axvline(x=config.T1, color='gray', linestyle=':', alpha=0.7, label='End of first stage')
        
        # Combined legend bottom-left with scenario explanation
        handles_ax, labels_ax = ax.get_legend_handles_labels()
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
        scenario_proxy = Line2D([0], [0], color='gray', linewidth=1.0, alpha=0.6,
                     label='Scenario trajectories (thin lines)')
        all_handles = handles_ax + handles_ax2 + [scenario_proxy]
        all_labels = labels_ax + labels_ax2 + [scenario_proxy.get_label()]
        ax.legend(all_handles, all_labels, loc='lower left')
        plt.tight_layout()
        plt.show()
    
    return {
        'objective': lower_bound if lower_bound else None,
        'V24': V24_solution if 'V24_solution' in locals() else None,
        'iterations': max_iterations,
        'solve_time': solve_time,
        'upper_bounds': upper_bounds,
        'lower_bounds': lower_bounds,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'V24_sequence': v24_sequence,
        'alpha_sequence': alpha_sequence,
        'binding_cuts_last_master': [c for c in cuts_data['Set'] if abs(a_solution - (cuts_data['Phi'][c] + cuts_data['Lambda'][c] * (V24_solution - cuts_data['V24_hat'][c]))) <= binding_tol] if cuts_data['Set'] else []
    }