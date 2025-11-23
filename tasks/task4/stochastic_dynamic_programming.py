"""
Stochastic Dynamic Programming (SDP) for Hydropower Scheduling

This module implements a stochastic dynamic programming approach to the two-stage
hydropower scheduling problem. The method pre-computes optimal second-stage solutions
for all discretized first-stage reservoir states, then uses these to generate cuts
for the master problem.

The approach:
1. Discretize the first-stage decision space (V24 values)
2. For each discrete V24, solve the second-stage problem across all scenarios
3. Generate cuts from the dual information
4. Solve the master problem with all pre-computed cuts
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from config import Config

def build_sdp_master_problem(cuts_data):
    """
    Build the master problem for SDP with pre-computed cuts.
    
    Args:
        cuts_data: Dictionary containing cut information from all discrete states
        
    Returns:
        Pyomo ConcreteModel: The master problem with cuts
    """
    config = Config()
    
    # Create model
    m = pyo.ConcreteModel("SDP_Master")
    
    # Sets
    m.T1 = pyo.RangeSet(1, config.T1)
    
    # Parameters - first stage only
    m.pi = pyo.Param(m.T1, initialize={t: config.pi[t] for t in range(1, config.T1 + 1)}, doc="Electricity prices (EUR/MWh)")
    m.p_max = pyo.Param(initialize=config.Pmax, doc="Maximum power output (MW)")
    m.V_min = pyo.Param(initialize=0.0, doc="Minimum reservoir volume (Mm³)")
    m.V_max = pyo.Param(initialize=config.Vmax, doc="Maximum reservoir volume (Mm³)")
    m.V0 = pyo.Param(initialize=config.V0, doc="Initial reservoir volume (Mm³)")
    m.spillage_cost = pyo.Param(initialize=config.spillage_cost, doc="Spillage cost (EUR/m³)")
    
    # First-stage decision variables
    m.q = pyo.Var(m.T1, within=pyo.NonNegativeReals, doc="Discharge (m³/s)")
    m.V = pyo.Var(m.T1, bounds=(0.0, config.Vmax), doc="Reservoir volume (Mm³)")
    m.s = pyo.Var(m.T1, within=pyo.NonNegativeReals, doc="Spillage (m³/s)")
    
    # Variable for expected second-stage profit
    m.alpha = pyo.Var(bounds=(-1e6,None), doc="Expected second-stage profit (EUR)")
    
    # Cut information
    if cuts_data["cuts"]:
        m.CUT_SET = pyo.Set(initialize=list(range(len(cuts_data["cuts"]))))
        
        # Parameters for cuts
        cut_phi = {}
        cut_lambda = {}
        cut_v24_hat = {}
        
        for i, cut in enumerate(cuts_data["cuts"]):
            cut_phi[i] = cut["phi"]
            cut_lambda[i] = cut["lambda"]
            cut_v24_hat[i] = cut["v24_hat"]
        
        m.cut_phi = pyo.Param(m.CUT_SET, initialize=cut_phi)
        m.cut_lambda = pyo.Param(m.CUT_SET, initialize=cut_lambda) 
        m.cut_v24_hat = pyo.Param(m.CUT_SET, initialize=cut_v24_hat)
        
        # Cut constraints
        def cut_constraint(m, c):
            return m.alpha <= m.cut_phi[c] + m.cut_lambda[c] * (m.V[config.T1] - m.cut_v24_hat[c])
        
        m.cuts = pyo.Constraint(m.CUT_SET, rule=cut_constraint)
    else:
        # No cuts available - use a large negative bound on alpha
        m.alpha.setlb(-1000000)
    
    # First-stage constraints
    def water_balance(m, t):
        # Reservoir evolution during first stage: add inflow, subtract discharge and spillage
        if t == 1:
            return m.V[t] == m.V0 + config.alpha * (config.certain_inflow - m.q[t] - m.s[t])
        else:
            return m.V[t] == m.V[t-1] + config.alpha * (config.certain_inflow - m.q[t] - m.s[t])
    
    def turbine_capacity(m, t):
        return m.q[t] <= config.Qmax
    
    def power_constraint(m, t):
        return config.E_conv * 3.6 * m.q[t] <= m.p_max
    
    m.water_balance = pyo.Constraint(m.T1, rule=water_balance)
    m.turbine_capacity = pyo.Constraint(m.T1, rule=turbine_capacity)
    m.power_constraint = pyo.Constraint(m.T1, rule=power_constraint)
    
    # Objective: First-stage profit + expected second-stage profit
    def objective(m):
        first_stage_revenue = sum(m.pi[t] * config.E_conv * 3.6 * m.q[t] for t in m.T1)
        spillage_cost = sum(m.spillage_cost * m.s[t] * config.alpha for t in m.T1)
        return first_stage_revenue - spillage_cost + m.alpha
    
    m.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    
    return m

def build_sdp_subproblem(v24_value):
    """
    Build second-stage subproblem for a given V24 value.
    
    Args:
        v24_value: Fixed reservoir volume at hour 24 (Mm³)
        
    Returns:
        Pyomo ConcreteModel: The second-stage subproblem
    """
    config = Config()
    
    # Create model
    m = pyo.ConcreteModel("SDP_Subproblem")
    
    # Sets
    m.T2 = pyo.RangeSet(config.T1 + 1, config.T)
    m.S = pyo.Set(initialize=config.scenarios)
    
    # Parameters
    m.pi = pyo.Param(m.T2, initialize={t: config.pi[t] for t in range(config.T1 + 1, config.T + 1)})
    m.prob = pyo.Param(m.S, initialize=dict(zip(config.scenarios, config.scenario_probabilities)))
    m.inflow = pyo.Param(m.S, initialize=dict(zip(config.scenarios, config.scenario_inflows)))
    m.V24_start = pyo.Param(initialize=v24_value)
    # Linking variable for V24 (reservoir volume at end of first stage)
    m.v24_dec = pyo.Var(bounds=(0.0, config.Vmax))
    
    m.p_max = pyo.Param(initialize=config.Pmax)
    m.q_max = pyo.Param(initialize=config.Qmax)
    m.V_min = pyo.Param(initialize=0.0)
    m.V_max = pyo.Param(initialize=config.Vmax)
    m.WV_end = pyo.Param(initialize=config.WV_end)
    m.spillage_cost = pyo.Param(initialize=config.spillage_cost)
    
    # Second-stage variables
    m.q = pyo.Var(m.T2, m.S, bounds=(0.0, config.Qmax))
    m.V = pyo.Var(m.T2, m.S, bounds=(0.0, config.Vmax))
    m.s = pyo.Var(m.T2, m.S, within=pyo.NonNegativeReals)
    
    
    def water_balance(m, t, s):
        if t == config.T1 + 1:
            return m.V[t, s] == m.v24_dec + config.alpha * (m.inflow[s] - m.q[t, s] - m.s[t, s])
        else:
            return m.V[t, s] == m.V[t-1, s] + config.alpha * (m.inflow[s] - m.q[t, s] - m.s[t, s])

    # Fix linking variable to supplied candidate value to obtain correct dual value
    def fix_v24(m):
        return m.V24_start == m.v24_dec

    # discharge capacity
    def discharge_capacity(m, t, s):
        return m.q[t, s] <= m.q_max

    # power capacity
    def power_capacity(m, t, s):
        return config.E_conv * 3.6 * m.q[t, s] <= m.p_max

    
    m.water_balance = pyo.Constraint(m.T2, m.S, rule=water_balance)
    m.v24_fix = pyo.Constraint(rule=fix_v24)
    m.discharge_capacity = pyo.Constraint(m.T2, m.S, rule=discharge_capacity)
    m.power_capacity = pyo.Constraint(m.T2, m.S, rule=power_capacity)

    # Objective: Expected second-stage profit
    def objective(m):
        revenue = sum(m.prob[s] * sum(m.pi[t] * 3.6 * config.E_conv * m.q[t, s]
                                     for t in m.T2) for s in m.S)
        
        spillage_cost = sum(m.prob[s] * sum(m.spillage_cost * m.s[t, s] * config.alpha for t in m.T2) for s in m.S)
        
        end_value = sum(m.prob[s] * m.WV_end * m.V[config.T, s] for s in m.S)
        
        return revenue - spillage_cost + end_value
    
    m.obj = pyo.Objective(rule=objective, sense=pyo.maximize)
    
    return m

def solve_model(model):
    """Solve a Pyomo model and return results."""
    config = Config()
    solver = config.get_solver()
    
    # Add dual suffix for cut generation
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    results = solver.solve(model, tee=False)
    
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        if (results.solver.termination_condition == pyo.TerminationCondition.infeasible or
            results.solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded):
            return None, None  # Return None for infeasible cases
        else:
            raise RuntimeError(f"Solver failed: {results.solver.termination_condition}")
    
    return results, model

def generate_discrete_states(number_of_intervals=500):
    """
    Generate discrete V24 values for SDP exploration.
    
    Returns:
        list: Discrete V24 values to evaluate
    """
    config = Config()
    
    # Create discrete grid for V24
    # Based on Benders solution (V24=4.16), use a range close to optimal
    # Use manual points around the known optimal region
    v24_states = []
    for v in np.arange(0, 4.5, 4.5/number_of_intervals):
        v24_states.append(round(v, 2))
    
    return list(v24_states)

def run_sdp(plot=True, summary=True):
    """
    Run Stochastic Dynamic Programming approach.
    
    Args:
        plot (bool): Whether to create plots
        summary (bool): Whether to print summary statistics
        
    Returns:
        dict: Results including objective value, decisions, and timing
    """
    """
    Steps:
    Step 1: Make list of V24 values to explore
    Step 2: Run second-stage problem for each of the V24 values to generate cuts
    Step 3: Solve master problem with all pre-computed cuts
    """
    
    start_time = time.time()
    config = Config()
    
    # Step 1: Generate discrete states
    v24_states = generate_discrete_states(number_of_intervals=200)
    
    # Step 2: Pre-compute all cuts
    cuts_data = {"cuts": []}
    
    for i, v24 in enumerate(v24_states):
        print(f"State {i+1}/{len(v24_states)}: V24 = {v24:.2f} Mm³", end="")
        
        # Solve subproblem for this V24
        subproblem = build_sdp_subproblem(v24)
        results, solved_sub = solve_model(subproblem)
        
        # Handle infeasible cases
        if results is None or solved_sub is None:
            print(" → INFEASIBLE - skipped")
            continue
        
        # Extract cut information
        phi = pyo.value(solved_sub.obj)  # Second-stage objective value at v24_hat
        lambda_val = pyo.value(solved_sub.dual[solved_sub.v24_fix]) # dual value of linking constraint
        
        # Store cut
        cut = {
            "phi": phi,
            "lambda": lambda_val,
            "v24_hat": v24
        }
        cuts_data["cuts"].append(cut)
        
        print(f" → φ = {phi:,.0f}, λ = {lambda_val:,.0f}")
    
    print(f"Generated {len(cuts_data['cuts'])} cuts")
    print()
    
    # Check if we have any cuts
    if len(cuts_data['cuts']) == 0:
        print("No feasible cuts generated - SDP cannot proceed")
        return {"method": "SDP", "status": "failed", "reason": "No feasible V24 values"}
    
    # Step 3: Solve master problem with all cuts
    print("--- STEP 3: Solving master problem ---")
    cuts_data['raw_cuts'] = list(cuts_data['cuts'])

    master = build_sdp_master_problem(cuts_data)
    results, solved_master = solve_model(master)
    
    solve_time = time.time() - start_time
    
    # Extract results
    objective = pyo.value(solved_master.obj)
    v24_optimal = pyo.value(solved_master.V[config.T1])
    alpha_value = pyo.value(solved_master.alpha)
    
    # Get first-stage decisions
    discharge_stage1 = [pyo.value(solved_master.q[t]) for t in solved_master.T1]
    reservoir_stage1 = [pyo.value(solved_master.V[t]) for t in solved_master.T1]
    spillage_stage1 = [pyo.value(solved_master.s[t]) for t in solved_master.T1]
    
    print(f"Master problem solved:")
    print(f"  Total objective: {objective:,.2f} NOK")
    print(f"  Optimal V24: {v24_optimal:.3f} Mm³")
    print(f"  Expected stage 2 profit: {alpha_value:,.2f} NOK")
    print()
    
    # Solve the actual second-stage problem with optimal V24 for validation
    validation_sub = build_sdp_subproblem(v24_optimal)
    val_results, solved_val = solve_model(validation_sub)
    actual_stage2_profit = pyo.value(solved_val.obj)
    
    print(f"Validation (stage 2 at optimal V24): {actual_stage2_profit:,.2f} NOK")
    print(f"SDP estimate vs actual: {alpha_value:,.2f} vs {actual_stage2_profit:,.2f}")
    print(f"Difference: {abs(alpha_value - actual_stage2_profit):,.2f} NOK")

    
    # Build full-horizon trajectories:
    # - First stage: exact values from master
    # - Second stage: expected values across scenarios from validation_sub
    hours_q = list(range(1, config.T + 1))       # 1..T
    hours_V = [0] + hours_q                      # volume at time 0..T

    # Expected second-stage discharge and volume (T1+1..T)
    discharge_stage2 = []
    reservoir_stage2 = []
    # Scenario-specific series for plotting thin lines
    scenario_discharge = {s: [] for s in solved_val.S}
    scenario_reservoir = {s: [] for s in solved_val.S}

    for t in range(config.T1 + 1, config.T + 1):
        q_exp = 0.0
        V_exp = 0.0
        for s in solved_val.S:
            prob_s = pyo.value(solved_val.prob[s])
            q_exp += prob_s * pyo.value(solved_val.q[t, s])
            V_exp += prob_s * pyo.value(solved_val.V[t, s])
            # store raw scenario values
            scenario_discharge[s].append(pyo.value(solved_val.q[t, s]))
            scenario_reservoir[s].append(pyo.value(solved_val.V[t, s]))
        discharge_stage2.append(q_exp)
        reservoir_stage2.append(V_exp)

    # Concatenate: first stage + expected second stage
    discharge_all = discharge_stage1 + discharge_stage2
    reservoir_all = [config.V0] + reservoir_stage1 + reservoir_stage2

    
    results_dict = {
        "method": "Stochastic Dynamic Programming",
        "objective": objective,
        "solve_time": solve_time,
        "v24_optimal": v24_optimal,
        "alpha_estimate": alpha_value,
        "actual_stage2": actual_stage2_profit,
        "num_cuts": len(cuts_data["cuts"]),
        "discharge_stage1": discharge_stage1,
        "reservoir_stage1": reservoir_stage1,
        "spillage_stage1": spillage_stage1,
        "cuts_data": cuts_data,
        "discharge_all": discharge_all,
        "reservoir_all": reservoir_all,
        "hours_q": hours_q,
        "hours_V": hours_V,
        "scenario_discharge_stage2": scenario_discharge,
        "scenario_reservoir_stage2": scenario_reservoir,
    }

    if summary:
        print_summary(results_dict)
    
    if plot:
        create_plots(results_dict)
    
    return results_dict

def print_summary(results):
    """Print summary statistics for SDP results."""
    print("================================================================================")
    print("Stochastic Dynamic Programming Results")
    print("================================================================================")
    print(f"Total solve time: {results['solve_time']:.3f} seconds")
    print(f"Number of cuts generated: {results['num_cuts']}")
    print(f"Final objective value: {results['objective']:,.2f} NOK")
    print(f"Optimal V24 decision: {results['v24_optimal']:.3f} Mm³")
    print(f"Expected vs actual stage 2: {results['alpha_estimate']:,.0f} vs {results['actual_stage2']:,.0f} NOK")
    
    # First stage statistics
    discharge_avg = np.mean(results['discharge_stage1'])
    discharge_max = max(results['discharge_stage1'])
    discharge_min = min(results['discharge_stage1'])
    
    reservoir_final = results['reservoir_stage1'][-1]
    reservoir_max = max(results['reservoir_stage1'])
    reservoir_min = min(results['reservoir_stage1'])
    
    total_spillage = sum(results['spillage_stage1'])
    
    print("First stage statistics:")
    print(f"  Discharge - Avg: {discharge_avg:.2f} m³/s, Max: {discharge_max:.2f} m³/s, Min: {discharge_min:.2f} m³/s")
    print(f"  Reservoir - Final: {reservoir_final:.2f} Mm³, Max: {reservoir_max:.2f} Mm³, Min: {reservoir_min:.2f} Mm³")
    print(f"  Total spillage: {total_spillage:.2f} m³/s·h")
    print("================================================================================")

def create_plots(results):
    """Create visualization plots for SDP results."""
    config = Config()

    # =======================
    # Plot 1: q & V over time
    # =======================
    hours_q = results["hours_q"]
    hours_V = results["hours_V"]
    q_all = results["discharge_all"]
    V_all = results["reservoir_all"]
    scenario_q = results.get("scenario_discharge_stage2", {})
    scenario_V = results.get("scenario_reservoir_stage2", {})

    plt.figure(figsize=(11, 4.5))
    ax = plt.gca()
    ax2 = ax.twinx()

    # Discharge (primary y-axis)
    lq, = ax.plot(hours_q, q_all, linewidth=2, label="Discharge q (expected)")
    # Scenario-specific discharge lines (second stage only)
    if scenario_q:
        hours_stage2 = list(range(config.T1 + 1, config.T + 1))
        first_label_added = False
        for s, series in scenario_q.items():
            lbl = "Scenario discharge" if not first_label_added else None
            ax.plot(hours_stage2, series, color="#1f77b4", alpha=0.4, linewidth=1.0, label=lbl)
            first_label_added = True

    # Reservoir (secondary y-axis)
    lV, = ax2.plot(hours_V, V_all, linestyle="--", linewidth=2,
                   label="Reservoir V (expected)", zorder=3)
    # Scenario-specific reservoir lines (second stage only)
    if scenario_V:
        hours_stage2 = list(range(config.T1 + 1, config.T + 1))
        first_label_added_R = False
        for s, series in scenario_V.items():
            lbl = "Scenario reservoir" if not first_label_added_R else None
            ax2.plot(hours_stage2, series, color="#ff7f0e", alpha=0.4, linewidth=1.0, label=lbl)
            first_label_added_R = True

    ax.set_title("SDP — first stage exact; second stage expected", fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Discharge q (m³/s)")
    ax2.set_ylabel("Reservoir level V (Mm³)")

    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, config.T)
    ax.axvline(x=config.T1, linestyle=":", alpha=0.7, label="End of first stage")

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

    # ================================
    # Plot 2: Second-stage objective φ
    # ================================
    cuts_data = results.get("cuts_data", {})
    raw_cuts = cuts_data.get("raw_cuts") or cuts_data.get("cuts") or []
    if raw_cuts:
        v_vals = [c["v24_hat"] for c in raw_cuts]
        phi_vals = [c["phi"] for c in raw_cuts]

        plt.figure(figsize=(10.5, 4.2))
        plt.plot(v_vals, phi_vals, color="#1f77b4", linewidth=1.6)
        plt.title("Second-stage objective value vs starting V24", fontweight="bold")
        plt.xlabel("Starting reservoir volume V24 (Mm³)")
        plt.ylabel("Second-stage objective φ (NOK)")
        plt.grid(alpha=0.5, linewidth=0.6)

        # Highlight optimal V24 if present
        v24_opt = results.get("v24_optimal")
        if v24_opt is not None:
            plt.axvline(v24_opt, color="red", linestyle="--", alpha=0.7, label=f"Optimal V24 = {v24_opt:.3f}")
            plt.legend(loc="lower left")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Run SDP standalone
    results = run_sdp(plot=True, summary=True)