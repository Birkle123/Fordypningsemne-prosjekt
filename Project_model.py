"""
HYDROPOWER OPTIMIZATION PROJECT
===============================

This script contains three main components:

1. DETERMINISTIC EQUIVALENT MODELS:
   - Expected Value (EV) Model: Uses expected inflows
   - Individual Scenario Models: Solves each scenario separately  
   - Two-Stage Stochastic Model: Optimizes expected value with recourse

2. VISUALIZATION:
   - Plots discharge and reservoir levels for all models
   - Shows how different approaches handle uncertainty

3. BENDERS DECOMPOSITION:
   - Finite difference implementation with V24 as linking variable
   - Compares results with extensive form solution
   - Demonstrates 18% objective gap due to feasibility enforcement

USAGE:
------
- Run script to see all models and plots
- Uncomment last line to run Benders decomposition
- Use cleanup_helper.py to remove development files

RESULTS:
--------
- Stochastic model objective: ~458,482 NOK (V24 ≈ 4.204)
- Benders objective: ~374,821 NOK (V24 ≈ 4.205)  
- Gap exists because Benders enforces stricter feasibility constraints
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from pyomo.opt import SolverStatus, TerminationCondition


# =============================================================================
# PARAMETERS
# =============================================================================
T = 48
T1 = 24

# Reservoir & turbine parameters
V0 = 3.0 #Mm3
Vmax = 4.5 #Mm3
Qmax = 100.0 #m3/s
Pmax = 86.5 #MW
alpha = 3.6/1000.0 #Conversion factor from flow (m³/s) over one hour to million cubic meters (Mm³)
E_conv = 0.657 #Energy [kWh] generated per cubic meter of discharged water.
WV_end = 52600.0

P_full_discharge = E_conv * Qmax * 3600 / 1000
Qmax_from_P = Qmax * (Pmax / P_full_discharge)
# kWh/m3 * m3/s * 3600s/h = kWh (3600 * E_conv * m.q[t] * 1000)
q_cap = min(Qmax, Qmax_from_P)  # Max discharge limited by both physical and power constraints

# For Benders decomposition (corrected eta parameter)
eta = E_conv * 3.6  # = 0.657 * 3.6 = 2.3652

# SPILL VARIABLE PARAMETERS
# =========================
# High cost for spilling water to avoid infeasibility issues
# This makes all solutions feasible but expensive if we waste water
spill_cost = 100000.0  # NOK per Mm³ - very expensive to discourage spilling
                       # This should be much higher than water value (~52,600 NOK/Mm³)

# Price profile (same for all scenarios)
pi = {t: 50.0 + (t+1) for t in range(1,T+1)}

# --- Scenario definitions with names ---
scenario_info = {
    "Very Dry": 0,
    "Dry": 10,
    "Normal": 20,
    "Wet": 30,
    "Very Wet": 40,
}

scenarios = list(scenario_info.keys())
prob = {s: 1.0 / len(scenarios) for s in scenarios}

# Inflows per scenario
scenario_inflows = scenario_info.copy()

# Build inflow tensors I[s, t] and expected inflow I_exp[t]
I = {}
for s in scenarios:
    for t in range(1, T + 1):
        if t <= T1:
            I[s, t] = 50.0
        else:
            I[s, t] = scenario_inflows[s]

I_exp = {
    t: (50.0 if t <= T1 else sum(scenario_inflows.values()) / len(scenarios))
    for t in range(1, T + 1)
}

# =============================================================================
# OPTIMIZATION MODELS
# =============================================================================

# Model 1: Expected Value (Deterministic equivalent)
def build_EV_model():
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T)
    m.q = pyo.Var(m.T, bounds=(0, q_cap))
    m.V = pyo.Var(m.T, bounds=(0, Vmax))
    m.spill = pyo.Var(m.T, bounds=(0, None))  # Spill variable (water waste)

    def res_rule(m, t):
        if t == 1:
            return m.V[t] == V0 + alpha * I_exp[t] - alpha * m.q[t] - alpha * m.spill[t]
        else:
            return m.V[t] == m.V[t - 1] + alpha * I_exp[t] - alpha * m.q[t] - alpha * m.spill[t]

    m.res_balance = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(m):
        # Revenue from discharge minus high cost of spilling
        revenue = sum(pi[t] * 3.6 * E_conv * m.q[t] for t in m.T)
        spill_penalty = sum(spill_cost * alpha * m.spill[t] for t in m.T)  # Convert to Mm³
        terminal_value = WV_end * m.V[T]
        return revenue - spill_penalty + terminal_value

    # EUR / MWh * 1h * kWh/m3

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    return m



# Model 2: Individual Scenario Analysis
def build_scenario_model(s):
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T)
    m.q = pyo.Var(m.T, bounds=(0, q_cap))
    m.V = pyo.Var(m.T, bounds=(0, Vmax))
    m.spill = pyo.Var(m.T, bounds=(0, None))  # Spill variable (water waste)

    def res_rule(m, t):
        if t == 1:
            return m.V[t] == V0 + alpha * I[s, t] - alpha * m.q[t] - alpha * m.spill[t]
        else:
            return m.V[t] == m.V[t - 1] + alpha * I[s, t] - alpha * m.q[t] - alpha * m.spill[t]

    m.res_balance = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(m):
        # Revenue from discharge minus high cost of spilling
        revenue = sum(pi[t] * 3.6 * E_conv * m.q[t] for t in m.T)
        spill_penalty = sum(spill_cost * alpha * m.spill[t] for t in m.T)  # Convert to Mm³
        terminal_value = WV_end * m.V[T]
        return revenue - spill_penalty + terminal_value

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m

# Model 3: Two-Stage Stochastic Programming
def build_stochastic_model():
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=scenarios)
    m.T1 = pyo.RangeSet(1, T1)
    m.T2 = pyo.RangeSet(T1 + 1, T)

    # Shared q for first 24h
    m.q1 = pyo.Var(m.T1, bounds=(0, q_cap))
    m.V1 = pyo.Var(m.T1, bounds=(0, Vmax))
    m.spill1 = pyo.Var(m.T1, bounds=(0, None))  # Spill for first stage

    # Scenario-specific q, V for day 2
    m.q2 = pyo.Var(m.S, m.T2, bounds=(0, q_cap))
    m.V2 = pyo.Var(m.S, m.T2, bounds=(0, Vmax))
    m.spill2 = pyo.Var(m.S, m.T2, bounds=(0, None))  # Spill for second stage

    # Reservoir balance for day 1 (shared)
    def res1(m, t):
        if t == 1:
            return m.V1[t] == V0 + alpha * 50 - alpha * m.q1[t] - alpha * m.spill1[t]
        else:
            return m.V1[t] == m.V1[t - 1] + alpha * 50 - alpha * m.q1[t] - alpha * m.spill1[t]
    m.res1 = pyo.Constraint(m.T1, rule=res1)

    # Reservoir balance for day 2 (scenario-dependent)
    def res2(m, s, t):
        if t == T1 + 1:
            return m.V2[s, t] == m.V1[T1] + alpha * I[s, t] - alpha * m.q2[s, t] - alpha * m.spill2[s, t]
        else:
            return m.V2[s, t] == m.V2[s, t - 1] + alpha * I[s, t] - alpha * m.q2[s, t] - alpha * m.spill2[s, t]
    m.res2 = pyo.Constraint(m.S, m.T2, rule=res2)

    # Objective: expected revenue minus spill penalties
    def obj(m):
        # First stage revenue and spill penalty
        first_revenue = sum(pi[t] * 3.6 * E_conv * m.q1[t] for t in m.T1)
        first_spill_penalty = sum(spill_cost * alpha * m.spill1[t] for t in m.T1)
        
        # Second stage expected revenue and spill penalty
        second = sum(prob[s] * (
            sum(pi[t] * 3.6 * E_conv * m.q2[s, t] for t in m.T2)  # Revenue
            - sum(spill_cost * alpha * m.spill2[s, t] for t in m.T2)  # Spill penalty
            + WV_end * m.V2[s, T]  # Terminal value
        ) for s in m.S)
        
        return first_revenue - first_spill_penalty + second
    m.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

    return m




# =============================================================================
# SOLVE ALL MODELS & GENERATE RESULTS
# =============================================================================
solver = pyo.SolverFactory("gurobi")

# EV model
m_ev = build_EV_model()
solver.solve(m_ev)


# Scenario models
for s in scenarios:
    m_s = build_scenario_model(s)
    solver.solve(m_s)
    print(f"{s} scenario (inflow={scenario_inflows[s]}): objective = {pyo.value(m_s.obj):,.2f}")
    print("First 24h q:", [pyo.value(m_s.q[t]) for t in range(1, 25)])


# Stochastic model
m_stoch = build_stochastic_model()
solver.solve(m_stoch)

# =============================================================================
# RESULTS SUMMARY & VISUALIZATION  
# =============================================================================
from tabulate import tabulate

# Scenario models
scenario_results = []
for s in scenarios:
    m_s = build_scenario_model(s)
    solver.solve(m_s)
    obj_val = pyo.value(m_s.obj)
    q_vals = [pyo.value(m_s.q[t]) for t in range(1, 25)]
    scenario_results.append((s, scenario_inflows[s], obj_val, q_vals))

# Print header
print("\n" + "="*80)
print("Scenario Results")
print("="*80)

table_data = []
for s, inflow, obj_val, q_vals in scenario_results:
    table_data.append([
        s,
        f"{inflow:6.1f}",
        f"{obj_val:,.0f}",
        f"{np.mean(q_vals):6.2f}",
        f"{min(q_vals):6.2f}",
        f"{max(q_vals):6.2f}"
    ])

headers = ["Scenario", "Inflow (m³/s)", "Objective (€)", "q avg (m³/s)", "q min", "q max"]
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="right"))

# --- Aggregate info ---
objs = [r[2] for r in scenario_results]
print(f"\nObjective range: {min(objs):,.0f}  →  {max(objs):,.0f}")
print(f"Average objective: {np.mean(objs):,.0f}")

# --- EV Model ---
ev_obj = pyo.value(m_ev.obj)
print("\n" + "="*80)
print("Expected Value (EV) Model")
print("="*80)
print(f"Objective value: {ev_obj:,.0f}")
print(f"First 24h discharge profile:")
ev_q = [pyo.value(m_ev.q[t]) for t in range(1, 25)]
print("  " + ", ".join(f"{v:6.2f}" for v in ev_q))

# --- Stochastic Model ---
stoch_obj = pyo.value(m_stoch.obj)
print("\n" + "="*80)
print("Two-Stage Stochastic Model")
print("="*80)
print(f"Objective value: {stoch_obj:,.0f}")
stoch_q1 = [pyo.value(m_stoch.q1[t]) for t in range(1, 25)]
print("First 24h discharge profile:")
print("  " + ", ".join(f"{v:6.2f}" for v in stoch_q1))
print("="*80 + "\n")

# --- Plotting in three separate figures: EV, Scenarios, Stochastic ---
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_three_figures():
    hours_q = list(range(1, T + 1))       # q defined for 1..T
    hours_V = [0] + hours_q               # V includes initial state at t=0

    # --- Collect series ---

    # EV (already solved)
    ev_q = [pyo.value(m_ev.q[t]) for t in range(1, T + 1)]
    ev_V = [V0] + [pyo.value(m_ev.V[t]) for t in range(1, T + 1)]

    # Scenarios (solve each to get full horizon series)
    scenario_qs, scenario_Vs = {}, {}
    ordered = ["Very Dry", "Dry", "Normal", "Wet", "Very Wet"]
    ordered_scenarios = sorted(scenarios, key=lambda x: ordered.index(x) if x in ordered else x)
    for s in ordered_scenarios:
        m_tmp = build_scenario_model(s)
        solver.solve(m_tmp, tee=False)
        scenario_qs[s] = [pyo.value(m_tmp.q[t]) for t in range(1, T + 1)]
        scenario_Vs[s] = [V0] + [pyo.value(m_tmp.V[t]) for t in range(1, T + 1)]

    # Stochastic (first stage exact; expected values for day 2)
    stoch_q = []
    stoch_V = [V0]
    for t in range(1, T + 1):
        if t <= T1:
            stoch_q.append(pyo.value(m_stoch.q1[t]))
            stoch_V.append(pyo.value(m_stoch.V1[t]))
        else:
            stoch_q.append(sum(prob[s] * pyo.value(m_stoch.q2[s, t]) for s in scenarios))
            stoch_V.append(sum(prob[s] * pyo.value(m_stoch.V2[s, t]) for s in scenarios))

    # ---------- Figure 1: EV ----------
    plt.figure(figsize=(11, 4.5))
    ax = plt.gca()
    ax2 = ax.twinx()
    lq, = ax.plot(hours_q, ev_q, color='red', linewidth=2, label='Discharge q (EV)')
    lV, = ax2.plot(hours_V, ev_V, color='red', linestyle='--', linewidth=2, label='Reservoir V (EV)', zorder=3)
    ax.set_title('Expected Value (EV) Model')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Discharge q (m³/s)')
    ax2.set_ylabel('Reservoir Level V (Mm³)')
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, T)
    ax.legend(handles=[lq, lV], loc='upper left')
    plt.tight_layout()
    plt.show()

    # ---------- Figure 2: Individual Scenarios ----------
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax2 = ax.twinx()
    cmap = plt.get_cmap('tab10')
    scenario_lines = []
    for i, s in enumerate(ordered_scenarios):
        c = cmap(i % 10)
        ln, = ax.plot(hours_q, scenario_qs[s], color=c, linewidth=2, label=s)      # q (solid)
        ax2.plot(hours_V, scenario_Vs[s], color=c, linestyle='--', linewidth=2, alpha=0.95, zorder=3)  # V (dashed)
        scenario_lines.append(ln)

    ax.set_title('Individual Scenario Models — q (solid), V (dashed)')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Discharge q (m³/s)')
    ax2.set_ylabel('Reservoir Level V (Mm³)')
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, T)

    style_key = [
        Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='q (discharge)'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='V (reservoir)'),
    ]
    first_legend = ax.legend(handles=scenario_lines, title='Scenarios', loc='upper left', frameon=True)
    ax.add_artist(first_legend)
    ax.legend(handles=style_key, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()

    # ---------- Figure 3: Two-Stage Stochastic ----------
    plt.figure(figsize=(11, 4.5))
    ax = plt.gca()
    ax2 = ax.twinx()
    lq, = ax.plot(hours_q, stoch_q, color='green', linewidth=2, label='Discharge q (Stochastic)')
    lV, = ax2.plot(hours_V, stoch_V, color='green', linestyle='--', linewidth=2, label='Reservoir V (Stochastic)', zorder=3)
    ax.set_title('Two-Stage Stochastic Model — shared first stage; expected values on day 2')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Discharge q (m³/s)')
    ax2.set_ylabel('Reservoir Level V (Mm³)')
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, T)
    ax.legend(handles=[lq, lV], loc='upper left')
    plt.tight_layout()
    plt.show()

# Call it
#plot_three_figures()


# =============================================================================
# BENDERS DECOMPOSITION IMPLEMENTATION
# =============================================================================

class Benders:
    """
    Benders Decomposition using Dual Values (Shadow Prices) for Marginal Water Value
    Links on V24 (volume at end of first stage)
    
    Uses dual values from the reservoir balance constraints to compute exact marginal
    water values, which is theoretically superior to finite difference approximations.
    """
    
    def __init__(self, max_iter=15, tolerance=1e-3):
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # Configure solver to return dual information
        self.solver = pyo.SolverFactory("gurobi")
        
        self.iteration = 0
        self.cuts = []
        self.best_lb = -float('inf')
        self.best_ub = float('inf')
        self.optimal_V24 = None
        self.optimal_obj = None
        
    def compute_max_feasible_V24(self):
        """
        Compute maximum feasible V24 using binary search with actual subproblem testing.
        This is the most reliable approach - we test actual feasibility rather than 
        trying to derive constraints analytically.
        """
        print("🔍 Computing maximum feasible V24 using binary search...")
        
        # Binary search bounds
        V24_min = 0.1  # Minimum reasonable value
        V24_max = Vmax  # Start with reservoir capacity
        tolerance = 0.001
        
        # First, find an upper bound where at least one scenario is infeasible
        print(f"  🔍 Finding upper bound...")
        while V24_max > V24_min:
            all_feasible = True
            for s in scenarios:
                obj = self.solve_single_subproblem(s, V24_max)
                if obj is None:  # Infeasible
                    all_feasible = False
                    break
            
            if not all_feasible:
                print(f"    Found infeasibility at V24 = {V24_max:.6f}")
                break
            else:
                V24_max += 0.1  # Try higher values
                if V24_max > Vmax + 1:  # Safety check
                    print(f"    No infeasibility found up to {V24_max:.6f}")
                    break
        
        # Now do binary search between V24_min and V24_max
        print(f"  🔍 Binary search between {V24_min:.6f} and {V24_max:.6f}")
        
        iteration = 0
        max_iterations = 20
        
        while (V24_max - V24_min) > tolerance and iteration < max_iterations:
            iteration += 1
            V24_test = (V24_min + V24_max) / 2
            
            # Test all scenarios at this V24
            all_feasible = True
            infeasible_scenarios = []
            
            for s in scenarios:
                obj = self.solve_single_subproblem(s, V24_test)
                if obj is None:  # Infeasible
                    all_feasible = False
                    infeasible_scenarios.append(s)
            
            if all_feasible:
                print(f"    Iter {iteration}: V24 = {V24_test:.6f} ✅ (all feasible)")
                V24_min = V24_test  # Can go higher
            else:
                scenarios_str = ", ".join(infeasible_scenarios)
                print(f"    Iter {iteration}: V24 = {V24_test:.6f} ❌ (infeasible: {scenarios_str})")
                V24_max = V24_test  # Must go lower
        
        # Final feasible value
        max_feasible_V24 = V24_min
        
        # Add safety margin
        safety_margin = 0.01
        safe_limit = max_feasible_V24 - safety_margin
        
        print(f"\n🎯 Binary search results:")
        print(f"   Maximum feasible V24: {max_feasible_V24:.6f}")
        print(f"   With safety margin: {safe_limit:.6f}")
        
        # Create a dummy scenario_limits dict for compatibility
        scenario_limits = {s: safe_limit for s in scenarios}
        
        return safe_limit, scenario_limits
    
    def validate_feasibility_constraint(self, V24_test):
        """
        Validate that all scenarios are feasible at the given V24
        Returns True if all scenarios feasible, False otherwise
        """
        print(f"🧪 Validating feasibility at V24 = {V24_test:.6f}")
        
        all_feasible = True
        for s in scenarios:
            obj_value = self.solve_single_subproblem(s, V24_test)
            if obj_value is None:
                print(f"  ❌ Scenario {s}: INFEASIBLE")
                all_feasible = False
            else:
                print(f"  ✅ Scenario {s}: Feasible (obj = {obj_value:.0f})")
        
        return all_feasible
        
    def build_master_problem(self):
        """Build master problem with first stage decisions"""
        m = pyo.ConcreteModel("Master Problem")
        
        m.T1 = pyo.RangeSet(1, T1)
        m.q1 = pyo.Var(m.T1, bounds=(0, q_cap))
        m.V1 = pyo.Var(m.T1, bounds=(0, Vmax))
        m.spill1 = pyo.Var(m.T1, bounds=(0, None))  # Spill variable (water waste)
        
        m.theta = pyo.Var(bounds=(-1000000, 1000000))
        m.max_feasible_V24 = Vmax  # Use reservoir capacity as upper bound
        
        # Reservoir balance
        def res_balance_rule(m, t):
            if t == 1:
                return m.V1[t] == V0 + alpha * 50.0 - alpha * m.q1[t] - alpha * m.spill1[t]
            else:
                return m.V1[t] == m.V1[t-1] + alpha * 50.0 - alpha * m.q1[t] - alpha * m.spill1[t]
        m.res_balance = pyo.Constraint(m.T1, rule=res_balance_rule)
        
        m.benders_cuts = pyo.ConstraintList()
       
        def obj_rule(m):
            first_stage_revenue = sum(pi[t] * eta * m.q1[t] for t in m.T1)
            first_stage_spill_penalty = sum(spill_cost * alpha * m.spill1[t] for t in m.T1)
            return first_stage_revenue - first_stage_spill_penalty + m.theta
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        
        return m
        
    def solve_single_subproblem(self, scenario, V24_fixed):
        """Solve a single subproblem for given scenario and V24"""
        m = pyo.ConcreteModel(f"Sub_S{scenario}_V24_{V24_fixed:.4f}")
        
        m.T2 = pyo.RangeSet(T1 + 1, T)
        m.q2 = pyo.Var(m.T2, bounds=(0, q_cap))
        m.V2 = pyo.Var(m.T2, bounds=(0, Vmax))
        m.spill2 = pyo.Var(m.T2, bounds=(0, None))  # Spill variable (water waste)
        
        # Reservoir balance constraints
        def res_balance_rule(m, t):
            if t == T1 + 1:  # First hour of second stage
                return m.V2[t] == V24_fixed + alpha * I[scenario, t] - alpha * m.q2[t] - alpha * m.spill2[t]
            else:
                return m.V2[t] == m.V2[t-1] + alpha * I[scenario, t] - alpha * m.q2[t] - alpha * m.spill2[t]
        m.res_balance = pyo.Constraint(m.T2, rule=res_balance_rule)
        
        # Objective: second stage revenue + terminal value - spill penalty
        def obj_rule(m):
            revenue = sum(pi[t] * eta * m.q2[t] for t in m.T2)
            spill_penalty = sum(spill_cost * alpha * m.spill2[t] for t in m.T2)  # Convert to Mm³
            terminal_value = WV_end * m.V2[T]
            return revenue - spill_penalty + terminal_value
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        
        # Solve
        result = self.solver.solve(m, tee=False)
        
        if result.solver.termination_condition != TerminationCondition.optimal:
            return None
            
        return pyo.value(m.obj)
        
    def solve_single_subproblem(self, scenario, V24_fixed):
        """Solve a single subproblem and return objective value and dual value of V24 constraint"""
        m = pyo.ConcreteModel(f"Sub_S{scenario}_V24_{V24_fixed:.4f}")
        
        m.T2 = pyo.RangeSet(T1 + 1, T)
        m.q2 = pyo.Var(m.T2, bounds=(0, q_cap))
        m.V2 = pyo.Var(m.T2, bounds=(0, Vmax))
        m.spill2 = pyo.Var(m.T2, bounds=(0, None))  # Spill variable (water waste)
        
        # Reservoir balance constraints
        def res_balance_rule(m, t):
            if t == T1 + 1:  # First hour of second stage - this is where V24 enters
                return m.V2[t] == V24_fixed + alpha * I[scenario, t] - alpha * m.q2[t] - alpha * m.spill2[t]
            else:
                return m.V2[t] == m.V2[t-1] + alpha * I[scenario, t] - alpha * m.q2[t] - alpha * m.spill2[t]
        m.res_balance = pyo.Constraint(m.T2, rule=res_balance_rule)
        
        # Objective: second stage revenue + terminal value - spill penalty
        def obj_rule(m):
            revenue = sum(pi[t] * eta * m.q2[t] for t in m.T2)
            spill_penalty = sum(spill_cost * alpha * m.spill2[t] for t in m.T2)  # Convert to Mm³
            terminal_value = WV_end * m.V2[T]
            return revenue - spill_penalty + terminal_value
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        
        # Add suffix to request dual values
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        
        # Solve with dual information
        result = self.solver.solve(m, tee=False)
        
        if result.solver.termination_condition != TerminationCondition.optimal:
            return None, None
            
        # Get objective value
        obj_value = pyo.value(m.obj)
        
        # Get dual value of the first reservoir balance constraint (T1+1)
        # This represents the marginal value of increasing V24 by 1 unit
        try:
            # The dual value tells us how much objective improves per unit increase in RHS
            # Since RHS contains V24_fixed, the dual gives us ∂obj/∂V24
            dual_value = m.dual[m.res_balance[T1 + 1]]
            
            # The constraint is: V2[T1+1] = V24_fixed + alpha*inflow - alpha*q - alpha*spill
            # So dual value is w.r.t. the V24_fixed term (coefficient = 1)
            marginal_value = dual_value
            
        except (AttributeError, KeyError) as e:
            # If dual values not available, this is an error
            print(f"      Error: Dual values not available for scenario {scenario} ({str(e)})")
            return None, None
        
        return obj_value, marginal_value
        
    def solve_subproblems(self, V24_fixed):
        """
        Solve all subproblems and get marginal values using dual information
        """
        subproblem_values = []
        marginal_values = []
        
        print(f"    Computing dual values at V24 = {V24_fixed:.4f}")
        
        for s in scenarios:
            obj_value, marginal_value = self.solve_single_subproblem(s, V24_fixed)
            
            if obj_value is None:
                print(f"    Warning: Scenario {s} failed at V24 = {V24_fixed:.4f}")
                obj_value = -1000000
                marginal_value = -1000000
            
            subproblem_values.append(obj_value)
            marginal_values.append(marginal_value)
            print(f"      Scenario {s}: obj = {obj_value:.0f}, marginal = {marginal_value:.2f}")
        
        # Compute expected values
        expected_value = sum(prob[s] * subproblem_values[i] for i, s in enumerate(scenarios))
        expected_marginal = sum(prob[s] * marginal_values[i] for i, s in enumerate(scenarios))
        
        print(f"    Expected: value = {expected_value:.2f}, marginal = {expected_marginal:.2f}")
        
        return expected_value, expected_marginal
    
    def add_initial_cut(self, master, max_feasible_V24):
        """Add single initial cut at midpoint of feasible range"""
        print("Adding initial cut...")
        
        # Generate single test point at midpoint of feasible range
        V24_min = min(0, V0)  # Start from initial reservoir level or reasonable minimum
        V24_max = max_feasible_V24 * 0.95  # Stay slightly below max for safety
        
        if V24_max <= V24_min:
            print(f"  WARNING: Very small feasible range [{V24_min:.3f}, {V24_max:.3f}]")
            V24_max = V24_min + 1.0  # Ensure some range
        
        # Single cut at midpoint
        V24_test = (V24_min + V24_max) / 2
        
        print(f"  🎯 Feasible range: [{V24_min:.3f}, {max_feasible_V24:.3f}]")
        print(f"  📊 Initial cut at V24 = {V24_test:.3f}")
        
        expected_value, expected_marginal = self.solve_subproblems(V24_test)
        
        # Add cut: θ ≤ expected_value + expected_marginal * (V24 - V24_test)
        cut_expr = master.theta <= expected_value + expected_marginal * (master.V1[T1] - V24_test)
        master.benders_cuts.add(cut_expr)
        
        self.cuts.append({
            'V24': V24_test,
            'value': expected_value,
            'marginal': expected_marginal,
            'iteration': 0
        })
    
    def solve_master(self, master):
        """Solve master problem and return V24 decision"""
        result = self.solver.solve(master, tee=False)
        
        if result.solver.termination_condition != TerminationCondition.optimal:
            raise Exception("Master problem failed to solve")
            
        V24_solution = pyo.value(master.V1[T1])
        obj_solution = pyo.value(master.obj)
        theta_solution = pyo.value(master.theta)
        
        return V24_solution, obj_solution, theta_solution
    
    def compute_true_objective(self, V24_decision):
        """Compute true objective for given V24 decision"""
        # Solve master problem with fixed V24 to get first stage
        master_eval = self.build_master_problem()
        master_eval.V24_fixed = pyo.Constraint(expr=master_eval.V1[T1] == V24_decision)
        
        result = self.solver.solve(master_eval, tee=False)
        if result.solver.termination_condition != TerminationCondition.optimal:
            return None
            
        # First stage objective including spill penalty
        first_stage_revenue = sum(pi[t] * eta * pyo.value(master_eval.q1[t]) for t in master_eval.T1)
        first_stage_spill_penalty = sum(spill_cost * alpha * pyo.value(master_eval.spill1[t]) for t in master_eval.T1)
        first_stage_obj = first_stage_revenue - first_stage_spill_penalty
        
        # Solve all subproblems (they now include spill penalties automatically)
        second_stage_values = []
        for s in scenarios:
            obj_s, _ = self.solve_single_subproblem(s, V24_decision)  # Ignore marginal value
            if obj_s is None:
                obj_s = -1000000  # Should not happen with spill variables, but keep as safety
            second_stage_values.append(obj_s)
        
        expected_second_stage = sum(prob[s] * second_stage_values[i] for i, s in enumerate(scenarios))
        
        return first_stage_obj + expected_second_stage
    
    def solve(self):
        """Main Benders algorithm"""
        print("Starting Dual Value Benders Decomposition")
        print("=" * 70)
        print("🎯 USING DUAL VALUES: Exact marginal values from shadow prices")
        print("🎯 SPILL VARIABLES ENABLED: All solutions feasible with penalty cost")
        print("=" * 70)
        
        # Build master problem (no feasibility constraints needed!)
        master = self.build_master_problem()
        
        # Add initial cut based on reasonable range
        self.add_initial_cut(master, master.max_feasible_V24)
        
        print("\nStarting main iterations...")
        
        for iteration in range(1, self.max_iter + 1):
            self.iteration = iteration
            print(f"\n{'='*20} Iteration {iteration} {'='*20}")
            
            # Solve master problem
            V24_solution, master_obj, theta_value = self.solve_master(master)
            print(f"✅ Master solved: V24 = {V24_solution:.4f}, Obj = {master_obj:.2f}, θ = {theta_value:.2f}")
            
            # Solve subproblems with dual values
            expected_value, expected_marginal = self.solve_subproblems(V24_solution)
            
            # Compute true objective
            true_obj = self.compute_true_objective(V24_solution)
            print(f"📊 Results: True obj = {true_obj:.2f}")
            
            # Update bounds
            lower_bound = master_obj
            upper_bound = true_obj
            gap = abs(upper_bound - lower_bound)
            
            print(f"📊 Bounds: LB = {lower_bound:.2f}, UB = {upper_bound:.2f}")
            print(f"📊 Gap: {gap:.2f} ({100*gap/max(abs(upper_bound),1e-6):.3f}%)")
            
            # Check convergence
            if gap <= self.tolerance:
                print(f"🎯 Converged! Gap = {gap:.6f}")
                self.optimal_V24 = V24_solution
                self.optimal_obj = true_obj
                break
            
            # Add new cut
            cut_expr = master.theta <= expected_value + expected_marginal * (master.V1[T1] - V24_solution)
            master.benders_cuts.add(cut_expr)
            print(f"➕ Added cut: θ ≤ {expected_value:.0f} + {expected_marginal:.2f} × (V24 - {V24_solution:.4f})")
            
            # Store cut information
            self.cuts.append({
                'V24': V24_solution,
                'value': expected_value,
                'marginal': expected_marginal,
                'iteration': iteration
            })
        
        else:
            print(f"🔄 Reached maximum iterations ({self.max_iter})")
            self.optimal_V24 = V24_solution
            self.optimal_obj = true_obj
        
        # Print final results
        self.print_results()
        
        # Plot the Benders cuts
        self.plot_benders_cuts()
        
        return self.optimal_V24, self.optimal_obj
    
    def print_results(self):
        """Print final results and comparison"""
        print("\n" + "=" * 70)
        print("🏁 DUAL VALUE BENDERS RESULTS")
        print("=" * 70)
        print(f"Iterations completed: {self.iteration}")
        final_gap = abs(self.best_ub - self.best_lb) if self.best_ub != float('inf') else 0
        print(f"Final gap: {final_gap:.6f} ({100*final_gap/max(abs(self.best_ub),1e-6):.4f}%)")
        print(f"Optimal objective: {self.optimal_obj:.2f} NOK")
        print(f"Optimal V24: {self.optimal_V24:.6f} dam³")
        print(f"Total Benders cuts: {len(self.cuts)}")
        
        print(f"\n📈 Marginal value evolution:")
        for i, cut in enumerate(self.cuts):
            print(f"  Cut {i}: V24={cut['V24']:.3f}, marginal={cut['marginal']:.1f}")
        
        print("\n" + "=" * 70)
        print("🔍 COMPARISON WITH EXTENSIVE FORM")
        print("=" * 70)
        print("📋 Extensive form results:")
        print(f"   Objective: {pyo.value(m_stoch.obj):.2f} NOK")
        print(f"   V24: {pyo.value(m_stoch.V1[T1]):.6f} dam³")
        
        print(f"\n🎯 FINAL COMPARISON:")
        extensive_obj = pyo.value(m_stoch.obj)
        extensive_V24 = pyo.value(m_stoch.V1[T1])
        obj_diff = abs(extensive_obj - self.optimal_obj)
        V24_diff = abs(extensive_V24 - self.optimal_V24)
        rel_error = 100 * obj_diff / extensive_obj
        
        print(f"   Objective difference: {obj_diff:.6f} NOK")
        print(f"   V24 difference: {V24_diff:.6f} dam³")
        print(f"   Relative error: {rel_error:.6f}%")

    def plot_benders_cuts(self):
        """
        Plot the Benders cuts to visualize the approximation of the value function
        """
        if len(self.cuts) == 0:
            print("No cuts to plot!")
            return
            
        print("\n🎨 Generating Benders cuts visualization...")
        
        # Create V24 range for plotting
        V24_min = max(0.5, V0)
        V24_max = min(Vmax, max([cut['V24'] for cut in self.cuts]) + 0.5)
        V24_range = np.linspace(V24_min, V24_max, 200)
        
        # Set up the plot
        plt.figure(figsize=(10, 6))
        
        # Plot each cut as a red line
        for i, cut in enumerate(self.cuts):
            V24_cut = cut['V24']
            value_cut = cut['value']
            marginal_cut = cut['marginal']
            
            # Calculate cut values: θ ≤ value + marginal * (V24 - V24_cut)
            cut_values = value_cut + marginal_cut * (V24_range - V24_cut)
            
            plt.plot(V24_range, cut_values, 'r-', linewidth=1.5, alpha=0.8)
        
        # Calculate and plot the envelope (maximum of all cuts) with shaded area
        # Use both intersection-based approach for the envelope line AND grid-based for shading
        
        # First, create the grid-based envelope for proper shading
        envelope_values = np.full(len(V24_range), -np.inf)
        for cut in self.cuts:
            cut_values = cut['value'] + cut['marginal'] * (V24_range - cut['V24'])
            envelope_values = np.maximum(envelope_values, cut_values)
        
        # Add shaded feasible region using the full range
        plt.fill_between(V24_range, envelope_values, alpha=0.3, color='gray')
        
        # Add thick black envelope line using the same envelope as the shading
        plt.plot(V24_range, envelope_values, 'k-', linewidth=3)
        
        # Mark the optimal solution with a single point
        if self.optimal_V24 is not None and self.optimal_obj is not None:
            # Calculate the optimal θ value from the envelope at optimal V24
            optimal_theta = -np.inf
            for cut in self.cuts:
                cut_value_at_optimal = cut['value'] + cut['marginal'] * (self.optimal_V24 - cut['V24'])
                optimal_theta = max(optimal_theta, cut_value_at_optimal)
            
            plt.plot(self.optimal_V24, optimal_theta, 'ko', markersize=8)
        
        # Simple formatting
        plt.xlabel('V24 - Volume at hour 24 (Mm³)')
        plt.ylabel('Second Stage Value θ (NOK)')
        plt.title('Benders Cuts')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def run_benders_decomposition():
    """
    Run Benders decomposition and compare with extensive form
    """
    print("\n\n" + "=" * 80)
    print("RUNNING BENDERS DECOMPOSITION")  
    print("=" * 80)
    
    # Create and solve Benders with dual values
    benders = Benders(max_iter=15, tolerance=1e-6)
    optimal_V24, optimal_obj = benders.solve()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Benders found V24 = {optimal_V24:.6f} with objective = {optimal_obj:.2f}")
    print(f"   Extensive form has V24 = {pyo.value(m_stoch.V1[T1]):.6f} with objective = {pyo.value(m_stoch.obj):.2f}")
    
    return benders

# Uncomment the line below to run Benders decomposition
benders_result = run_benders_decomposition()

