import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# --- Parameters ---
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

# Expected inflow
I_exp = {
    t: (50.0 if t <= T1 else sum(scenario_inflows.values()) / len(scenarios))
    for t in range(1, T + 1)
}

# Global variable to hold fixed V24 from master problem
V24_fixed = 0.0  # Initialize to 0.0

# --- Master Problem ---

# Create master problem
master = pyo.ConcreteModel()

# Sets
master.T = pyo.RangeSet(1, T1)

# Variables
master.q = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, q_cap), initialize=0.0) # Discharge [m3/s]
master.V = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, Vmax), initialize=V0) # Reservoir volume [Mm3]
master.theta = pyo.Var(within=pyo.Reals, bounds=(-1e6, 1e6), initialize=0.0) # Approximation of expected future cost with bounds

# Reservoir balance constraint
def master_res_rule(m, t):
    if t == 1:
        return m.V[t] == V0 + alpha * I_exp[t] - alpha * m.q[t]
    else:
        return m.V[t] == m.V[t-1] + alpha * I_exp[t] - alpha * m.q[t]

# Connect constraint to model
master.res_balance = pyo.Constraint(master.T, rule=master_res_rule)

# Objective function
def master_obj_rule(m):
    return sum(pi[t] * E_conv * m.q[t] for t in m.T) + m.theta

# Connect objective to model
master.obj = pyo.Objective(rule=master_obj_rule, sense=pyo.maximize)

# --- Subproblem ---

# Create subproblem
sub = pyo.ConcreteModel()

# Sets
sub.T = pyo.RangeSet(T1+1, T)
sub.S = pyo.Set(initialize=scenarios)

# Variables with initialization
sub.q = pyo.Var(sub.T, within=pyo.NonNegativeReals, bounds=(0, q_cap), initialize=0.0) # Discharge [m3/s]
sub.V = pyo.Var(sub.T, within=pyo.NonNegativeReals, bounds=(0, Vmax), initialize=V0) # Reservoir volume [Mm3]

# Suffix to capture dual values
sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Reservoir balance constraint for linking with master problem
def V25_rule(m, s):
    print(f"Creating V25 constraint for scenario {s}")
    print(f"V24_fixed={V24_fixed}, inflow={alpha * I[s, T1+1]}")
    # Calculate max feasible discharge to maintain non-negative volume
    max_feasible_discharge = (V24_fixed + alpha * I[s, T1+1]) / alpha
    # Update the upper bound of q[T1+1] for this scenario
    m.q[T1+1].setub(min(q_cap, max_feasible_discharge))
    return m.V[T1+1] == V24_fixed + alpha * I[s, T1+1] - alpha * m.q[T1+1]
sub.V25 = pyo.Constraint(sub.S, rule=V25_rule)

# Reservoir balance constraint
def sub_res_rule(m, s, t):
    if t == T1 + 1:
        return pyo.Constraint.Skip
    return m.V[t] == m.V[t-1] + alpha * I[s, t] - alpha * m.q[t]
sub.res_balance = pyo.Constraint(sub.S, sub.T, rule=sub_res_rule)

# Objective function
def sub_obj_rule(m, s):
    return sum(pi[t] * E_conv * m.q[t] for t in m.T) + (WV_end * m.V[T] if T in m.T else 0)
sub.obj = pyo.Objective(rule=sub_obj_rule, sense=pyo.maximize)

# --- Benders Decomposition Functions ---

opt = pyo.SolverFactory("gurobi")
cut_counter = 1
upper_bounds = float('inf')
lower_bounds = -float('inf')
tolerance = 1e-4
iteration = 1

# Lists to store upper and lower bounds for plotting
upper_bound_list = []
lower_bound_list = []

while abs(upper_bounds - lower_bounds) > tolerance:
    print(f"--- Benders Iteration {iteration} ---")
    
    # Solve master problem
    master_result = opt.solve(master)
    
    if master_result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"Master problem failed to solve optimally. Status: {master_result.solver.termination_condition}")
        break
        
    master_V24 = pyo.value(master.V[T1])
    master_theta = pyo.value(master.theta)
    master_obj_value = pyo.value(master.obj)
    
    print(f"Master Problem Objective: {master_obj_value}")
    print(f"V24 from Master: {master_V24}")
    
    # Update fixed V24 for subproblem
    V24_fixed = master_V24
    
    expected_sub_obj = 0.0
    duals_V25 = []
    
    # Solve subproblems for each scenario
    for s in scenarios:
        print(f"\nSolving subproblem for scenario {s}")
        sub_result = opt.solve(sub)
        
        if sub_result.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(f"Subproblem for scenario '{s}' failed to solve optimally.")
            print(f"Status: {sub_result.solver.termination_condition}")
            print(f"Message: {sub_result.solver.message}")
            print(f"V24_fixed value: {V24_fixed}")
            print(f"V[T1+1] bound: (0, {Vmax})")
            print(f"Current inflow I[{s}, {T1+1}]: {I[s, T1+1]}")
            # Write the LP file for debugging
            sub.write(f"subproblem_{s}.lp", io_options={'symbolic_solver_labels': True})
            print(f"Wrote problem to subproblem_{s}.lp for inspection")
            break
            
        sub_obj_value = pyo.value(sub.obj)
        expected_sub_obj += prob[s] * sub_obj_value
        
        # Get dual value for V25 constraint
        dual_value = sub.dual[sub.V25]
        duals_V25.append(dual_value)
        
        print(f" Subproblem '{s}' Objective: {sub_obj_value}, Dual V25: {dual_value}")
    
    # Update bounds
    lower_bound = master_obj_value
    upper_bound = pyo.value(master.obj) - master_theta + expected_sub_obj
    
    upper_bound_list.append(upper_bound)
    lower_bound_list.append(lower_bound)
    
    print(f"Updated bounds - Lower: {lower_bound}, Upper: {upper_bound}")
    
    # print current convergence tracking values (use the lists' last entries)
    print(f" Lower Bound list last: {lower_bound_list[-1]}, Upper Bound list last: {upper_bound_list[-1]}")
    
    # If any subproblem failed, duals_V25 will be shorter than scenarios.
    if len(duals_V25) != len(scenarios):
        print("One or more subproblems infeasible; adding feasibility cut on master and skipping Benders cut for this iteration.")
        # Add a simple feasibility cut preventing the master from choosing the same V24 next time
        # Use a small epsilon to avoid numerical equality
        eps = 1e-6
        cut_name = f"FeasibilityCut_{iteration}"
        setattr(master, cut_name, pyo.Constraint(expr=master.V[T1] <= master_V24 - eps))
    else:
        # Add Benders cut to master problem (expected_sub_obj added once, not inside sum)
        def benders_cut_rule(m):
            return m.theta <= sum(prob[s] * duals_V25[i] * (m.V[T1] - V24_fixed) for i, s in enumerate(scenarios)) + expected_sub_obj
        master.BendersCut = pyo.Constraint(rule=benders_cut_rule)
    
    iteration += 1
