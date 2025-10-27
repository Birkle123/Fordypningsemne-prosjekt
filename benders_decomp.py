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
V24_fixed = {}

# --- Master Problem ---

# Create master problem
master = pyo.ConcreteModel()

# Sets
master.T = pyo.RangeSet(1, T1)

# Variables
master.q = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, q_cap)) # Discharge [m3/s]
master.V = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, Vmax)) # Reservoir volume [Mm3]
master.theta = pyo.Var(within=pyo.Reals) # Approximation of expected future cost

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

# Variables
sub.q = pyo.Var(sub.T, within=pyo.NonNegativeReals, bounds=(0, q_cap)) # Discharge [m3/s]
sub.V = pyo.Var(sub.T, within=pyo.NonNegativeReals, bounds=(0, Vmax)) # Reservoir volume [Mm3]

# Suffix to capture dual values
sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Reservoir balance constraint
def V25_rule(m, s):
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

opt = pyo.SolverFactory("gruobi")
cut_counter = 1
upper_bounds = float('inf')
lower_bounds = -float('inf')
tolerance = 1e-4
iteration = 1

# Lists to store upper and lower bounds for plotting
upper_bound_list = []
lower_bound_list = []

while abs(upper_bound - lower_bound) > tolerance:
    print(f"--- Benders Iteration {iteration} ---")
    
    # Solve master problem
    master_result = opt.solve(master)
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
        sub_result = opt.solve(sub)
        sub_obj_value = pyo.value(sub.obj)
        expected_sub_obj += prob[s] * sub_obj_value
        
        # Get dual value for V25 constraint
        dual_value = sub.dual[sub.V25]
        duals_V25.append(dual_value)
        
        print(f" Subproblem '{s}' Objective: {sub_obj_value}, Dual V25: {dual_value}")
    
    # Update bounds
    lower_bound = master_obj_value
    upper_bound = sum(pyo.value(master.obj) - master_theta + expected_sub_obj)
    
    upper_bound_list.append(upper_bound)
    lower_bound_list.append(lower_bound)
    
    print(f" Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    
    # Add Benders cut to master problem
    def benders_cut_rule(m):
        return m.theta <= sum(prob[s] * (duals_V25[i] * (m.V[T1] - V24_fixed) + expected_sub_obj) for i, s in enumerate(scenarios))
    
    master.BendersCut = pyo.Constraint(rule=benders_cut_rule)
    
    iteration += 1
