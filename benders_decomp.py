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
def V25_rule(m):
    return m.V[T1+1] == V24_fixed + alpha * I[s, T1+1] - alpha * m.q[T1+1]
sub.V25 = pyo.Constraint(rule=V25_rule)

def sub_res_rule(m, t):
    if t == T1 + 1:
        return pyo.Constraint.Skip
    return m.V[t] == m.V[t-1] + alpha * (I[s][t] - alpha * m.q[t])
sub.res_balance = pyo.Constraint(sub.T, rule=sub_res_rule)

# Objective function
def sub_obj_rule(m, s):
    return sum(pi[t] * E_conv * m.q[t] for t in m.T) + (WV_end * m.V[T] if T in m.T else 0)
