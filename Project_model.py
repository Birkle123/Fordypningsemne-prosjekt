import numpy as np
import pandas as pd
import plotly as plt
import pyomo.environ as pyo


# --- Parameters ---
T = 48
T1 = 24
scenarios = [1,2,3,4,5]
prob = {s:1/len(scenarios) for s in scenarios}

# Reservoir & turbine parameters
V0 = 3.0
Vmax = 4.5
Qmax = 100.0
Pmax = 86.5
alpha = 3.6/1000.0
eta = 0.657 * 3.6
WV_end = 52600.0
q_cap = min(Qmax, Pmax/eta)

# Price profile (same for all scenarios)
pi = {t: 50.0 + (t+1) for t in range(1,T+1)}

# Inflows
scenario_inflows = {1:0, 2:10, 3:20, 4:30, 5:40}

I = {}
for s in scenarios:
    for t in range(1,T+1):
        if t <= T1:
            I[s,t] = 50.0
        else:
            I[s,t] = scenario_inflows[s]

# Expected inflow for EV model
I_exp = {t: (50.0 if t<=T1 else sum(scenario_inflows.values())/len(scenarios)) for t in range(1,T+1)}

#Determenistic equivalent number 1 (Expected value model)
def build_EV_model():
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T)
    m.q = pyo.Var(m.T, bounds=(0, q_cap))
    m.V = pyo.Var(m.T, bounds=(0, Vmax))

    def res_rule(m, t):
        if t == 1:
            return m.V[t] == V0 + alpha * I_exp[t] - alpha * m.q[t]
        else:
            return m.V[t] == m.V[t - 1] + alpha * I_exp[t] - alpha * m.q[t]

    m.res_balance = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(m):
        return sum(pi[t] * eta * m.q[t] for t in m.T) + WV_end * m.V[T]

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    return m



#Determenistic equivalent number 2 (individual scenario)
def build_scenario_model(s):
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T)
    m.q = pyo.Var(m.T, bounds=(0, q_cap))
    m.V = pyo.Var(m.T, bounds=(0, Vmax))

    def res_rule(m, t):
        if t == 1:
            return m.V[t] == V0 + alpha * I[s, t] - alpha * m.q[t]
        else:
            return m.V[t] == m.V[t - 1] + alpha * I[s, t] - alpha * m.q[t]

    m.res_balance = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(m):
        return sum(pi[t] * eta * m.q[t] for t in m.T) + WV_end * m.V[T]

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m

#Determenistic equivalent number 3
def build_stochastic_model():
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=scenarios)
    m.T1 = pyo.RangeSet(1, T1)
    m.T2 = pyo.RangeSet(T1 + 1, T)

    # Shared q for first 24h
    m.q1 = pyo.Var(m.T1, bounds=(0, q_cap))
    m.V1 = pyo.Var(m.T1, bounds=(0, Vmax))

    # Scenario-specific q, V for day 2
    m.q2 = pyo.Var(m.S, m.T2, bounds=(0, q_cap))
    m.V2 = pyo.Var(m.S, m.T2, bounds=(0, Vmax))

    # Reservoir balance for day 1 (shared)
    def res1(m, t):
        if t == 1:
            return m.V1[t] == V0 + alpha * 50 - alpha * m.q1[t]
        else:
            return m.V1[t] == m.V1[t - 1] + alpha * 50 - alpha * m.q1[t]
    m.res1 = pyo.Constraint(m.T1, rule=res1)

    # Reservoir balance for day 2 (scenario-dependent)
    def res2(m, s, t):
        if t == T1 + 1:
            return m.V2[s, t] == m.V1[T1] + alpha * I[s, t] - alpha * m.q2[s, t]
        else:
            return m.V2[s, t] == m.V2[s, t - 1] + alpha * I[s, t] - alpha * m.q2[s, t]
    m.res2 = pyo.Constraint(m.S, m.T2, rule=res2)

    # Objective: expected revenue
    def obj(m):
        first = sum(pi[t] * eta * m.q1[t] for t in m.T1)
        second = sum(prob[s] * (
            sum(pi[t] * eta * m.q2[s, t] for t in m.T2)
            + WV_end * m.V2[s, T]
        ) for s in m.S)
        return first + second
    m.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

    # ✅ Return model
    return m




#Solving the models
solver = pyo.SolverFactory("gurobi")

# EV model
m_ev = build_EV_model()
solver.solve(m_ev)


# Scenario models
for s in scenarios:
    m_s = build_scenario_model(s)
    solver.solve(m_s)
    print(f"Scenario {s}, inflow={scenario_inflows[s]}: objective =", pyo.value(m_s.obj))
    print("First 24h q:", [pyo.value(m_s.q[t]) for t in range(1,25)])

# Stochastic model
m_stoch = build_stochastic_model()
solver.solve(m_stoch)


#hei

# After solving EV model:
print("EV objective:", pyo.value(m_ev.obj))
print("First 24h discharge profile (q):")
print([pyo.value(m_ev.q[t]) for t in range(1,25)])

# After solving one scenario:
"""for s in scenarios:
    print("Scenario 1 objective:", pyo.value(m_s.obj))
    print("First 24h q:", [pyo.value(m_s.q[t]) for t in range(1,25)])"""

# After solving stochastic model:
print("Stochastic objective:", pyo.value(m_stoch.obj))
print("Shared first-stage q (hours 1-24):")
print([pyo.value(m_stoch.q1[t]) for t in range(1,25)])



