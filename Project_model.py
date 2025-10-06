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
    m.T = pyo.RangeSet(1,T)

    # Variables
    m.q = pyo.Var(m.T, bounds=(0,q_cap))   # discharge
    m.s = pyo.Var(m.T, bounds=(0,None))    # spillage
    m.V = pyo.Var(m.T, bounds=(0,Vmax))    # volume

    # Reservoir balance
    def res_rule(m,t):
        if t == 1:
            return m.V[t] == V0 + alpha*I_exp[t] - alpha*m.q[t] - alpha*m.s[t]
        else:
            return m.V[t] == m.V[t-1] + alpha*I_exp[t] - alpha*m.q[t] - alpha*m.s[t]
    m.res_balance = pyo.Constraint(m.T, rule=res_rule)

    # Objective
    def obj_rule(m):
        revenue = sum(pi[t]*eta*m.q[t] for t in m.T)
        terminal = WV_end*m.V[T]
        return revenue + terminal
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m

#Determenistic equivalent number 2 (individual scenario)
def build_scenario_model(s):
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1,T)

    # Variables
    m.q = pyo.Var(m.T, bounds=(0,q_cap))
    m.s = pyo.Var(m.T, bounds=(0,None))
    m.V = pyo.Var(m.T, bounds=(0,Vmax))

    # Balance
    def res_rule(m,t):
        if t == 1:
            return m.V[t] == V0 + alpha*I[s,t] - alpha*m.q[t] - alpha*m.s[t]
        else:
            return m.V[t] == m.V[t-1] + alpha*I[s,t] - alpha*m.q[t] - alpha*m.s[t]
    m.res_balance = pyo.Constraint(m.T, rule=res_rule)

    # Objective
    def obj_rule(m):
        return sum(pi[t]*eta*m.q[t] for t in m.T) + WV_end*m.V[T]
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m

#Determenistic equivalent number 3
def build_stochastic_model():
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=scenarios)
    m.T1 = pyo.RangeSet(1,T1)
    m.T2 = pyo.RangeSet(T1+1,T)

    # First stage variables
    m.q1 = pyo.Var(m.T1, bounds=(0,q_cap))
    m.s1 = pyo.Var(m.T1, bounds=(0,None))
    m.V1 = pyo.Var(m.T1, bounds=(0,Vmax))

    # Second stage variables (scenario-dependent)
    m.q2 = pyo.Var(m.S, m.T2, bounds=(0,q_cap))
    m.s2 = pyo.Var(m.S, m.T2, bounds=(0,None))
    m.V2 = pyo.Var(m.S, m.T2, bounds=(0,Vmax))

    # Reservoir balance
    def res_balance_first(m,t):
        if t == 1:
            return m.V1[t] == V0 + alpha*50 - alpha*m.q1[t] - alpha*m.s1[t]
        else:
            return m.V1[t] == m.V1[t-1] + alpha*50 - alpha*m.q1[t] - alpha*m.s1[t]
    m.res_first = pyo.Constraint(m.T1, rule=res_balance_first)

    def res_balance_second(m,s,t):
        if t == T1+1:
            return m.V2[s,t] == m.V1[T1] + alpha*I[s,t] - alpha*m.q2[s,t] - alpha*m.s2[s,t]
        else:
            return m.V2[s,t] == m.V2[s,t-1] + alpha*I[s,t] - alpha*m.q2[s,t] - alpha*m.s2[s,t]
    m.res_second = pyo.Constraint(m.S, m.T2, rule=res_balance_second)

    # Objective (expected value across scenarios)
    def obj_rule(m):
        revenue1 = sum(pi[t]*eta*m.q1[t] for t in m.T1)
        rev2 = sum(prob[s]*(sum(pi[t]*eta*m.q2[s,t] for t in m.T2) + WV_end*m.V2[s,T]) for s in m.S)
        terminal1 = 0 # only count second-stage terminal value
        return revenue1 + rev2 + terminal1
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m



#Solving the models
solver = pyo.SolverFactory("gurobi")

# EV model
m_ev = build_EV_model()
solver.solve(m_ev)
print("EV objective:", pyo.value(m_ev.obj))

# Scenario models
for s in scenarios:
    m_s = build_scenario_model(s)
    solver.solve(m_s)
    print(f"Scenario {s}, inflow={scenario_inflows[s]}: objective =", pyo.value(m_s.obj))

# Stochastic model
m_stoch = build_stochastic_model()
solver.solve(m_stoch)
print("Stochastic objective:", pyo.value(m_stoch.obj))

# After solving EV model:
print("EV objective:", pyo.value(m_ev.obj))
print("First 24h discharge profile (q):")
print([pyo.value(m_ev.q[t]) for t in range(1,25)])

# After solving one scenario:
print("Scenario 1 objective:", pyo.value(m_s.obj))
print("First 24h q:", [pyo.value(m_s.q[t]) for t in range(1,25)])

# After solving stochastic model:
print("Stochastic objective:", pyo.value(m_stoch.obj))
print("Shared first-stage q (hours 1-24):")
print([pyo.value(m_stoch.q1[t]) for t in range(1,25)])



