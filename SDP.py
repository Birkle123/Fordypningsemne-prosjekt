import numpy as np
import pyomo.environ as pyo

# --- Parameters (copied from benders_decomp.py) ---
T = 48
T1 = 24

V0 = 3.0  # Mm3
Vmax = 4.5  # Mm3
Qmax = 100.0  # m3/s
Pmax = 86.5  # MW
alpha = 3.6/1000.0  # Mm3 per (m3/s over one hour)
E_conv = 0.657  # kWh per m3
WV_end = 52600.0

P_full_discharge = E_conv * Qmax * 3600 / 1000
Qmax_from_P = Qmax * (Pmax / P_full_discharge)
q_cap = min(Qmax, Qmax_from_P)

spill_cost = WV_end

# Price profile (€/MWh assumed; use same multiplier as in benders if needed)
pi = {t: 50.0 + (t+1) for t in range(1, T+1)}

# Scenarios
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
I = {}
for s in scenarios:
    for t in range(1, T + 1):
        if t <= T1:
            I[s, t] = 50.0
        else:
            I[s, t] = scenario_info[s]

# Expected inflow for first stage
I_exp = {t: (50.0 if t <= T1 else sum(scenario_info.values()) / len(scenarios)) for t in range(1, T+1)}

# Solver
opt = pyo.SolverFactory('gurobi')

# Discretize the state (V at time T1) into 10 points
n_grid = 10
V_grid = np.linspace(0.0, Vmax, n_grid)

# Function to solve second-stage deterministic problem for given initial V_T1 and scenario
def solve_second_stage(initial_V24, inflows_for_scenario):
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(T1+1, T)

    m.q = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, q_cap))
    m.V = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, Vmax))
    m.s = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, None))

    # Linking: V[T1+1] == initial_V24 + alpha*I_{T1+1} - alpha*q_{T1+1} - s_{T1+1}
    def V25_rule(mm):
        t = T1+1
        return mm.V[t] == initial_V24 + alpha * inflows_for_scenario[t] - alpha * mm.q[t] - mm.s[t]
    m.V25 = pyo.Constraint(rule=V25_rule)

    def res_rule(mm, t):
        if t == T1+1:
            return pyo.Constraint.Skip
        return mm.V[t] == mm.V[t-1] + alpha * inflows_for_scenario[t] - alpha * mm.q[t] - mm.s[t]
    m.res = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(mm):
        return sum(pi[t] * 3.6 * E_conv * mm.q[t] for t in mm.T) - spill_cost * sum(mm.s[t] for t in mm.T) + WV_end * mm.V[T]
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    res = opt.solve(m)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"Warning: second-stage solve not optimal (status={res.solver.termination_condition}) for initial V={initial_V24}")
        return None
    return pyo.value(m.obj)

# Precompute second-stage values for each grid point and scenario
second_values = {s: [] for s in scenarios}
for s in scenarios:
    print(f"Computing second-stage values for scenario {s}...")
    for Vinit in V_grid:
        val = solve_second_stage(Vinit, {t: (50.0 if t <= T1 else I[s, t]) for t in range(1, T+1)})
        if val is None:
            val = -1e20
        second_values[s].append(val)

# Compute expected second-stage value for each grid point
expected_second = []
for i in range(n_grid):
    ev = sum(prob[s] * second_values[s][i] for s in scenarios)
    expected_second.append(ev)

# Now solve first-stage problem for each gridpoint by fixing V[T1] and adding expected second-stage value as constant
best_total = -1e30
best_solution = None
print('\nSolving first-stage problems over grid points...')
for i, Vfixed in enumerate(V_grid):
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T1)
    m.q = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, q_cap))
    m.V = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, Vmax))
    m.s = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, None))

    def res_rule(mm, t):
        if t == 1:
            return mm.V[t] == V0 + alpha * I_exp[t] - alpha * mm.q[t] - mm.s[t]
        return mm.V[t] == mm.V[t-1] + alpha * I_exp[t] - alpha * mm.q[t] - mm.s[t]
    m.res = pyo.Constraint(m.T, rule=res_rule)

    # Fix final first-stage state
    m.V_fix = pyo.Constraint(expr=m.V[T1] == Vfixed)

    def obj_rule(mm):
        stage1 = sum(pi[t] * 3.6 * E_conv * mm.q[t] for t in mm.T)
        stage1 -= spill_cost * sum(mm.s[t] for t in mm.T)
        # Add expected second-stage value as a constant
        return stage1 + expected_second[i]
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    res = opt.solve(m)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"Warning: first-stage (V={Vfixed}) not optimal: {res.solver.termination_condition}")
        continue
    stage1_val = pyo.value(sum(pi[t] * 3.6 * E_conv * m.q[t] for t in m.T)) - pyo.value(spill_cost * sum(m.s[t] for t in m.T))
    total = stage1_val + expected_second[i]
    print(f"Vfixed={Vfixed:.3f}  Stage1={stage1_val:.2f}  ExpectedSecond={expected_second[i]:.2f}  Total={total:.2f}")
    if total > best_total:
        best_total = total
        best_solution = (Vfixed, stage1_val, expected_second[i], total)

# Report best
print('\nBEST SOLUTION (approx via state discretization)')
if best_solution is not None:
    Vbest, st1, exp2, tot = best_solution
    print(f"Chosen V[T1] = {Vbest:.4f} Mm3")
    print(f"Stage-1 profit = {st1:.2f}")
    print(f"Expected second-stage profit = {exp2:.2f}")
    print(f"Total expected profit = {tot:.2f}")
else:
    print("No feasible solution found over grid.")

# Done
