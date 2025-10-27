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

I_exp = {
    t: (50.0 if t <= T1 else sum(scenario_inflows.values()) / len(scenarios))
    for t in range(1, T + 1)
}

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
            return sum(pi[t] * 3.6 * E_conv * m.q[t] for t in m.T) + WV_end * m.V[T]

    # EUR / MWh * 1h * kWh/m3

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
        return sum(pi[t] * 3.6 * E_conv * m.q[t] for t in m.T) + WV_end * m.V[T]

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
        first = sum(pi[t] * 3.6 * E_conv * m.q1[t] for t in m.T1)
        second = sum(prob[s] * (
            sum(pi[t] * 3.6 * E_conv * m.q2[s, t] for t in m.T2)
            + WV_end * m.V2[s, T]
        ) for s in m.S)
        return first + second
    m.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

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
    print(f"{s} scenario (inflow={scenario_inflows[s]}): objective = {pyo.value(m_s.obj):,.2f}")
    print("First 24h q:", [pyo.value(m_s.q[t]) for t in range(1, 25)])


# Stochastic model
m_stoch = build_stochastic_model()
solver.solve(m_stoch)

# ---- Results summary printing ----
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
plot_three_figures()

