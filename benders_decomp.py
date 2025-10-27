import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# --- Parameters ---
T = 48
T1 = 24

# Reservoir & turbine parameters
V0 = 3.0  # Mm3
Vmax = 4.5  # Mm3
Qmax = 100.0  # m3/s
Pmax = 86.5  # MW
alpha = 3.6/1000.0  # Conversion factor from flow (m³/s) over one hour to million cubic meters (Mm³)
E_conv = 0.657  # Energy [kWh] generated per cubic meter of discharged water.
WV_end = 52600.0

P_full_discharge = E_conv * Qmax * 3600 / 1000
Qmax_from_P = Qmax * (Pmax / P_full_discharge)
# kWh/m3 * m3/s * 3600s/h = kWh (3600 * E_conv * m.q[t] * 1000)
q_cap = min(Qmax, Qmax_from_P)  # Max discharge limited by both physical and power constraints

# Spillage penalty (€/Mm3). Default: equal to terminal water value (strong discouragement)
spill_cost = WV_end

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
master.q = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, q_cap), initialize=0.0)  # Discharge [m3/s]
master.V = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, Vmax), initialize=V0)  # Reservoir volume [Mm3]
master.theta = pyo.Var(within=pyo.Reals, bounds=(-1e6, 1e6), initialize=0.0)  # Approximation of expected future cost
master.s = pyo.Var(master.T, within=pyo.NonNegativeReals, bounds=(0, None), initialize=0.0)  # Spillage [Mm3]

# Reservoir balance constraint
def master_res_rule(m, t):
    if t == 1:
        return m.V[t] == V0 + alpha * I_exp[t] - alpha * m.q[t] - m.s[t]
    else:
        return m.V[t] == m.V[t-1] + alpha * I_exp[t] - alpha * m.q[t] - m.s[t]

# Connect constraint to model
master.res_balance = pyo.Constraint(master.T, rule=master_res_rule)

# Objective function
def master_obj_rule(m):
    # Multiply by 3.6 to convert from kWh to MWh when pi is in €/MWh
    return sum(pi[t] * 3.6 * E_conv * m.q[t] for t in m.T) + m.theta - spill_cost * sum(m.s[t] for t in m.T)

# Connect objective to model
master.obj = pyo.Objective(rule=master_obj_rule, sense=pyo.maximize)

# --- Subproblem ---

# Create subproblem
sub = pyo.ConcreteModel()

# Sets
sub.T = pyo.RangeSet(T1+1, T)
sub.S = pyo.Set(initialize=scenarios)

# Variables with initialization (indexed by scenario and time)
sub.q = pyo.Var(sub.S, sub.T, within=pyo.NonNegativeReals, bounds=(0, q_cap), initialize=0.0)  # Discharge [m3/s]
sub.V = pyo.Var(sub.S, sub.T, within=pyo.NonNegativeReals, bounds=(0, Vmax), initialize=V0)  # Reservoir volume [Mm3]
sub.s = pyo.Var(sub.S, sub.T, within=pyo.NonNegativeReals, bounds=(0, None), initialize=0.0)  # Spillage [Mm3]

# Suffix to capture dual values
sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Mutable parameter to receive V24 from master
sub.V24 = pyo.Param(initialize=V24_fixed, mutable=True)

# Reservoir balance constraint for linking with master problem
# V25 linking constraint (per scenario)
def V25_rule(m, s):
    # Include spillage in the linking equation (spillage is volume, in Mm3)
    return m.V[s, T1+1] == m.V24 + alpha * I[s, T1+1] - alpha * m.q[s, T1+1] - m.s[s, T1+1]
sub.V25 = pyo.Constraint(sub.S, rule=V25_rule)


# Reservoir balance constraint
def sub_res_rule(m, s, t):
    if t == T1 + 1:
        return pyo.Constraint.Skip
    return m.V[s, t] == m.V[s, t-1] + alpha * I[s, t] - alpha * m.q[s, t] - m.s[s, t]
sub.res_balance = pyo.Constraint(sub.S, sub.T, rule=sub_res_rule)

# Objective function
def sub_obj_rule(m, s):
    # Revenue minus spillage penalty + terminal water value
    return sum(pi[t] * 3.6 * E_conv * m.q[s, t] for t in m.T) - spill_cost * sum(m.s[s, t] for t in m.T) + (WV_end * m.V[s, T] if T in m.T else 0)
sub.obj = pyo.Objective(sub.S, rule=sub_obj_rule, sense=pyo.maximize)

# --- Benders Decomposition Algorithm ---

opt = pyo.SolverFactory("gurobi")
cut_counter = 1
global_LB = -float('inf')     # best lower bound seen so far
global_UB = float('inf')      # best upper bound seen so far
tolerance = 1e-3
max_iterations = 10           # Reduced from 30 since convergence usually happens earlier
iteration = 1

# Lists to store upper and lower bounds for plotting
upper_bound_list = []
lower_bound_list = []

while abs(global_UB-global_LB) > tolerance and iteration <= max_iterations:
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
    any_subproblem_failed = False

    w_s_list = []      # Stores subproblem objective values w_s
    duals_V25 = []     # Stores duals λ_s for each scenario

    for s in scenarios:
        print(f"\nSolving subproblem for scenario {s}")
        # update V24 in the subproblem (mutable Param)
        sub.V24.set_value(master_V24)
        # compute maximum feasible q at t = T1+1 for this V24
        max_feasible_discharge = (master_V24 + alpha * I[s, T1+1]) / alpha   # careful: algebraically equivalent as in your code
        sub.q[s, T1+1].setub(max(0.0, min(q_cap, max_feasible_discharge)))

        # activate only the objective for scenario s
        for obj in sub.obj.values():
            try:
                obj.deactivate()
            except Exception:
                pass
        sub.obj[s].activate()

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
            any_subproblem_failed = True
            break


        sub_obj_value = pyo.value(sub.obj[s])

        # After solving the subproblem: store objective once
        w_s = sub_obj_value   # objective of the subproblem
        w_s_list.append(w_s)  # Store it
        expected_sub_obj += prob[s] * w_s

        # Get dual value for V25 constraint for this scenario
        try:
            dual_value = sub.dual[sub.V25[s]]
            if dual_value is None:
                dual_value = 0.0
        except:
            dual_value = 0.0
        duals_V25.append(dual_value)

        print(f" Subproblem '{s}' Objective: {sub_obj_value}, Dual V25: {dual_value}")
    
    # Update bounds
    if not any_subproblem_failed:
        lower_bound = master_obj_value
        upper_bound = pyo.value(master.obj) - master_theta + expected_sub_obj
        
        # Update bounds with best values seen so far
        global_LB = max(global_LB, lower_bound)
        global_UB = min(global_UB, upper_bound)
        
        # Store the current bounds for plotting
        lower_bound_list.append(lower_bound)  # Store the actual bound at this iteration
        upper_bound_list.append(upper_bound)  # Store the actual bound at this iteration
        
        print(f"Updated bounds - Lower: {lower_bound}, Upper: {upper_bound}")
        print(f"Current iteration bounds - Lower: {lower_bound}, Upper: {upper_bound}")
        print(f"Global bounds - Lower: {global_LB}, Upper: {global_UB}")
    
    # If any subproblem failed, duals_V25 will be shorter than scenarios.
    if any_subproblem_failed:
        print("One or more subproblems infeasible; adding feasibility cut on master and skipping Benders cut for this iteration.")
        infeasible_s = s
        inflow = I[infeasible_s, T1+1]
        
        # Direct feasibility cut based on V25 water balance:
        # V25 = V24 + inflow - α*q25 must be in [0, Vmax]
        # We need to find a value for V24 that ensures q25 exists to make V25 feasible
        
        # Calculate minimum V24 needed:
        # Worst case: Need maximum discharge to get V25 ≤ Vmax
        # V25 = V24 + inflow - α*q25 ≤ Vmax
        # V24 + inflow ≤ Vmax   (minimum occurs at q25 = 0)
        max_feasible_V24 = Vmax - inflow
        
        # Add a simple feasibility cut: directly cut off the current V24 value
        # by requiring it to be strictly less than the current value
        cut_name = f"FeasibilityCut_{iteration}"
        setattr(master, cut_name, pyo.Constraint(expr=master.V[T1] <= V24_fixed - 0.1))
        
        print(f"Added feasibility cut: V[{T1}] <= {V24_fixed - 0.1} (from scenario {infeasible_s})")
        # Also add a hard upper bound if we haven't already
        if iteration == 1:
            cut_name = "HardUpperBound"
            setattr(master, cut_name, pyo.Constraint(expr=master.V[T1] <= max_feasible_V24))
            print(f"Added hard upper bound: V[{T1}] <= {max_feasible_V24}")
    else:
        # Add Benders cut to master problem (expected_sub_obj added once, not inside sum)
        cut_name = f"BendersCut_{iteration}"

        def benders_cut_rule(m):
            return m.theta <= sum(
                prob[s] * (w_s_list[i] + duals_V25[i] * (m.V[T1] - V24_fixed))
                for i, s in enumerate(scenarios)
            )

        setattr(master, cut_name, pyo.Constraint(rule=benders_cut_rule))

    
    iteration += 1

# Print final results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

if global_UB < float('inf'):
    print(f"\nOptimal V24 value: {V24_fixed:.4f} Mm³")
    print(f"Total expected profit (feasible): {global_UB:.2f}")
else:
    print("\nBenders did not produce a feasible solution.")
    print("Check if cuts are added correctly or if master problem is too restrictive.")

print("\nScenario Details:")
print("-" * 70)
print(f"{'Scenario':<15} {'Profit':>12} {'V25 Dual':>12} {'Inflow':>10} {'V25':>10}")
print("-" * 70)

if 'duals_V25' in globals() and len(duals_V25) == len(scenarios):
    for i, s in enumerate(scenarios):
        obj = pyo.value(sub.obj[s])
        dual = duals_V25[i]
        inflow = I[s, T1+1]
        v25 = pyo.value(sub.V[s, T1+1])
        print(f"{s:<15} {obj:>12.2f} {dual:>12.2f} {inflow:>10.3f} {v25:>10.3f}")
else:
    print("No dual or scenario information available.")


# ------------------------
# Additional diagnostics (EV, EV policy, Wait-and-See)
# ------------------------
def solve_deterministic(inflows_map, solver=opt):
    """Solve full-horizon deterministic problem for given inflows_map keyed by t=1..T.
    Returns (objective_value, q_dict, V_dict, model) or (None,...) if not optimal.
    """
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T)
    m.q = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, q_cap), initialize=0.0)
    m.V = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, Vmax), initialize=V0)
    m.s = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, None), initialize=0.0)

    def res_rule(mm, t):
        if t == 1:
            return mm.V[t] == V0 + alpha * inflows_map[t] - alpha * mm.q[t] - mm.s[t]
        return mm.V[t] == mm.V[t-1] + alpha * inflows_map[t] - alpha * mm.q[t] - mm.s[t]
    m.res = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(mm):
        return sum(pi[t] * 3.6 * E_conv * mm.q[t] for t in mm.T) - spill_cost * sum(mm.s[t] for t in mm.T) + WV_end * mm.V[T]
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    res = solver.solve(m)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        return None, None, None, m

    q_sol = {t: pyo.value(m.q[t]) for t in m.T}
    V_sol = {t: pyo.value(m.V[t]) for t in m.T}
    return pyo.value(m.obj), q_sol, V_sol, m


# Wait-and-see: average optimal objective if we could see the scenario
ws_objs = []
for s in scenarios:
    inflows_s = {t: (50.0 if t <= T1 else I[s, t]) for t in range(1, T+1)}
    obj, q_sol, V_sol, _ = solve_deterministic(inflows_s)
    if obj is None:
        print(f"Warning: deterministic solve for scenario {s} failed")
    else:
        ws_objs.append(obj)

if ws_objs:
    EVWS = sum(ws_objs) / len(ws_objs)
    print(f"\nWait-and-See (average over scenarios) objective: {EVWS:.2f}")

# Expected-value solution: deterministic on expected inflow
inflows_ev = {t: (50.0 if t <= T1 else I_exp[t]) for t in range(1, T+1)}
ev_obj, ev_q, ev_V, _ = solve_deterministic(inflows_ev)
if ev_obj is not None:
    print(f"Expected-value (deterministic on expected inflow) objective: {ev_obj:.2f}")
    # Evaluate EV policy in stochastic sense: fix V24 from EV solution and solve subproblems for each scenario
    V24_ev = ev_V[T1]
    stage1_profit_ev = sum(pi[t] * 3.6 * E_conv * ev_q[t] for t in range(1, T1+1))
    ev_second = 0.0
    for s in scenarios:
        sub_ev = pyo.ConcreteModel()
        sub_ev.T = pyo.RangeSet(T1+1, T)
        sub_ev.q = pyo.Var(sub_ev.T, within=pyo.NonNegativeReals, bounds=(0, q_cap))
        sub_ev.V = pyo.Var(sub_ev.T, within=pyo.NonNegativeReals, bounds=(0, Vmax))
        sub_ev.s = pyo.Var(sub_ev.T, within=pyo.NonNegativeReals, bounds=(0, None))
        sub_ev.V24 = pyo.Param(initialize=V24_ev)

        def V25_ev(mm):
            return sub_ev.V[T1+1] == sub_ev.V24 + alpha * I[s, T1+1] - alpha * sub_ev.q[T1+1] - sub_ev.s[T1+1]
        sub_ev.V25 = pyo.Constraint(rule=V25_ev)

        def res_ev(mm, t):
            if t == T1+1:
                return pyo.Constraint.Skip
            return mm.V[t] == mm.V[t-1] + alpha * I[s, t] - alpha * mm.q[t] - mm.s[t]
        sub_ev.res = pyo.Constraint(sub_ev.T, rule=res_ev)

        def obj_ev(mm):
            return sum(pi[t] * 3.6 * E_conv * mm.q[t] for t in mm.T) - spill_cost * sum(mm.s[t] for t in mm.T) + WV_end * mm.V[T]
        sub_ev.obj = pyo.Objective(rule=obj_ev, sense=pyo.maximize)

        r = opt.solve(sub_ev)
        if r.solver.termination_condition == pyo.TerminationCondition.optimal:
            ev_second += prob[s] * pyo.value(sub_ev.obj)
        else:
            print(f"Warning: EV policy evaluation subproblem for {s} not optimal: {r.solver.termination_condition}")

    ev_policy_expected = stage1_profit_ev + ev_second
    print(f"Expected profit of EV policy (stage1 from EV, averaged subproblems): {ev_policy_expected:.2f}")

# Compare stochastic (Benders) result
if 'global_UB' in globals():
    print(f"\nStochastic solution (Benders) expected profit: {global_UB:.2f}")
    if ws_objs:
        print(f"EVPI (Wait-and-See - Stochastic): {EVWS - global_UB:.2f}")
    if ev_obj is not None:
        print(f"VSS (Stochastic - EV policy): {global_UB - ev_policy_expected:.2f}")


# ------------------------
# Deterministic extensive form (all scenarios, non-anticipativity)
# ------------------------
def solve_extensive_form(scenarios, probs, solver=opt):
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=scenarios)
    m.T1 = pyo.RangeSet(1, T1)
    m.T2 = pyo.RangeSet(T1+1, T)

    # First-stage variables (shared across scenarios)
    m.q = pyo.Var(m.T1, within=pyo.NonNegativeReals, bounds=(0, q_cap))
    m.V = pyo.Var(m.T1, within=pyo.NonNegativeReals, bounds=(0, Vmax))

    # Second-stage scenario-specific variables
    m.qs = pyo.Var(m.S, m.T2, within=pyo.NonNegativeReals, bounds=(0, q_cap))
    m.Vs = pyo.Var(m.S, m.T2, within=pyo.NonNegativeReals, bounds=(0, Vmax))

    # First-stage reservoir balance
    def res1(mm, t):
        if t == 1:
            return mm.V[t] == V0 + alpha * I_exp[t] - alpha * mm.q[t]
        return mm.V[t] == mm.V[t-1] + alpha * I_exp[t] - alpha * mm.q[t]
    m.res1 = pyo.Constraint(m.T1, rule=res1)

    # Second-stage balances per scenario
    def res2_first(mm, s):
        # V_{T1+1} linking
        return m.Vs[s, T1+1] == m.V[T1] + alpha * I[s, T1+1] - alpha * m.qs[s, T1+1]
    m.res2_1 = pyo.Constraint(m.S, rule=res2_first)

    def res2(mm, s, t):
        if t == T1+1:
            return pyo.Constraint.Skip
        return mm.Vs[s, t] == mm.Vs[s, t-1] + alpha * I[s, t] - alpha * mm.qs[s, t]
    m.res2 = pyo.Constraint(m.S, m.T2, rule=res2)

    # Objective: expected profit (prob-weighted)
    def obj_ext(mm):
        stage1 = sum(pi[t] * E_conv * mm.q[t] for t in mm.T1)
        stage2 = sum(probs[s] * (sum(pi[t] * E_conv * mm.qs[s, t] for t in mm.T2) + probs[s] * 0) for s in mm.S)
        # Add terminal value WV_end * V_T for each scenario (weighted)
        term = sum(probs[s] * (WV_end * mm.Vs[s, T]) for s in mm.S)
        return stage1 + stage2 + term
    m.obj = pyo.Objective(rule=obj_ext, sense=pyo.maximize)

    res = solver.solve(m)
    return res, m


# Solve EF and report
res_ext, m_ext = solve_extensive_form(scenarios, prob)
if res_ext.solver.termination_condition == pyo.TerminationCondition.optimal:
    ef_obj = pyo.value(m_ext.obj)
    print(f"\nExtensive form objective: {ef_obj:.2f}")
else:
    print(f"\nExtensive form did not solve optimally: {res_ext.solver.termination_condition}")


# ------------------------
# Convergence plot
# ------------------------
try:
    import matplotlib.pyplot as plt
    
    # Create iteration indices
    iters = list(range(1, len(upper_bound_list) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iters, upper_bound_list, 'b-o', label='Upper bound', linewidth=2)
    plt.plot(iters, lower_bound_list, 'r-s', label='Lower bound', linewidth=2)
    
    # Add gap line
    gaps = [u - l for u, l in zip(upper_bound_list, lower_bound_list)]
    max_gap = max(gaps)
    
    # Add annotations
    plt.fill_between(iters, lower_bound_list, upper_bound_list, color='gray', alpha=0.2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('Benders Convergence')
    plt.legend(loc='center right')
    plt.grid(True)
    

    
    plt.show()
except Exception as e:
    print(f"Could not produce convergence plot: {e}")


# ------------------------
# Convergence plot
# ------------------------
try:
    import matplotlib.pyplot as plt

    # Prepare iteration indices — use length of lists; skip initial placeholder if present
    iters = list(range(1, max(len(lower_bound_list), len(upper_bound_list)) + 1))

    # Pad lists to same length for plotting
    lb = list(lower_bound_list)
    ub = list(upper_bound_list)
    n = len(iters)
    if len(lb) < n:
        lb += [lb[-1]] * (n - len(lb)) if lb else [None] * (n - len(lb))
    if len(ub) < n:
        ub += [ub[-1]] * (n - len(ub)) if ub else [None] * (n - len(ub))

    plt.figure(figsize=(8, 4.5))
    plt.plot(iters, ub, marker='o', label='Upper bound')
    plt.plot(iters, lb, marker='s', label='Lower bound')
    # Plot the gap on a secondary axis if bounds are numeric
    try:
        gap = [u - l if (u is not None and l is not None) else None for u, l in zip(ub, lb)]
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(iters, gap, color='gray', linestyle='--', label='Gap')
        ax2.set_ylabel('Gap')
    except Exception:
        pass

    plt.xlabel('Benders iteration')
    plt.ylabel('Objective value')
    plt.title('Benders convergence — upper/lower bounds')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Show the plot (in interactive environments this will pop up)
    plt.show()
except Exception as e:
    print(f"Could not produce convergence plot: {e}")
