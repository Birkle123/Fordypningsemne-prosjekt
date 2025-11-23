import numpy as np
import pandas as pd
import pyomo.environ as pyo
import time
from config import config

def build_extensive_form():
    """Build two-stage extensive form (deterministic equivalent) model."""
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=config.scenarios)
    m.T1 = pyo.RangeSet(1, config.T1)
    m.T2 = pyo.RangeSet(config.T1 + 1, config.T)

    # First-stage variables (shared across scenarios - nonanticipativity)
    m.q1 = pyo.Var(m.T1, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
    m.V1 = pyo.Var(m.T1, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
    m.s1 = pyo.Var(m.T1, within=pyo.NonNegativeReals, bounds=(0, None))

    # Second-stage scenario-specific variables
    m.q2 = pyo.Var(m.S, m.T2, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
    m.V2 = pyo.Var(m.S, m.T2, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
    m.s2 = pyo.Var(m.S, m.T2, within=pyo.NonNegativeReals, bounds=(0, None))

    # Build inflow data
    I_exp = {}
    for t in range(1, config.T + 1):
        if t <= config.T1:
            I_exp[t] = config.certain_inflow
        else:
            I_exp[t] = sum(config.scenario_inflows) / len(config.scenario_inflows)

    I = {}
    for s in config.scenarios:
        for t in range(1, config.T + 1):
            if t <= config.T1:
                I[s, t] = config.certain_inflow
            else:
                I[s, t] = config.scenario_info[s]

    # First-stage reservoir balance
    def res1(mm, t):
        if t == 1:
            return mm.V1[t] == config.V0 + config.alpha * I_exp[t] - config.alpha * mm.q1[t] - mm.s1[t]
        return mm.V1[t] == mm.V1[t-1] + config.alpha * I_exp[t] - config.alpha * mm.q1[t] - mm.s1[t]
    m.res1 = pyo.Constraint(m.T1, rule=res1)

    # Second-stage linking constraints
    def res2_first(mm, s):
        return m.V2[s, config.T1 + 1] == m.V1[config.T1] + config.alpha * I[s, config.T1 + 1] - config.alpha * m.q2[s, config.T1 + 1] - m.s2[s, config.T1 + 1]
    m.res2_link = pyo.Constraint(m.S, rule=res2_first)

    # Second-stage reservoir balance
    def res2(mm, s, t):
        if t == config.T1 + 1:
            return pyo.Constraint.Skip
        return mm.V2[s, t] == mm.V2[s, t-1] + config.alpha * I[s, t] - config.alpha * mm.q2[s, t] - mm.s2[s, t]
    m.res2 = pyo.Constraint(m.S, m.T2, rule=res2)

    # Objective: expected profit
    def obj_ext(mm):
        # First stage deterministic profit
        stage1 = sum(config.pi[t] * 3.6 * config.E_conv * mm.q1[t] for t in mm.T1) - config.spillage_cost * sum(mm.s1[t] for t in mm.T1)
        
        # Second stage expected profit
        stage2 = sum(config.prob[s] * (
            sum(config.pi[t] * 3.6 * config.E_conv * mm.q2[s, t] for t in mm.T2) 
            - config.spillage_cost * sum(mm.s2[s, t] for t in mm.T2)
            + config.WV_end * mm.V2[s, config.T]
        ) for s in mm.S)
        
        return stage1 + stage2
    m.obj = pyo.Objective(rule=obj_ext, sense=pyo.maximize)

    return m

def run_extensive_form(plot=False, summary=True):
    """Run extensive form (deterministic equivalent) analysis."""
    start_time = time.time()
    
    if summary:
        print("\n" + "="*80)
        print("Extensive Form (Deterministic Equivalent) Model")
        print("="*80)
    
    # Build and solve model
    m = build_extensive_form()
    solver = config.get_solver()
    
    result = solver.solve(m)
    solve_time = time.time() - start_time
    
    if summary:
        print(f"Solve time: {solve_time:.3f} seconds")
        
        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            obj_value = pyo.value(m.obj)
            V24_value = pyo.value(m.V1[config.T1])
            
            print(f"Objective value: {obj_value:,.2f} NOK")
            print(f"Optimal V24: {V24_value:.3f} Mm³")
            
            # First stage statistics
            q1_vals = [pyo.value(m.q1[t]) for t in range(1, config.T1 + 1)]
            V1_vals = [pyo.value(m.V1[t]) for t in range(1, config.T1 + 1)]
            s1_vals = [pyo.value(m.s1[t]) for t in range(1, config.T1 + 1)]
            
            print(f"First stage statistics:")
            print(f"  Discharge - Avg: {np.mean(q1_vals):.2f} m³/s, Max: {np.max(q1_vals):.2f} m³/s")
            print(f"  Reservoir - Final: {V1_vals[-1]:.2f} Mm³, Max: {np.max(V1_vals):.2f} Mm³")
            print(f"  Total spillage: {np.sum(s1_vals):.2f} m³/s·h")
            
            # Second stage expected statistics
            expected_q2 = []
            expected_V2 = []
            expected_s2 = []
            
            for t in range(config.T1 + 1, config.T + 1):
                eq = sum(config.prob[s] * pyo.value(m.q2[s, t]) for s in config.scenarios)
                ev = sum(config.prob[s] * pyo.value(m.V2[s, t]) for s in config.scenarios)
                es = sum(config.prob[s] * pyo.value(m.s2[s, t]) for s in config.scenarios)
                expected_q2.append(eq)
                expected_V2.append(ev)
                expected_s2.append(es)
            
            print(f"Second stage expected statistics:")
            print(f"  Discharge - Avg: {np.mean(expected_q2):.2f} m³/s, Max: {np.max(expected_q2):.2f} m³/s")
            print(f"  Reservoir - Final: {expected_V2[-1]:.2f} Mm³, Max: {np.max(expected_V2):.2f} Mm³")
            print(f"  Total spillage: {np.sum(expected_s2):.2f} m³/s·h")
        else:
            print(f"Model did not solve optimally: {result.solver.termination_condition}")
            obj_value = None
            V24_value = None
        
        print("="*80)
    
    return {
        'objective': obj_value if result.solver.termination_condition == pyo.TerminationCondition.optimal else None,
        'V24': V24_value if result.solver.termination_condition == pyo.TerminationCondition.optimal else None,
        'solve_time': solve_time,
        'model': m,
        'result': result
    }