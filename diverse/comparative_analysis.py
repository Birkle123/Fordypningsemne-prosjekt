import numpy as np
import pandas as pd
import pyomo.environ as pyo
import time
from config import config

def solve_deterministic_model(inflows_map):
    """Solve full-horizon deterministic problem for given inflows."""
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, config.T)
    m.q = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap), initialize=0.0)
    m.V = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax), initialize=config.V0)
    m.s = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, None), initialize=0.0)

    def res_rule(mm, t):
        if t == 1:
            return mm.V[t] == config.V0 + config.alpha * inflows_map[t] - config.alpha * mm.q[t] - mm.s[t]
        return mm.V[t] == mm.V[t-1] + config.alpha * inflows_map[t] - config.alpha * mm.q[t] - mm.s[t]
    m.res = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(mm):
        return sum(config.pi[t] * 3.6 * config.E_conv * mm.q[t] for t in mm.T) - config.spillage_cost * sum(mm.s[t] for t in mm.T) + config.WV_end * mm.V[config.T]
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    solver = config.get_solver()
    res = solver.solve(m)
    
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        return None, None, None, m

    q_sol = {t: pyo.value(m.q[t]) for t in m.T}
    V_sol = {t: pyo.value(m.V[t]) for t in m.T}
    return pyo.value(m.obj), q_sol, V_sol, m

def run_comparative_analysis(plot=False, summary=True):
    """Run comparative analysis: EV, Wait-and-See, EV policy evaluation."""
    start_time = time.time()
    
    if summary:
        print("\n" + "="*80)
        print("Comparative Analysis: EV vs Wait-and-See")
        print("="*80)
    
    # Expected inflow calculation
    I_exp = {}
    for t in range(1, config.T + 1):
        if t <= config.T1:
            I_exp[t] = config.certain_inflow
        else:
            I_exp[t] = sum(config.scenario_inflows) / len(config.scenario_inflows)
    
    # 1. Expected Value (EV) solution
    ev_obj, ev_q, ev_V, _ = solve_deterministic_model(I_exp)
    
    # 2. Wait-and-See analysis
    ws_objs = []
    for s in config.scenarios:
        inflows_s = {}
        for t in range(1, config.T + 1):
            if t <= config.T1:
                inflows_s[t] = config.certain_inflow
            else:
                inflows_s[t] = config.scenario_info[s]
        
        obj, q_sol, V_sol, _ = solve_deterministic_model(inflows_s)
        if obj is not None:
            ws_objs.append(obj)
    
    EVWS = sum(ws_objs) / len(ws_objs) if ws_objs else None
    
    # 3. EV policy evaluation
    if ev_obj is not None:
        V24_ev = ev_V[config.T1]
        stage1_profit_ev = sum(config.pi[t] * 3.6 * config.E_conv * ev_q[t] for t in range(1, config.T1 + 1))
        
        ev_second = 0.0
        for s in config.scenarios:
            # Build subproblem for EV policy evaluation
            sub_ev = pyo.ConcreteModel()
            sub_ev.T = pyo.RangeSet(config.T1 + 1, config.T)
            sub_ev.q = pyo.Var(sub_ev.T, within=pyo.NonNegativeReals, bounds=(0, config.q_cap))
            sub_ev.V = pyo.Var(sub_ev.T, within=pyo.NonNegativeReals, bounds=(0, config.Vmax))
            sub_ev.s = pyo.Var(sub_ev.T, within=pyo.NonNegativeReals, bounds=(0, None))
            
            # Scenario inflow
            I_s = {}
            for t in range(config.T1 + 1, config.T + 1):
                I_s[t] = config.scenario_info[s]
            
            def V25_ev(mm):
                return sub_ev.V[config.T1 + 1] == V24_ev + config.alpha * I_s[config.T1 + 1] - config.alpha * sub_ev.q[config.T1 + 1] - sub_ev.s[config.T1 + 1]
            sub_ev.V25 = pyo.Constraint(rule=V25_ev)

            def res_ev(mm, t):
                if t == config.T1 + 1:
                    return pyo.Constraint.Skip
                return mm.V[t] == mm.V[t-1] + config.alpha * I_s[t] - config.alpha * mm.q[t] - mm.s[t]
            sub_ev.res = pyo.Constraint(sub_ev.T, rule=res_ev)

            def obj_ev(mm):
                return sum(config.pi[t] * 3.6 * config.E_conv * mm.q[t] for t in mm.T) - config.spillage_cost * sum(mm.s[t] for t in mm.T) + config.WV_end * mm.V[config.T]
            sub_ev.obj = pyo.Objective(rule=obj_ev, sense=pyo.maximize)

            solver = config.get_solver()
            r = solver.solve(sub_ev)
            if r.solver.termination_condition == pyo.TerminationCondition.optimal:
                ev_second += config.prob[s] * pyo.value(sub_ev.obj)
        
        ev_policy_expected = stage1_profit_ev + ev_second
    else:
        ev_policy_expected = None
    
    solve_time = time.time() - start_time
    
    if summary:
        print(f"Solve time: {solve_time:.3f} seconds")
        print()
        
        if ev_obj is not None:
            print(f"Expected Value (EV) solution:")
            print(f"  Objective: {ev_obj:,.2f} NOK")
            print(f"  V24: {ev_V[config.T1]:.3f} Mm³")
        
        if EVWS is not None:
            print(f"Wait-and-See average:")
            print(f"  Objective: {EVWS:,.2f} NOK")
        
        if ev_policy_expected is not None:
            print(f"EV Policy (stochastic evaluation):")
            print(f"  Expected objective: {ev_policy_expected:,.2f} NOK")
        
        print()
        print("Individual scenario results:")
        print("-" * 50)
        for i, s in enumerate(config.scenarios):
            if i < len(ws_objs):
                print(f"  {s}: {ws_objs[i]:,.2f} NOK")
        
        print("="*80)
    
    return {
        'ev_obj': ev_obj,
        'ev_V24': ev_V[config.T1] if ev_V else None,
        'wait_and_see': EVWS,
        'ev_policy': ev_policy_expected,
        'scenario_objs': ws_objs,
        'solve_time': solve_time
    }