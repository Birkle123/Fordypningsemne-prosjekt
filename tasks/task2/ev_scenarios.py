import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import time
from config import config

def build_EV_model():
    """Expected Value model using average inflows."""
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, config.T)
    m.q = pyo.Var(m.T, bounds=(0, config.q_cap)) # discharge
    m.V = pyo.Var(m.T, bounds=(0, config.Vmax)) # reservoir level
    m.s = pyo.Var(m.T, bounds=(0, None)) # spillage

    # finding expected average inflow
    I_exp = {}
    for t in range(1, config.T + 1):
        if t <= config.T1:
            I_exp[t] = config.certain_inflow
        else:
            I_exp[t] = sum(scenario_inflow * prob for scenario_inflow, prob in zip(config.scenario_inflows, config.scenario_probabilities))

    def reservoir_balance(m, t):
        if t == 1:
            return m.V[t] == config.V0 + config.alpha * I_exp[t] - config.alpha * m.q[t] - config.alpha * m.s[t]
        else:
            return m.V[t] == m.V[t - 1] + config.alpha * I_exp[t] - config.alpha * m.q[t] - config.alpha * m.s[t]

    m.res_balance = pyo.Constraint(m.T, rule=reservoir_balance)

    def obj_rule(m):
        return sum(config.pi[t] * 3.6 * config.E_conv * m.q[t] for t in m.T) + config.WV_end * m.V[config.T]
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    return m

def run_EV_scenarios(plot=True, summary=True):
    """Run Expected Value model analysis."""
    # Build and solve model
    m_ev = build_EV_model()
    solver = config.get_solver()
    
    start_time = time.time()
    result = solver.solve(m_ev)
    solve_time = time.time() - start_time
    
    if summary:
        print("\n" + "="*80)
        print("Expected Value (EV) Model Results")
        print("="*80)
        print(f"Objective value: {pyo.value(m_ev.obj):,.0f} NOK")
        print(f"Solve time: {solve_time:.3f} seconds")
        
        # Get discharge profile statistics
        ev_q = [pyo.value(m_ev.q[t]) for t in range(1, config.T + 1)]
        ev_V = [pyo.value(m_ev.V[t]) for t in range(1, config.T + 1)]
        ev_s = [pyo.value(m_ev.s[t]) for t in range(1, config.T + 1)]
        
        print(f"Discharge statistics:")
        print(f"  Average: {np.mean(ev_q):.2f} m³/s, Max: {np.max(ev_q):.2f} m³/s, Min: {np.min(ev_q):.2f} m³/s")
        print(f"Reservoir level:")
        print(f"  Final: {ev_V[-1]:.2f} Mm³, Max: {np.max(ev_V):.2f} Mm³, Min: {np.min(ev_V):.2f} Mm³")
        print(f"Total spillage: {np.sum(ev_s):.2f} m³/s·h")
        print("="*80)
    
    if plot:
        # Create visualization
        hours_q = list(range(1, config.T + 1))
        hours_V = [0] + hours_q
        
        ev_q_full = [pyo.value(m_ev.q[t]) for t in range(1, config.T + 1)]
        ev_V = [config.V0] + [pyo.value(m_ev.V[t]) for t in range(1, config.T + 1)]
        
        plt.figure(figsize=(11, 4.5))
        ax = plt.gca()
        ax2 = ax.twinx()
        
        lq, = ax.plot(hours_q, ev_q_full, color='red', linewidth=2, label='Discharge q (EV)')
        lV, = ax2.plot(hours_V, ev_V, color='red', linestyle='--', linewidth=2, label='Reservoir V (EV)', zorder=3)
        
        ax.set_title('Expected Value (EV) Model', fontweight='bold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Discharge q (m³/s)')
        ax2.set_ylabel('Reservoir Level V (Mm³)')
        ax.grid(True, linewidth=0.5, alpha=0.6)
        ax.set_xlim(0, config.T)
        ax.axvline(x=config.T1, color='gray', linestyle=':', alpha=0.7, label='End of certain period')
        
        ax.legend(handles=[lq, lV], loc='upper left')
        plt.tight_layout()
        plt.show()
    
    return m_ev