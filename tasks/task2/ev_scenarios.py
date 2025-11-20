import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from config import config

def build_EV_model():
    """Build Expected Value model using average inflows."""
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, config.T)
    m.q = pyo.Var(m.T, bounds=(0, config.q_cap))
    m.V = pyo.Var(m.T, bounds=(0, config.Vmax))

    # Expected inflow calculation
    I_exp = {}
    for t in range(1, config.T + 1):
        if t <= config.T1:
            I_exp[t] = config.certain_inflow
        else:
            I_exp[t] = sum(config.scenario_inflows) / len(config.scenario_inflows)

    def res_rule(m, t):
        if t == 1:
            return m.V[t] == config.V0 + config.alpha * I_exp[t] - config.alpha * m.q[t]
        else:
            return m.V[t] == m.V[t - 1] + config.alpha * I_exp[t] - config.alpha * m.q[t]

    m.res_balance = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(m):
        return sum(config.pi[t] * 3.6 * config.E_conv * m.q[t] for t in m.T) + config.WV_end * m.V[config.T]
    # pi[eur/MWh] * q[m³/s] * E[kWh/m³] *  = (eur*m³*kWh) / (MWh*s*m³)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    return m

def run_EV_scenarios(plot=True, summary=True):
    """Run Expected Value model analysis."""
    # Build and solve model
    m_ev = build_EV_model()
    solver = config.get_solver()
    result = solver.solve(m_ev)
    
    if summary:
        print("\n" + "="*80)
        print("Expected Value (EV) Model Results")
        print("="*80)
        print(f"Objective value: {pyo.value(m_ev.obj):,.0f} NOK")
        
        # Get discharge profile
        ev_q = [pyo.value(m_ev.q[t]) for t in range(1, 25)]
        print(f"First 24h discharge profile:")
        print("  " + ", ".join(f"{v:6.2f}" for v in ev_q))
        print(f"Average discharge: {np.mean(ev_q):.2f} m³/s")
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