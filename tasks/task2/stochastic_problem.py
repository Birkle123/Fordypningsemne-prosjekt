import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from config import config

def build_stochastic_model():
    """Build two-stage stochastic model with nonanticipativity."""
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=config.scenarios)
    m.T1 = pyo.RangeSet(1, config.T1)
    m.T2 = pyo.RangeSet(config.T1 + 1, config.T)

    # Shared variables for first stage (nonanticipativity)
    m.q1 = pyo.Var(m.T1, bounds=(0, config.q_cap))
    m.V1 = pyo.Var(m.T1, bounds=(0, config.Vmax))

    # Scenario-specific variables for second stage
    m.q2 = pyo.Var(m.S, m.T2, bounds=(0, config.q_cap))
    m.V2 = pyo.Var(m.S, m.T2, bounds=(0, config.Vmax))

    # Reservoir balance for first stage (shared)
    def res1(m, t):
        if t == 1:
            return m.V1[t] == config.V0 + config.alpha * config.certain_inflow - config.alpha * m.q1[t]
        else:
            return m.V1[t] == m.V1[t - 1] + config.alpha * config.certain_inflow - config.alpha * m.q1[t]
    m.res1 = pyo.Constraint(m.T1, rule=res1)

    # Reservoir balance for second stage (scenario-dependent)
    def res2(m, s, t):
        scenario_inflow = config.scenario_info[s]
        if t == config.T1 + 1:
            return m.V2[s, t] == m.V1[config.T1] + config.alpha * scenario_inflow - config.alpha * m.q2[s, t]
        else:
            return m.V2[s, t] == m.V2[s, t - 1] + config.alpha * scenario_inflow - config.alpha * m.q2[s, t]
    m.res2 = pyo.Constraint(m.S, m.T2, rule=res2)

    # Objective: expected revenue
    def obj(m):
        # First stage revenue (deterministic)
        first = sum(config.pi[t] * 3.6 * config.E_conv * m.q1[t] for t in m.T1)
        
        # Second stage expected revenue
        second = sum(config.prob[s] * (
            sum(config.pi[t] * 3.6 * config.E_conv * m.q2[s, t] for t in m.T2)
            + config.WV_end * m.V2[s, config.T]
        ) for s in m.S)
        
        return first + second
    m.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

    return m

def run_stochastic_problem(plot=True, summary=True):
    """Run two-stage stochastic model analysis."""
    # Build and solve model
    m_stoch = build_stochastic_model()
    solver = config.get_solver()
    result = solver.solve(m_stoch)
    
    if summary:
        print("\n" + "="*80)
        print("Two-Stage Stochastic Model Results")
        print("="*80)
        print(f"Objective value: {pyo.value(m_stoch.obj):,.0f} NOK")
        
        # First stage discharge profile
        stoch_q1 = [pyo.value(m_stoch.q1[t]) for t in range(1, config.T1 + 1)]
        print(f"First stage discharge profile:")
        print("  " + ", ".join(f"{v:6.2f}" for v in stoch_q1))
        print(f"Average first stage discharge: {np.mean(stoch_q1):.2f} m³/s")
        
        # Second stage expected values
        print(f"\nSecond stage expected discharge by hour:")
        for t in range(config.T1 + 1, config.T + 1):
            expected_q = sum(config.prob[s] * pyo.value(m_stoch.q2[s, t]) for s in config.scenarios)
            print(f"  Hour {t}: {expected_q:.2f} m³/s")
        
        print("="*80)
    
    if plot:
        # Create visualization with expected values for second stage
        hours_q = list(range(1, config.T + 1))
        hours_V = [0] + hours_q
        
        # Combine first stage (exact) with second stage (expected)
        stoch_q = []
        stoch_V = [config.V0]
        
        # First stage
        for t in range(1, config.T1 + 1):
            stoch_q.append(pyo.value(m_stoch.q1[t]))
            stoch_V.append(pyo.value(m_stoch.V1[t]))
        
        # Second stage (expected values)
        for t in range(config.T1 + 1, config.T + 1):
            expected_q = sum(config.prob[s] * pyo.value(m_stoch.q2[s, t]) for s in config.scenarios)
            expected_V = sum(config.prob[s] * pyo.value(m_stoch.V2[s, t]) for s in config.scenarios)
            stoch_q.append(expected_q)
            stoch_V.append(expected_V)
        
        plt.figure(figsize=(11, 4.5))
        ax = plt.gca()
        ax2 = ax.twinx()
        
        lq, = ax.plot(hours_q, stoch_q, color='green', linewidth=2, label='Discharge q (Stochastic)')
        lV, = ax2.plot(hours_V, stoch_V, color='green', linestyle='--', linewidth=2, label='Reservoir V (Stochastic)', zorder=3)
        
        ax.set_title('Two-Stage Stochastic Model — shared first stage; expected values on day 2', fontweight='bold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Discharge q (m³/s)')
        ax2.set_ylabel('Reservoir Level V (Mm³)')
        ax.grid(True, linewidth=0.5, alpha=0.6)
        ax.set_xlim(0, config.T)
        ax.axvline(x=config.T1, color='gray', linestyle=':', alpha=0.7, label='End of first stage')
        
        ax.legend(handles=[lq, lV], loc='upper left')
        plt.tight_layout()
        plt.show()
    
    return m_stoch