import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from config import config

def build_stochastic_model():
    """Build two-stage stochastic model with nonanticipativity."""
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=config.scenarios)
    m.T1 = pyo.RangeSet(1, config.T1)
    m.T2 = pyo.RangeSet(config.T1 + 1, config.T)

    # Shared variables for first stage
    m.q1 = pyo.Var(m.T1, bounds=(0, config.q_cap)) # discharge
    m.V1 = pyo.Var(m.T1, bounds=(0, config.Vmax)) # reservoir level
    m.s1 = pyo.Var(m.T1, bounds=(0, None))  # spillage

    # Scenario-specific variables for second stage
    m.q2 = pyo.Var(m.S, m.T2, bounds=(0, config.q_cap)) # discharge
    m.V2 = pyo.Var(m.S, m.T2, bounds=(0, config.Vmax)) # reservoir level
    m.s2 = pyo.Var(m.S, m.T2, bounds=(0, None))  # spillage

    # Reservoir balance for first stage
    def res1(m, t):
        if t == 1:
            return m.V1[t] == config.V0 + config.alpha * config.certain_inflow - config.alpha * m.q1[t] - config.alpha * m.s1[t]
        else:
            return m.V1[t] == m.V1[t - 1] + config.alpha * config.certain_inflow - config.alpha * m.q1[t] - config.alpha * m.s1[t]
    m.res1 = pyo.Constraint(m.T1, rule=res1)

    # Reservoir balance for second stage
    def res2(m, s, t):
        scenario_inflow = config.scenario_info[s]
        if t == config.T1 + 1:
            return m.V2[s, t] == m.V1[config.T1] + config.alpha * scenario_inflow - config.alpha * m.q2[s, t] - config.alpha * m.s2[s, t]
        else:
            return m.V2[s, t] == m.V2[s, t - 1] + config.alpha * scenario_inflow - config.alpha * m.q2[s, t] - config.alpha * m.s2[s, t]
    m.res2 = pyo.Constraint(m.S, m.T2, rule=res2)

    # objective function
    def obj(m):
        # First stage revenue (deterministic)
        first_revenue = sum(config.pi[t] * 3.6 * config.E_conv * m.q1[t] for t in m.T1)
        first_spillage_cost = sum(config.spillage_cost * m.s1[t] * config.alpha for t in m.T1)
        
        # Second stage expected revenue (stochastic)
        second = sum(config.prob[s] * (
            sum(config.pi[t] * 3.6 * config.E_conv * m.q2[s, t] for t in m.T2)
            - sum(config.spillage_cost * m.s2[s, t] * config.alpha for t in m.T2)
            + config.WV_end * m.V2[s, config.T]
        ) for s in m.S)
        
        return first_revenue - first_spillage_cost + second
    m.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

    return m

def run_stochastic_problem(plot=True, summary=True):
    """Run two-stage stochastic model analysis."""
    # Build and solve model
    m_stoch = build_stochastic_model()
    solver = config.get_solver()
    
    start_time = time.time()
    result = solver.solve(m_stoch)
    solve_time = time.time() - start_time
    
    if summary:
        print("\n" + "="*80)
        print("Two-Stage Stochastic Model Results")
        print("="*80)
        print(f"Objective value: {pyo.value(m_stoch.obj):,.0f} NOK")
        print(f"Solve time: {solve_time:.3f} seconds")
        
        # First stage statistics
        stoch_q1 = [pyo.value(m_stoch.q1[t]) for t in range(1, config.T1 + 1)]
        stoch_V1 = [pyo.value(m_stoch.V1[t]) for t in range(1, config.T1 + 1)]
        stoch_s1 = [pyo.value(m_stoch.s1[t]) for t in range(1, config.T1 + 1)]
        
        print(f"First stage statistics:")
        print(f"  Discharge - Avg: {np.mean(stoch_q1):.2f} m³/s, Max: {np.max(stoch_q1):.2f} m³/s")
        print(f"  Reservoir - Final: {stoch_V1[-1]:.2f} Mm³, Max: {np.max(stoch_V1):.2f} Mm³")
        print(f"  Total spillage: {np.sum(stoch_s1):.2f} m³/s·h")
        
        # Second stage expected statistics
        expected_q2 = []
        expected_V2 = []
        expected_s2 = []
        for t in range(config.T1 + 1, config.T + 1):
            eq = sum(config.prob[s] * pyo.value(m_stoch.q2[s, t]) for s in config.scenarios)
            ev = sum(config.prob[s] * pyo.value(m_stoch.V2[s, t]) for s in config.scenarios)
            es = sum(config.prob[s] * pyo.value(m_stoch.s2[s, t]) for s in config.scenarios)
            expected_q2.append(eq)
            expected_V2.append(ev)
            expected_s2.append(es)
        
        print(f"Second stage expected statistics:")
        print(f"  Discharge - Avg: {np.mean(expected_q2):.2f} m³/s, Max: {np.max(expected_q2):.2f} m³/s")
        print(f"  Reservoir - Final: {expected_V2[-1]:.2f} Mm³, Max: {np.max(expected_V2):.2f} Mm³")
        print(f"  Total spillage: {np.sum(expected_s2):.2f} m³/s·h")
        
        print("="*80)
    
    if plot:
        # Create visualization with expected values and scenario lines for second stage
        hours_q = list(range(1, config.T + 1))
        hours_V = [0] + hours_q

        # Combine first stage (exact) with second stage (expected)
        stoch_q = []
        stoch_V = [config.V0]

        # First stage exact trajectories
        for t in range(1, config.T1 + 1):
            stoch_q.append(pyo.value(m_stoch.q1[t]))
            stoch_V.append(pyo.value(m_stoch.V1[t]))

        # Scenario-specific second-stage trajectories
        scenario_q = {s: [] for s in config.scenarios}
        scenario_V = {s: [] for s in config.scenarios}

        # Second stage expected + collect scenario raw values
        for t in range(config.T1 + 1, config.T + 1):
            expected_q = 0.0
            expected_V = 0.0
            for s in config.scenarios:
                val_q = pyo.value(m_stoch.q2[s, t])
                val_V = pyo.value(m_stoch.V2[s, t])
                scenario_q[s].append(val_q)
                scenario_V[s].append(val_V)
                expected_q += config.prob[s] * val_q
                expected_V += config.prob[s] * val_V
            stoch_q.append(expected_q)
            stoch_V.append(expected_V)

        plt.figure(figsize=(11, 4.5))
        ax = plt.gca()
        ax2 = ax.twinx()

        # Expected lines
        lq, = ax.plot(hours_q, stoch_q, color='green', linewidth=2, label='Discharge q (expected)')
        lV, = ax2.plot(hours_V, stoch_V, color='green', linestyle='--', linewidth=2, label='Reservoir V (expected)', zorder=3)

        # Scenario thin lines (second stage only, no individual legend labels)
        hours_stage2 = list(range(config.T1 + 1, config.T + 1))
        for s, series in scenario_q.items():
            ax.plot(hours_stage2, series, color='green', alpha=0.4, linewidth=1.0)
        for s, series in scenario_V.items():
            ax2.plot(hours_stage2, series, color='orange', alpha=0.4, linewidth=1.0)

        ax.set_title('Two-Stage Stochastic Model — shared first stage; expected & scenario lines on day 2', fontweight='bold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Discharge q (m³/s)')
        ax2.set_ylabel('Reservoir Level V (Mm³)')
        ax.grid(True, linewidth=0.5, alpha=0.6)
        ax.set_xlim(0, config.T)
        ax.axvline(x=config.T1, color='gray', linestyle=':', alpha=0.7, label='End of first stage')

        # Build combined legend bottom-left with scenario explanation
        handles_ax, labels_ax = ax.get_legend_handles_labels()
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
        scenario_proxy = Line2D([0], [0], color='gray', linewidth=1.0, alpha=0.6,
                     label='Scenario trajectories (thin lines)')
        all_handles = handles_ax + handles_ax2 + [scenario_proxy]
        all_labels = labels_ax + labels_ax2 + [scenario_proxy.get_label()]
        ax.legend(all_handles, all_labels, loc='lower left')
        plt.tight_layout()
        plt.show()
    
    return m_stoch