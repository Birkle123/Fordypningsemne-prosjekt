import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate
import time
from config import config

def build_scenario_model(scenario_name):
    """Build model for a specific scenario."""
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, config.T)
    m.q = pyo.Var(m.T, bounds=(0, config.q_cap)) # discharge
    m.V = pyo.Var(m.T, bounds=(0, config.Vmax)) # reservoir level
    m.s = pyo.Var(m.T, bounds=(0, None))  # spillage

    # Get inflow for this scenario
    scenario_inflow = config.scenario_info[scenario_name]
    
    def res_rule(m, t):
        if t <= config.T1:
            inflow = config.certain_inflow
        else:
            inflow = scenario_inflow
            
        if t == 1:
            return m.V[t] == config.V0 + config.alpha * inflow - config.alpha * m.q[t] - config.alpha * m.s[t]
        else:
            return m.V[t] == m.V[t - 1] + config.alpha * inflow - config.alpha * m.q[t] - config.alpha * m.s[t]

    m.res_balance = pyo.Constraint(m.T, rule=res_rule)

    def obj_rule(m):
        revenue = sum(config.pi[t] * 3.6 * config.E_conv * m.q[t] for t in m.T)
        spillage_cost = sum(config.spillage_cost * m.s[t] * config.alpha for t in m.T)
        return revenue + config.WV_end * m.V[config.T] - spillage_cost

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    return m

def run_individual_scenarios(plot=True, summary=True):
    """Run individual scenario analysis."""
    solver = config.get_solver()
    scenario_results = []
    scenario_qs = {}
    scenario_Vs = {}
    
    # Solve each scenario
    total_start_time = time.time()
    for scenario_name in config.scenarios:
        m_s = build_scenario_model(scenario_name)
        result = solver.solve(m_s)
        
        obj_val = pyo.value(m_s.obj)
        q_vals = [pyo.value(m_s.q[t]) for t in range(1, 25)]
        s_vals = [pyo.value(m_s.s[t]) for t in range(1, config.T + 1)]
        s_total = sum(s_vals)
        scenario_results.append((scenario_name, config.scenario_info[scenario_name], obj_val, q_vals, s_total))
        
        # Store full series for plotting
        scenario_qs[scenario_name] = [pyo.value(m_s.q[t]) for t in range(1, config.T + 1)]
        scenario_Vs[scenario_name] = [config.V0] + [pyo.value(m_s.V[t]) for t in range(1, config.T + 1)]
    
    if summary:
        print("\n" + "="*80)
        total_solve_time = time.time() - total_start_time
        print("Individual Scenario Results")
        print("="*80)
        print(f"Total solve time: {total_solve_time:.3f} seconds ({len(config.scenarios)} scenarios)")
        
        table_data = []
        for scenario_name, inflow, obj_val, q_vals, s_total in scenario_results:
            table_data.append([
                scenario_name,
                f"{inflow:6.1f}",
                f"{obj_val:,.0f}",
                f"{np.mean(q_vals):6.2f}",
                f"{min(q_vals):6.2f}",
                f"{max(q_vals):6.2f}",
                f"{s_total:6.2f}"
            ])
        
        headers = ["Scenario", "Inflow (m³/s)", "Objective (NOK)", "q avg (m³/s)", "q min", "q max", "Spillage"]
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="right"))
        
        # Aggregate info
        objs = [r[2] for r in scenario_results]
        print(f"\nObjective range: {{ {min(objs):,.0f}    {max(objs):,.0f} }}")
        print(f"Average objective: {np.mean(objs):,.0f}")
        print("="*80)
    
    if plot:
        # Create visualization
        hours_q = list(range(1, config.T + 1))
        hours_V = [0] + hours_q
        
        plt.figure(figsize=(11, 6))
        ax = plt.gca()
        ax2 = ax.twinx()
        
        cmap = plt.get_cmap('tab10')
        scenario_lines = []
        
        # Plot in consistent order
        ordered_scenarios = ["Very Dry", "Dry", "Normal", "Wet", "Very Wet"]
        for i, scenario_name in enumerate(ordered_scenarios):
            if scenario_name in scenario_qs:
                c = cmap(i % 10)
                ln, = ax.plot(hours_q, scenario_qs[scenario_name], color=c, linewidth=2, label=scenario_name)
                ax2.plot(hours_V, scenario_Vs[scenario_name], color=c, linestyle='--', linewidth=2, alpha=0.95, zorder=3)
                scenario_lines.append(ln)
        
        ax.set_title('Individual Scenario Models — q (solid), V (dashed)', fontweight='bold')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Discharge q (m³/s)')
        ax2.set_ylabel('Reservoir Level V (Mm³)')
        ax.grid(True, linewidth=0.5, alpha=0.6)
        ax.set_xlim(0, config.T)
        ax.axvline(x=config.T1, color='gray', linestyle=':', alpha=0.7)
        
        # Create legends
        style_key = [
            Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='q (discharge)'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='V (reservoir)'),
        ]
        first_legend = ax.legend(handles=scenario_lines, title='Scenarios', loc='upper left', frameon=True)
        ax.add_artist(first_legend)
        ax.legend(handles=style_key, loc='upper right', frameon=True)
        
        plt.tight_layout()
        plt.show()
    
    return scenario_results