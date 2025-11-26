import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import config, T, T1, pi, scenario_info, scenarios, prob

def _build_dataframe_from_config() -> pd.DataFrame:
    """Construct a DataFrame with hour, price and per-scenario inflows using central config."""
    hours = list(range(1, T + 1))
    price_series = [pi[t] for t in hours]
    data = {
        'hour': hours,
        'price': price_series,
    }
    # For each scenario: inflow is certain_inflow for first stage, then scenario-specific inflow value
    certain_inflow = config.certain_inflow
    for s in scenarios:
        col_name = 'inflow_' + s.lower().replace(' ', '_')
        inflow_value = scenario_info[s]
        inflow_series = [certain_inflow] * T1 + [inflow_value] * (T - T1)
        data[col_name] = inflow_series
    return pd.DataFrame(data)

def plot_scenario_data():
    """Plot price evolution and inflow scenarios sourced from config (no CSV dependency)."""
    df = _build_dataframe_from_config()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Price plot
    ax1.plot(df['hour'], df['price'], 'b-', linewidth=3, label='Price (NOK/MWh)')
    ax1.set_title(f'Price Evolution Over {T} Hours', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Price (NOK/MWh)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=T1 + 0.5, color='red', linestyle='--', alpha=0.7, label='End of first stage')
    ax1.legend()
    ax1.set_xlim(1, T)

    # Inflow scenarios plot
    scenario_colors = ['#8B0000', '#FF4500', '#FFD700', '#32CD32', '#4169E1']
    for color, s in zip(scenario_colors, scenarios):
        col_name = 'inflow_' + s.lower().replace(' ', '_')
        inflow_final = scenario_info[s]
        probability = prob[s]
        label = f"{s} (Stage2 {inflow_final} m³/s, p={probability:.2f})"
        ax2.plot(df['hour'], df[col_name], color=color, linewidth=2.5, alpha=0.85, label=label)

    ax2.set_title(f'Inflow Scenarios Over {T} Hours', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Inflow (m³/s)')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=T1 + 0.5, color='red', linestyle='--', alpha=0.7, label='Start of uncertainty')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlim(1, T)
    # Y-limits: include certain_inflow and scenario inflows comfortably
    inflow_values = [config.certain_inflow] + list(scenario_info.values())
    ymin = min(inflow_values) - 5
    ymax = max(inflow_values) + 15
    ax2.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()
    return df

if __name__ == "__main__":
    # Plot the main scenario data
    df = plot_scenario_data()
