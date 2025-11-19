import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_scenario_data():
    """
    Plot price and inflow data for all scenarios from CSV file.
    Creates a comprehensive visualization showing both price evolution and inflow scenarios.
    """
    # Load the data
    df = pd.read_csv('scenario_data.csv')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Price evolution over 48 hours
    ax1.plot(df['hour'], df['price'], 'b-', linewidth=3, label='Price (NOK/MWh)')
    ax1.set_title('Price Evolution Over 48 Hours', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Price (NOK/MWh)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=24.5, color='red', linestyle='--', alpha=0.7, label='End of first stage')
    ax1.legend()
    ax1.set_xlim(1, 48)
    
    # Plot 2: Inflow scenarios
    scenario_colors = ['#8B0000', '#FF4500', '#FFD700', '#32CD32', '#4169E1']  # Dark red to blue
    scenario_names = ['Very Dry', 'Dry', 'Normal', 'Wet', 'Very Wet']
    inflow_columns = ['inflow_very_dry', 'inflow_dry', 'inflow_normal', 'inflow_wet', 'inflow_very_wet']
    
    for i, (col, name, color) in enumerate(zip(inflow_columns, scenario_names, scenario_colors)):
        ax2.plot(df['hour'], df[col], color=color, linewidth=2.5, label=f'{name} ({df[col].iloc[-1]} m³/s)', alpha=0.8)
    
    ax2.set_title('Inflow Scenarios Over 48 Hours', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Inflow (m³/s)')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=24.5, color='red', linestyle='--', alpha=0.7, label='Start of uncertainty')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlim(1, 48)
    ax2.set_ylim(-5, 55)
    
    plt.tight_layout()
    plt.show()
    
    return df

if __name__ == "__main__":
    # Plot the main scenario data
    df = plot_scenario_data()
