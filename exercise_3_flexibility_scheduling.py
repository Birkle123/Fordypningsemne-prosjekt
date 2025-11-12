# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:30:27 2023

@author: merkebud, ivespe

Intro script for Exercise 3 ("Scheduling flexibility resources") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.opt import SolverFactory
from pyomo.core import Var
import pyomo.environ as en
import time

#%% Read battery specifications
parametersinput = pd.read_csv('./battery_data.csv', index_col=0)
parameters = parametersinput.loc[1]

#Parse battery specification
capacity=parameters['Energy_capacity']
charging_power_limit=parameters["Power_capacity"]
discharging_power_limit=parameters["Power_capacity"]
charging_efficiency=parameters["Charging_efficiency"]
discharging_efficiency=parameters["Discharging_efficiency"]
#%% Read load demand and PV production profile data
testData = pd.read_csv('./profile_input.csv')

# Convert the various timeseries/profiles to numpy arrays
Hours = testData['Hours'].values
Base_load = testData['Base_load'].values
PV_prod = testData['PV_prod'].values
Price = testData['Price'].values

# Make dictionaries (for simpler use in Pyomo)
dict_Prices = dict(zip(Hours, Price))
dict_Base_load = dict(zip(Hours, Base_load))
dict_PV_prod = dict(zip(Hours, PV_prod))
# %%

# Define optimization model
model = en.ConcreteModel()

# Sets
model.T = en.RangeSet(1, len(Hours))

# Variables
model.P_charge = en.Var(model.T, within=en.NonNegativeReals, bounds=(0, charging_power_limit))
model.P_discharge = en.Var(model.T, within=en.NonNegativeReals, bounds=(0, discharging_power_limit))
model.Charge = en.Var(model.T, within=en.NonNegativeReals, bounds=(0, capacity))
model.Grid_import = en.Var(model.T, within=en.NonNegativeReals)

# Constraints - Systematic fix with proper time indexing
def charge_rule(model, t):
    if t == 1:
        # Charge[1] is charge at beginning of hour 1 (must be 0)
        # The charge balance for hour 1 is: Charge[2] = Charge[1] + charging - discharging
        return model.Charge[t] == 0  # Battery starts empty
    else:
        # For t >= 2: Charge[t] = charge at beginning of hour t
        # This charge comes from the previous hour's operations
        return model.Charge[t] == model.Charge[t-1] + (model.P_charge[t-1] * charging_efficiency) - (model.P_discharge[t-1] / discharging_efficiency)

# Add end condition separately - charge after last hour operations must be 0
def end_charge_rule(model):
    # After the last hour operations, battery must be empty
    return model.Charge[model.T.last()] + (model.P_charge[model.T.last()] * charging_efficiency) - (model.P_discharge[model.T.last()] / discharging_efficiency) == 0

model.charge_constraint = en.Constraint(model.T, rule=charge_rule)
model.end_charge_constraint = en.Constraint(rule=end_charge_rule)

def grid_import_rule(model, t):
    return model.Grid_import[t] == dict_Base_load[t] - dict_PV_prod[t] + model.P_charge[t] - model.P_discharge[t]
model.Grid_import_constraint = en.Constraint(model.T, rule=grid_import_rule)

# No additional patches needed - systematic fix above handles boundary conditions properly

# Objective function
def objective_rule(model):
    return sum(model.Grid_import[t] * dict_Prices[t] for t in model.T)
model.objective = en.Objective(rule=objective_rule, sense=en.minimize)

# Solve the model
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

def plot_optimization_results(model, Hours, Base_load, PV_prod, Price, capacity, 
                            charging_power_limit, discharging_power_limit):
    """
    Plot and summarize battery optimization results.
    
    Parameters:
    -----------
    model : pyomo.ConcreteModel
        Solved optimization model
    Hours : array
        Time hours array
    Base_load : array
        Base load demand array
    PV_prod : array
        PV production array
    Price : array
        Electricity price array
    capacity : float
        Battery energy capacity
    charging_power_limit : float
        Maximum charging power
    discharging_power_limit : float
        Maximum discharging power
    """
    
    # Extract results from the model
    P_charge_arr = np.array([model.P_charge[t].value for t in model.T])
    P_discharge_arr = np.array([model.P_discharge[t].value for t in model.T])
    Charge_arr = np.array([model.Charge[t].value for t in model.T])
    Grid_import_arr = np.array([model.Grid_import[t].value for t in model.T])

    # Calculate common x-axis scale and grid
    x_min, x_max = Hours.min(), Hours.max()
    
    # Determine common power scale for plots 1 and 3
    all_power_values = np.concatenate([P_charge_arr, P_discharge_arr, Base_load, PV_prod, Grid_import_arr])
    power_max = np.max(np.abs(all_power_values)) * 1.1
    power_min = -power_max if np.any(P_discharge_arr > 0) else 0
    
    # Determine specific scale for plot 1 (battery power) to use full range
    battery_max = max(np.max(P_charge_arr), np.max(P_discharge_arr)) * 1.05
    battery_min = -battery_max if np.any(P_discharge_arr > 0) else 0
    
    # Create comprehensive plots with combined battery plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))  # Reduced to 3 subplots

    # Set up common formatting for all subplots
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(range(int(x_min), int(x_max)+1, 2))  # Ticks every 2 hours
        ax.grid(True, alpha=0.3, which='major')
        ax.grid(True, alpha=0.15, which='minor', linestyle=':')
        ax.set_axisbelow(True)

    # Plot 1: Combined Battery Power and State of Charge
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    # Battery power on left y-axis
    bars1 = ax1.bar(Hours, P_charge_arr, alpha=0.7, color='green', label='Charging Power', width=0.8)
    bars2 = ax1.bar(Hours, -P_discharge_arr, alpha=0.7, color='red', label='Discharging Power', width=0.8)
    ax1.set_ylabel('Battery Power (kW)', color='black')
    ax1.set_ylim(battery_min, battery_max)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Battery charge on right y-axis
    line1, = ax1_twin.plot(Hours, Charge_arr, color='blue', linewidth=3, marker='o', markersize=4, label='State of Charge')
    ax1_twin.set_ylabel('Battery Charge (kWh)', color='blue')
    ax1_twin.set_ylim(0, capacity * 1.1)
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    
    # Combined legend
    lines = [bars1, bars2, line1]
    labels = ['Charging Power', 'Discharging Power', 'State of Charge']
    ax1.legend(lines, labels, loc='upper right')
    ax1.set_title('Battery Power and State of Charge', pad=8)

    # Plot 2: Grid Import vs Load and PV
    axes[1].plot(Hours, Base_load, color='red', linewidth=2, label='Base Load', alpha=0.8)
    axes[1].plot(Hours, PV_prod, color='orange', linewidth=2, label='PV Production', alpha=0.8)
    axes[1].plot(Hours, Grid_import_arr, color='blue', linewidth=2, label='Grid Import (Optimized)', marker='o', markersize=3)
    axes[1].set_ylabel('Power (kW)')
    axes[1].set_title('Load, PV Production, and Optimized Grid Import', pad=8)
    axes[1].legend()
    axes[1].set_ylim(power_min, power_max)

    # Plot 3: Electricity Price
    axes[2].plot(Hours, Price, color='purple', linewidth=2, marker='s', markersize=4)
    axes[2].set_ylabel('Price (currency/kWh)')
    axes[2].set_xlabel('Hour')
    axes[2].set_title('Electricity Price', pad=8)
    # Keep original scale for price as it has different units

    # Adjust layout with more spacing between subplots and show
    plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.08)  # More space between panels, with some top margin
    plt.show()

    # Print optimization summary
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*50)
    print(f"Total cost: {model.objective.expr():.2f} currency units")
    print(f"Battery capacity: {capacity} kWh")
    print(f"Max charging power: {charging_power_limit} kW")
    print(f"Max discharging power: {discharging_power_limit} kW")
    print(f"Total energy charged: {sum(P_charge_arr):.2f} kWh")
    print(f"Total energy discharged: {sum(P_discharge_arr):.2f} kWh")
    print(f"Max battery charge level: {max(Charge_arr):.2f} kWh")
    print(f"Battery utilization: {max(Charge_arr)/capacity*100:.1f}%")
    print("="*50)
    
    return P_charge_arr, P_discharge_arr, Charge_arr, Grid_import_arr

#%% Plot optimization results
plot_optimization_results(model, Hours, Base_load, PV_prod, Price, capacity, 
                         charging_power_limit, discharging_power_limit)


