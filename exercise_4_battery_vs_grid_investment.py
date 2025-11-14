# -*- coding: utf-8 -*-
"""
Created on 2023-10-10

@author: ivespe

Intro script for Exercise 4 ("Battery energy storage system in the grid vs. grid investments") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""


# %% Dependencies

import pandas as pd
import os
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         = 'C:/Users/ivespe/Data_sets/CINELDI_MV_reference_system/'
path_data_set         = '/Users/tarjeireite/programmering/fleksibilitet/CINELDI_MV_reference_system/'

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')
filename_standard_overhead_lines = os.path.join(path_data_set,'standard_overhead_line_types.csv')
filename_reldata = os.path.join(path_data_set,'reldata_for_component_types.csv')
filename_load_point = os.path.join(path_data_set,'CINELDI_MV_reference_system_load_point.csv')

# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 4

# Factor to scale the loads for this exercise compared with the base version of the CINELDI reference system data set
scaling_factor = 10

# Read standard data for overhead lines
data_standard_overhead_lines = pd.read_csv(filename_standard_overhead_lines, delimiter=';')
data_standard_overhead_lines.set_index(keys = 'type', drop = True, inplace = True)

# Read standard component reliability data
data_comp_rel = pd.read_csv(filename_reldata, delimiter=';')
data_comp_rel.set_index(keys = 'main_type', drop = True, inplace = True)

# Read load point data (incl. specific rates of costs of energy not supplied) for data
data_load_point = pd.read_csv(filename_load_point, delimiter=';')
data_load_point.set_index(keys = 'bus_i', drop = True, inplace = True)


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)


# %% Set up hourly normalized load time series for a representative day (task 2; this code is provided to the students)

load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Consider only the day with the peak load in the area (28 February)
repr_days = [31+28]

# Get relative load profiles for representative days mapped to buses of the CINELDI test network;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])


# %% Aggregate the load demand in the area

# Aggregated load time series for the subset of load buses
load_time_series_subset = load_time_series_mapped[bus_i_subset] * scaling_factor
load_time_series_subset_aggr = load_time_series_subset.sum(axis=1)

P_max = load_time_series_subset_aggr.max()


## Task 2: Plotting a 3% increase over ten years.
years = 10
increase_rate = 0.03
peak_loads = [P_max * (1 + increase_rate) ** year for year in range(years)]
plot = False

def plot_peak_loads_step(years, peak_loads, limit_mw=5):
    year_range = list(range(0, years + 1))
    
    peak_loads_extended = peak_loads + [peak_loads[-1]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.step(year_range, peak_loads_extended, where='post', linewidth=2.5, color='blue', 
            label='Peak Load')
    
    ax.axhline(y=limit_mw, color='red', linestyle='--', linewidth=2, 
               label=f'{limit_mw} MW Limit', alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Peak Load (MW)')
    ax.set_title('Peak Load Development Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax.set_xticks(year_range)
    ax.set_xlim(-0.5, years + 0.5)
    
    y_min = min(min(peak_loads), limit_mw) * 0.9
    y_max = max(max(peak_loads), limit_mw) * 1.1
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()

if plot:
    plot_peak_loads_step(years, peak_loads)


# Task 7:
limit = P_lim
cost_pr_mwh = 2000
sum_costs = 0
if False:
    for year in range(years-1):
        adjusted_time_series = load_time_series_subset_aggr * (1 + increase_rate) ** year
        # Finding how much the limit is exceeded, summed up for every hour
        exceedance = adjusted_time_series[adjusted_time_series > limit] - limit
        total_exceedance = exceedance.sum()
        total_exceedance_adjusted = total_exceedance * 20
        total_cost = total_exceedance_adjusted * cost_pr_mwh
        sum_costs += total_cost
        print(f"Year {year+1:2d}: Total exceedance = {total_exceedance_adjusted:8.2f} MWh, Total cost = {total_cost:12.2f} NOK")
    print(f"\nTotal cost over {years} years: {sum_costs:12.2f} NOK")

# Task 8: Annual energy not supplied calculation
average_load_yearly = 1.841
lambda_perm = data_comp_rel.loc['Overhead line (1–22 kV)', "lambda_perm"]
r_perm = data_comp_rel.loc['Overhead line (1–22 kV)', "r_perm"]

# Task 9:
days_we_need_to_consider = list(range(0, 366))
load_profiles_for_every_day_in_the_year = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,days_we_need_to_consider)
load_profiles_now_scaled_to_peak_power = load_profiles_for_every_day_in_the_year.mul(net.load['p_mw'])
scaled_load_profiles_our_buses = load_profiles_now_scaled_to_peak_power[bus_i_subset] * scaling_factor
bus_90_time_series = scaled_load_profiles_our_buses[90]
bus_91_time_series = scaled_load_profiles_our_buses[91]
bus_92_time_series = scaled_load_profiles_our_buses[92]
bus_96_time_series = scaled_load_profiles_our_buses[96]

bus_total_time_series = bus_90_time_series + bus_91_time_series + bus_92_time_series + bus_96_time_series
sum_total_time_series = bus_total_time_series.sum()
avg_yearly_load = sum_total_time_series / 8760

# Calculate individual bus averages and percentages
bus_data = {
    90: bus_90_time_series,
    91: bus_91_time_series,  
    92: bus_92_time_series,
    96: bus_96_time_series
}

bus_averages = {}
bus_percentages = {}

for bus_num, time_series in bus_data.items():
    bus_sum = time_series.sum()
    bus_avg = bus_sum / 8760
    bus_averages[bus_num] = bus_avg
    bus_percentages[bus_num] = (bus_sum / sum_total_time_series) * 100

if False:
    # Pretty printing with aligned columns
    print("\n" + "="*60)
    print("           LOAD ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Bus':<6} {'Average Load':<15} {'Contribution':<12} {'Percentage':<10}")
    print("-"*60)

    for bus_num in sorted(bus_data.keys()):
        avg = bus_averages[bus_num]
        pct = bus_percentages[bus_num]
        print(f"{bus_num:<6} {avg:<12.3f} MW  {pct:<8.1f}%     {'█' * int(pct/2)}")

    print("-"*60)
    print(f"{'Total':<6} {avg_yearly_load:<12.3f} MW  {sum(bus_percentages.values()):<8.1f}%")
    print("="*60)

if False:
    # Calculate weighted cost_pr_kwh using 4h values from load point data
    cost_pr_kwh = 0
    for bus_num in bus_i_subset:
        # Get the 4h cost value for this bus from the load point data
        cost_4h = data_load_point.loc[bus_num, 'c_NOK_per_kWh_4h']
        # Weight it by the percentage contribution of this bus
        weighted_cost = cost_4h * (bus_percentages[bus_num] / 100)
        cost_pr_kwh += weighted_cost
        print(f"Bus {bus_num}: 4h cost = {cost_4h:.2f} NOK/kWh, weight = {bus_percentages[bus_num]:.1f}%, contribution = {weighted_cost:.2f} NOK/kWh")

    print(f"\nWeighted average cost_pr_kwh = {cost_pr_kwh:.2f} NOK/kWh")
    print("-"*60)
    for year in range(years):
        EENS = (avg_yearly_load) * ((1 + increase_rate) ** year) * (lambda_perm/5) * (r_perm)
        CENS = EENS * cost_pr_kwh * 1000  # Convert MWh to kWh
        print(f"Year {year+1:2d}: EENS = {EENS:.3f} MWh, CENS = {CENS:.2f} NOK")


# Task 10: Estimating CENS with battery installation
if False:
    # Calculate weighted cost_pr_kwh using 4h values from load point data
    cost_pr_kwh = 0
    for bus_num in bus_i_subset:
        # Get the 4h cost value for this bus from the load point data
        cost_4h = data_load_point.loc[bus_num, 'c_NOK_per_kWh_4h']
        # Weight it by the percentage contribution of this bus
        weighted_cost = cost_4h * (bus_percentages[bus_num] / 100)
        cost_pr_kwh += weighted_cost
        print(f"Bus {bus_num}: 4h cost = {cost_4h:.2f} NOK/kWh, weight = {bus_percentages[bus_num]:.1f}%, contribution = {weighted_cost:.2f} NOK/kWh")

    print(f"\nWeighted average cost_pr_kwh = {cost_pr_kwh:.2f} NOK/kWh")
    print("-"*60)
    for year in range(years):
        EENS = (avg_yearly_load) * ((1 + increase_rate) ** year) * (lambda_perm/5) * (r_perm-(2/((avg_yearly_load) * ((1 + increase_rate) ** year))))
        CENS = EENS * cost_pr_kwh * 1000  # Convert MWh to kWh
        print(f"{CENS:.2f}")

# Task 11: Battery optimization model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.opt import SolverFactory
from pyomo.core import Var
import pyomo.environ as en
import time

parametersinput = pd.read_csv('./battery_data.csv', index_col=0)
parameters = parametersinput.loc[1]

#Parse battery specification
capacity=2
charging_power_limit=1
discharging_power_limit=1
charging_efficiency=parameters["Charging_efficiency"]
discharging_efficiency=parameters["Discharging_efficiency"]
#%% Read load demand and PV production profile data
testData = pd.read_csv('./profile_input.csv')

# Convert the various timeseries/profiles to numpy arrays
Hours = testData['Hours'].values
Base_load = load_time_series_subset_aggr.values * ((1+ increase_rate) ** 5)
PV_prod = np.zeros_like(Base_load)
Price = testData['Price'].values
max_grid_import = P_lim

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

# Constraints
def charge_rule(model, t):
    if t == 1:
        # Charge[1] is charge at beginning of hour 1 (must be 0)
        # The charge balance for hour 1 is: Charge[2] = Charge[1] + charging - discharging
        return model.Charge[t] == 0  # Battery starts empty
    else:
        # For t >= 2: Charge[t] = charge at beginning of hour t
        # This charge comes from the previous hour's operations
        return model.Charge[t] == model.Charge[t-1] + (model.P_charge[t-1] * charging_efficiency) - (model.P_discharge[t-1] / discharging_efficiency)

def end_charge_rule(model):
    # After the last hour operations, battery must be empty
    return model.Charge[model.T.last()] + (model.P_charge[model.T.last()] * charging_efficiency) - (model.P_discharge[model.T.last()] / discharging_efficiency) == 0

def grid_import_rule(model, t):
    return model.Grid_import[t] <= max_grid_import

model.charge_constraint = en.Constraint(model.T, rule=charge_rule)
model.end_charge_constraint = en.Constraint(rule=end_charge_rule)
model.Grid_import_limit_constraint = en.Constraint(model.T, rule=grid_import_rule)

def grid_import_rule(model, t):
    return model.Grid_import[t] == dict_Base_load[t] - dict_PV_prod[t] + model.P_charge[t] - model.P_discharge[t]
model.Grid_import_constraint = en.Constraint(model.T, rule=grid_import_rule)



# Objective function
def objective_rule(model):
    return sum(model.Grid_import[t] * dict_Prices[t] for t in model.T)
model.objective = en.Objective(rule=objective_rule, sense=en.minimize)

# Solve Task 11 model (original without penalty)
solver = SolverFactory('gurobi')
results_task11 = solver.solve(model, tee=True)
model_task11 = model  # Store reference to Task 11 model

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
    
    # Determine specific scale for plot 1 (battery power) to use full range
    battery_max = max(np.max(P_charge_arr), np.max(P_discharge_arr)) * 1.05
    battery_min = -battery_max if np.any(P_discharge_arr > 0) else 0
    
    # Determine common power scale for plots 2
    all_power_values = np.concatenate([Base_load, PV_prod, Grid_import_arr])
    power_max = np.max(np.abs(all_power_values)) * 1.1
    power_min = -power_max if np.any(PV_prod > 0) else 0
    
    # Create single plot with load, grid import, and battery charge
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Set up formatting
    ax1.set_xlim(x_min, x_max)
    ax1.set_xticks(range(int(x_min), int(x_max)+1, 2))  # Ticks every 2 hours
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax1.set_axisbelow(True)
    
    # Left y-axis: Power (Load and Grid Import)
    ax1.plot(Hours, Base_load, color='red', linewidth=2.5, label='Base Load', alpha=0.8)
    ax1.plot(Hours, Grid_import_arr, color='blue', linewidth=2.5, label='Grid Import (Optimized)', marker='o', markersize=4)
    
    # Scale price to fit nicely in the power range (show shape only)
    power_range = np.max(Base_load) - np.min(Base_load)
    price_scaled = (Price - Price.min()) / (Price.max() - Price.min()) * power_range * 0.3 + np.min(Base_load)
    ax1.plot(Hours, price_scaled, color='orange', linewidth=2, linestyle='--', alpha=0.7, label='Price (scaled)')
    
    ax1.axhline(y=max_grid_import, color='purple', linestyle='--', linewidth=2, label=f'{max_grid_import} MW Limit', alpha=0.8)
    ax1.set_ylabel('Power (MW)', color='black', fontsize=12)
    ax1.set_xlabel('Hour', fontsize=12)
    
    # Right y-axis: Battery Charge
    ax2 = ax1.twinx()
    
    # Battery charge
    line_battery = ax2.plot(Hours, Charge_arr, color='green', linewidth=3, marker='s', markersize=5, label='Battery Charge', alpha=0.9)
    
    ax2.set_ylabel('Battery Charge (MWh)', color='black', fontsize=12)
    ax2.set_ylim(0, capacity * 1.1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    # Title
    ax1.set_title('Load, Grid Import, and Battery Operation', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.show()

    # Print optimization summary
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*50)
    print(f"Total cost: {model.objective.expr():.2f} currency units")
    print(f"Battery capacity: {capacity} MWh")
    print(f"Max charging power: {charging_power_limit} MW")
    print(f"Max discharging power: {discharging_power_limit} MW")
    print(f"Total energy charged: {sum(P_charge_arr):.2f} MWh")
    print(f"Total energy discharged: {sum(P_discharge_arr):.2f} MWh")
    print(f"Max battery charge level: {max(Charge_arr):.2f} MWh")
    print(f"Battery utilization: {max(Charge_arr)/capacity*100:.1f}%")
    print(f"Grid limit violations: {sum(1 for x in Grid_import_arr if x > max_grid_import)}")
    print("="*50)
    
    return P_charge_arr, P_discharge_arr, Charge_arr, Grid_import_arr

if False:
    # Call the plotting function
    plot_optimization_results(model, Hours, Base_load, PV_prod, Price, capacity, 
                             charging_power_limit, discharging_power_limit)






# ---------------------------------------------------------------------------
# Task 12: Pushing the battery to its limits
parametersinput = pd.read_csv('./battery_data.csv', index_col=0)
parameters = parametersinput.loc[1]

#Parse battery specification
capacity=2
charging_power_limit=1
discharging_power_limit=1
charging_efficiency=parameters["Charging_efficiency"]
discharging_efficiency=parameters["Discharging_efficiency"]
penalty_cost = 1.0  # Cost for exceeding grid import limit (moderate penalty to show difference)
#%% Read load demand and PV production profile data
testData = pd.read_csv('./profile_input.csv')

# Convert the various timeseries/profiles to numpy arrays
Hours = testData['Hours'].values
Base_load = load_time_series_subset_aggr.values * ((1+ increase_rate) ** 6)
PV_prod = np.zeros_like(Base_load)
Price = testData['Price'].values
max_grid_import = P_lim

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
model.Grid_excess = en.Var(model.T, within=en.NonNegativeReals)  # Variable for exceeding grid import limit

# Constraints
def charge_rule(model, t):
    if t == 1:
        # Charge[1] is charge at beginning of hour 1 (must be 0)
        # The charge balance for hour 1 is: Charge[2] = Charge[1] + charging - discharging
        return model.Charge[t] == 0  # Battery starts empty
    else:
        # For t >= 2: Charge[t] = charge at beginning of hour t
        # This charge comes from the previous hour's operations
        return model.Charge[t] == model.Charge[t-1] + (model.P_charge[t-1] * charging_efficiency) - (model.P_discharge[t-1] / discharging_efficiency)

def end_charge_rule(model):
    # After the last hour operations, battery must be empty
    return model.Charge[model.T.last()] + (model.P_charge[model.T.last()] * charging_efficiency) - (model.P_discharge[model.T.last()] / discharging_efficiency) == 0

def grid_import_rule(model, t):
    return model.Grid_import[t] <= max_grid_import + model.Grid_excess[t]

model.charge_constraint = en.Constraint(model.T, rule=charge_rule)
model.end_charge_constraint = en.Constraint(rule=end_charge_rule)
model.Grid_import_limit_constraint = en.Constraint(model.T, rule=grid_import_rule)

def grid_import_balance_rule(model, t):
    return model.Grid_import[t] == dict_Base_load[t] - dict_PV_prod[t] + model.P_charge[t] - model.P_discharge[t]
model.Grid_import_constraint = en.Constraint(model.T, rule=grid_import_balance_rule)



# Objective function
def objective_rule(model):
    return sum(model.Grid_import[t] * dict_Prices[t] + model.Grid_excess[t] * penalty_cost for t in model.T)
model.objective = en.Objective(rule=objective_rule, sense=en.minimize)

# Solve Task 12 model (with penalty cost)
solver = SolverFactory('gurobi')
results_task12 = solver.solve(model, tee=True)
model_task12 = model  # Store reference to Task 12 model

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
    Grid_excess_arr = np.array([model.Grid_excess[t].value for t in model.T])

    # Calculate common x-axis scale and grid
    x_min, x_max = Hours.min(), Hours.max()
    
    # Determine specific scale for plot 1 (battery power) to use full range
    battery_max = max(np.max(P_charge_arr), np.max(P_discharge_arr)) * 1.05
    battery_min = -battery_max if np.any(P_discharge_arr > 0) else 0
    
    # Determine common power scale for plots 2
    all_power_values = np.concatenate([Base_load, PV_prod, Grid_import_arr])
    power_max = np.max(np.abs(all_power_values)) * 1.1
    power_min = -power_max if np.any(PV_prod > 0) else 0
    
    # Create single plot with load, grid import, and battery charge
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Set up formatting
    ax1.set_xlim(x_min, x_max)
    ax1.set_xticks(range(int(x_min), int(x_max)+1, 2))  # Ticks every 2 hours
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.15, which='minor', linestyle=':')
    ax1.set_axisbelow(True)
    
    # Left y-axis: Power (Load and Grid Import)
    ax1.plot(Hours, Base_load, color='red', linewidth=2.5, label='Base Load', alpha=0.8)
    ax1.plot(Hours, Grid_import_arr, color='blue', linewidth=2.5, label='Grid Import (Optimized)', marker='o', markersize=4)
    
    # Scale price to fit nicely in the power range (show shape only)
    power_range = np.max(Base_load) - np.min(Base_load)
    price_scaled = (Price - Price.min()) / (Price.max() - Price.min()) * power_range * 0.3 + np.min(Base_load)
    ax1.plot(Hours, price_scaled, color='orange', linewidth=2, linestyle='--', alpha=0.7, label='Price (scaled)')
    
    ax1.axhline(y=max_grid_import, color='purple', linestyle='--', linewidth=2, label=f'{max_grid_import} MW Limit', alpha=0.8)
    
    ax1.set_ylabel('Power (MW)', color='black', fontsize=12)
    ax1.set_xlabel('Hour', fontsize=12)
    
    # Right y-axis: Battery Charge
    ax2 = ax1.twinx()
    
    # Battery charge
    line_battery = ax2.plot(Hours, Charge_arr, color='green', linewidth=3, marker='s', markersize=5, label='Battery Charge', alpha=0.9)
    
    ax2.set_ylabel('Battery Charge (MWh)', color='black', fontsize=12)
    ax2.set_ylim(0, capacity * 1.1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    # Title
    ax1.set_title('Load, Grid Import, and Battery Operation', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.show()

    # Print optimization summary
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*50)
    print(f"Total cost: {model.objective.expr():.2f} currency units")
    print(f"Battery capacity: {capacity} MWh")
    print(f"Max charging power: {charging_power_limit} MW")
    print(f"Max discharging power: {discharging_power_limit} MW")
    print(f"Total energy charged: {sum(P_charge_arr):.2f} MWh")
    print(f"Total energy discharged: {sum(P_discharge_arr):.2f} MWh")
    print(f"Max battery charge level: {max(Charge_arr):.2f} MWh")
    print(f"Battery utilization: {max(Charge_arr)/capacity*100:.1f}%")
    print(f"Grid limit violations: {sum(1 for x in Grid_import_arr if x > max_grid_import)}")
    
    # EXCESS COST ACTIVATION ANALYSIS
    excess_hours = Hours[Grid_excess_arr > 0.001]
    total_excess_cost = np.sum(Grid_excess_arr * penalty_cost)
    
    if len(excess_hours) > 0:
        print("\n EXCESS COST ACTIVATED")
        print(f"Hours when penalty cost applied: {list(excess_hours)}")
        print(f"Excess amounts (MW): {Grid_excess_arr[Grid_excess_arr > 0.001]}")
        print(f"Total excess cost: {total_excess_cost:.2f} currency units")
        print(f"Penalty cost per MW: {penalty_cost} currency units/MW")
    else:
        print("\n No excess cost activated - stayed within grid limits!")
        print(f"Penalty cost available but not needed: {penalty_cost} currency units/MW")
    
    print("="*50)
    
    return P_charge_arr, P_discharge_arr, Charge_arr, Grid_import_arr

def plot_comparison_results(model_task11, model_task12, Hours, Base_load_task12, PV_prod, Price, capacity, 
                          charging_power_limit, discharging_power_limit, max_grid_import, penalty_cost):
    """
    Plot and compare Task 11 (original) vs Task 12 (with penalty) optimization results.
    """
    
    # Calculate the different base loads for each task
    # Task 11 uses ((1+ increase_rate) ** 5), Task 12 uses ((1+ increase_rate) ** 6) 
    Base_load_task11 = Base_load_task12 / (1 + increase_rate)  # Back-calculate Task 11 base load
    
    # Extract results from Task 11 model (original)
    P_charge_task11 = np.array([model_task11.P_charge[t].value for t in model_task11.T])
    P_discharge_task11 = np.array([model_task11.P_discharge[t].value for t in model_task11.T])
    Charge_task11 = np.array([model_task11.Charge[t].value for t in model_task11.T])
    Grid_import_task11 = np.array([model_task11.Grid_import[t].value for t in model_task11.T])
    
    # Extract results from Task 12 model (with penalty)
    P_charge_task12 = np.array([model_task12.P_charge[t].value for t in model_task12.T])
    P_discharge_task12 = np.array([model_task12.P_discharge[t].value for t in model_task12.T])
    Charge_task12 = np.array([model_task12.Charge[t].value for t in model_task12.T])
    Grid_import_task12 = np.array([model_task12.Grid_import[t].value for t in model_task12.T])
    Grid_excess_task12 = np.array([model_task12.Grid_excess[t].value for t in model_task12.T])
    
    # Create comparison plot
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Set up formatting  
    x_min, x_max = Hours.min(), Hours.max()
    ax1.set_xlim(x_min, x_max)
    ax1.set_xticks(range(int(x_min), int(x_max)+1, 2))
    ax1.grid(True, alpha=0.3, which='major')
    ax1.set_axisbelow(True)
    
    # Plot both base loads
    ax1.plot(Hours, Base_load_task11, color='red', linewidth=2, alpha=0.4, linestyle='-', 
             label='Task 11: Base Load (Year 5)')
    ax1.plot(Hours, Base_load_task12, color='red', linewidth=3, alpha=0.9, linestyle='-', 
             label='Task 12: Base Load (Year 6)')
    
    # Plot Task 11 results (weaker colors)
    ax1.plot(Hours, Grid_import_task11, color='blue', linewidth=2, alpha=0.4, linestyle='-', 
             label='Task 11: Grid Import')
    
    # Plot Task 12 results (stronger colors)
    ax1.plot(Hours, Grid_import_task12, color='blue', linewidth=3, alpha=0.9, linestyle='-', 
             label='Task 12: Grid Import')
    
    # Add grid limit line
    ax1.axhline(y=max_grid_import, color='purple', linestyle='--', linewidth=2, 
                label=f'{max_grid_import} MW Grid Limit', alpha=0.8)
    
    # Scale and plot price curve
    power_range = np.max(Base_load) - np.min(Base_load)
    price_scaled = (Price - Price.min()) / (Price.max() - Price.min()) * power_range * 0.3 + np.min(Base_load)
    ax1.plot(Hours, price_scaled, color='orange', linewidth=2, linestyle='--', alpha=0.7, label='Price (scaled)')
    
    ax1.set_ylabel('Power (MW)', color='black', fontsize=12)
    ax1.set_xlabel('Hour', fontsize=12)
    
    # Right y-axis: Battery Charges
    ax2 = ax1.twinx()
    
    # Plot battery charges
    ax2.plot(Hours, Charge_task11, color='green', linewidth=2, alpha=0.4, linestyle='-', 
             marker='s', markersize=3, label='Task 11: Battery Charge')
    ax2.plot(Hours, Charge_task12, color='green', linewidth=3, alpha=0.9, linestyle='-', 
             marker='s', markersize=5, label='Task 12: Battery Charge')
    
    ax2.set_ylabel('Battery Charge (MWh)', color='black', fontsize=12)
    ax2.set_ylim(0, capacity * 1.1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Title
    ax1.set_title('Comparison: Task 11 (Year 5) vs Task 12 (Year 6) with Growing Base Load', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    total_cost_task11 = model_task11.objective.expr()
    total_cost_task12 = model_task12.objective.expr()
    total_excess_cost = np.sum(Grid_excess_task12 * penalty_cost)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY: TASK 11 vs TASK 12")
    print("="*60)
    print(f"Task 11 (Original) total cost:    {total_cost_task11:.2f} currency units")
    print(f"Task 12 (With penalty) total cost: {total_cost_task12:.2f} currency units")
    print(f"Cost difference:                   {total_cost_task12 - total_cost_task11:.2f} currency units")
    
    if total_excess_cost > 0:
        print(f"\n🚨 Task 12 used penalty cost:      {total_excess_cost:.2f} currency units")
        excess_hours = Hours[Grid_excess_task12 > 0.001]
        print(f"Hours with excess:                 {list(excess_hours)}")
    else:
        print(f"\n✅ Task 12 stayed within limits without using penalty cost")
    
    print("="*60)

if True:
    # Call the comparison plotting function
    plot_comparison_results(model_task11, model_task12, Hours, Base_load, PV_prod, Price, capacity, 
                          charging_power_limit, discharging_power_limit, max_grid_import, penalty_cost)