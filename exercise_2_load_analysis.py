# -*- coding: utf-8 -*-
"""
Created on 2023-07-14

@author: ivespe

Intro script for Exercise 2 ("Load analysis to evaluate the need for flexibility") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""

# %% Dependencies

import pandapower as pp
import pandapower.plotting as pp_plotting
import pandas as pd
import os
import load_scenarios as ls
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         =  'C:\\Users\\Birkl\\OneDrive\\Dokumenter\\Fordypningsemne\\7703070'
print(path_data_set)

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')
print(filename_load_data_fullpath)
# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 0.637 

# Maximum load demand of new load being added to the system
P_max_new = 0.4

# Which time series from the load data set that should represent the new load
i_time_series_new_load = 90


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)

# %% Extract hourly load time series for a full year for all the load points in the CINELDI reference system
# (this code is made available for solving task 3)

load_profiles = lp.load_profiles(filename_load_data_fullpath)

# Get all the days of the year
repr_days = list(range(1,366))

# Get normalized load profiles for representative days mapped to buses of the CINELDI reference grid;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Retrieve normalized load time series for new load to be added to the area
new_load_profiles = load_profiles.get_profile_days(repr_days)
new_load_time_series = new_load_profiles[i_time_series_new_load]*P_max_new

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])
# %%
# %% Run power flow
pp.runpp(net)

# Extract all bus voltages
all_bus_indices = net.bus.index.tolist()
all_voltages = net.res_bus["vm_pu"].values
# %%
# Plot all buses
plt.figure(figsize=(10, 6))
plt.plot(all_bus_indices, all_voltages, marker='o', linestyle='-', linewidth=1.5, label="All buses")
# %%
# Highlight your subset of interest
subset_voltages = net.res_bus.loc[bus_i_subset, "vm_pu"].values
plt.plot(bus_i_subset, subset_voltages, marker='s', linestyle='None', color="red", label="Subset buses")

# Formatting
plt.xlabel("Bus index")
plt.ylabel("Voltage magnitude [p.u.]")
plt.title("Voltage Profile of the Grid")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()

# Find bus with lowest voltage magnitude
min_bus = net.res_bus["vm_pu"].idxmin()     # index of bus with lowest voltage
min_voltage = net.res_bus["vm_pu"].min()   # value of lowest voltage

print(f"Lowest voltage is {min_voltage:.4f} p.u. at bus {min_bus}")

max_load_demand = []
max_loads_per_bus = [load_time_series_mapped[bus].max() for bus in bus_i_subset]
aggregated_max_loads = sum(max_loads_per_bus)

print("Max load per bus: ", max_loads_per_bus)
print("Aggregated max load: ", aggregated_max_loads)

scaling_factors = np.linspace(1.0, 2.0, 3)
scaled_max_loads = []

# Backup original loads so we can restore after each run
original_p_mw = net.load['p_mw'].copy()

lowest_voltages = []
aggregated_loads_scaled = []

# Loop over scaling factors
for scale in scaling_factors:
    # Restore original loads
    net.load['p_mw'] = original_p_mw.copy()

    # Scale loads in bus_i_subset proportionally
    for i, bus in enumerate(bus_i_subset):
        # Find load rows in net for this bus
        rows_on_bus = net.load[net.load['bus'] == bus].index
        for row in rows_on_bus:
            net.load.at[row, 'p_mw'] = max_loads_per_bus[i] * scale

    # Run power flow
    pp.runpp(net)

    # Record results
    aggregated_loads_scaled.append(aggregated_max_loads * scale)
    min_voltage = float(net.res_bus.loc[bus_i_subset, 'vm_pu'].min())
    lowest_voltages.append(min_voltage)

# Print results
for s, agg_load, v in zip(scaling_factors, aggregated_loads_scaled, lowest_voltages):
    print(f"Scale {s:.2f} -> Aggregated load = {agg_load:.3f} MW, Lowest voltage = {v:.3f} p.u.")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(aggregated_loads_scaled, lowest_voltages, marker='o', linestyle='-')
plt.axhline(0.95, color='r', linestyle='--', label='Voltage limit 0.95 p.u.')
plt.xlabel("Aggregated Load in Area [MW]")
plt.ylabel("Lowest Voltage in Area [p.u.]")
plt.title("Minimum Voltage vs Aggregated Max Load in Area")
plt.grid(True, alpha=0.6)
plt.legend()
plt.show()



# Extract load demand time series for bus subset
load_subset = load_time_series_mapped[bus_i_subset]

# Plot load demand for subset of buses
plt.figure(figsize=(12, 6))

for bus in bus_i_subset:
    plt.plot(load_subset.index, load_subset[bus], label=f"Bus {bus}")

plt.xlabel("Hour of the Year")
plt.ylabel("Load Demand [MW]")
plt.title("Load Demand Time Series for Selected Buses")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# Sum hourly load demand over the year for bus 90, 91, 92, 96
aggregated_loads = load_time_series_mapped[bus_i_subset].sum()

# Create a table
aggregated_table = pd.DataFrame({
    'Bus': bus_i_subset,
    'Annual Load Demand [MWh]': aggregated_loads.values
})

print(aggregated_table)

# Sum the load at each hour across the subset of buses
aggregated_hourly_load = load_time_series_mapped[bus_i_subset].sum(axis=1)  # sum across columns (buses)

# Find the maximum value and the corresponding hour
max_aggregated_load = aggregated_hourly_load.max()
hour_of_max_load = aggregated_hourly_load.idxmax()

print(f"Maximum aggregated load in area: {max_aggregated_load:.3f} MW")
print(f"Occurs at hour: {hour_of_max_load}")

# Task 5: Load Duration Curve (LDC) for aggregated load time series

# Sort the aggregated load in descending order
ldc = aggregated_hourly_load.sort_values(ascending=False).reset_index(drop=True)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(ldc.values, linewidth=1.2)
plt.xlabel("Hour rank (0 = highest load)")
plt.ylabel("Aggregated Load [MW]")
plt.title("Load Duration Curve for Aggregated Load in Area (Buses 90,91,92,96)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# --- Task 6: Utilization time and coincidence factor ---



# 1) Annual energy per bus (MWh)
annual_energy_per_bus = load_time_series_mapped[bus_i_subset].sum()   # sum of 8760 hourly MW -> MWh

# 2) Peak (annual max) per bus (MW)
peak_per_bus = load_time_series_mapped[bus_i_subset].max()

# 3) Utilization time per bus (hours) = annual energy / peak
#    handle zero-peak safely
utilization_time_per_bus = annual_energy_per_bus / peak_per_bus.replace({0: np.nan})

# 4) Load factor per bus (optional) = utilization_time / hours_per_year
hours_per_year = len( load_time_series_mapped.index )   # should be 8760 for non-leap year
load_factor_per_bus = utilization_time_per_bus / hours_per_year

# 5) Aggregated statistics
annual_energy_aggregated = aggregated_hourly_load.sum()   # MWh
peak_aggregated = aggregated_hourly_load.max()            # MW
utilization_time_aggregated = annual_energy_aggregated / peak_aggregated if peak_aggregated>0 else np.nan
load_factor_aggregated = utilization_time_aggregated / hours_per_year

# 6) Coincidence factor
sum_individual_peaks = peak_per_bus.sum()
coincidence_factor = peak_aggregated / sum_individual_peaks if sum_individual_peaks>0 else np.nan

# 7) Present results in a neat table
summary_df = pd.DataFrame({
    'bus': bus_i_subset,
    'annual_energy_MWh': annual_energy_per_bus.values,
    'peak_MW': peak_per_bus.values,
    'utilization_time_h': utilization_time_per_bus.values,
    'load_factor': load_factor_per_bus.values
})

print("Per-bus utilization / load-factor table:")
print(summary_df.to_string(index=False))

print("\nAggregated area:")
print(f"  Annual energy (sum of hourly) = {annual_energy_aggregated:.3f} MWh")
print(f"  Peak aggregated load (hourly max) = {peak_aggregated:.3f} MW")
print(f"  Utilization time (aggregated) = {utilization_time_aggregated:.1f} h")
print(f"  Load factor (aggregated) = {load_factor_aggregated:.4f} (fraction of year)")
print(f"  Sum of individual peaks = {sum_individual_peaks:.3f} MW")
print(f"  Coincidence factor = {coincidence_factor:.4f}")

# Optional: show a small bar chart of utilization times
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.bar(summary_df['bus'].astype(str), summary_df['utilization_time_h'])
plt.xlabel('Bus')
plt.ylabel('Utilization time [h]')
plt.title('Utilization time (equivalent full-load hours) per bus')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# Task 7: Load Duration Curve with new 0.4 MW load added

# Step 1: Add new load time series to the aggregated area load
aggregated_with_new = aggregated_hourly_load + new_load_time_series

# Step 2: Create load duration curves (sort in descending order)
ldc_original = aggregated_hourly_load.sort_values(ascending=False).reset_index(drop=True)
ldc_with_new = aggregated_with_new.sort_values(ascending=False).reset_index(drop=True)

# Step 3: Plot both for comparison
"""plt.figure(figsize=(10,5))
plt.plot(ldc_original.values, label="Original aggregated load", linewidth=1.2)
plt.plot(ldc_with_new.values, label="With new 0.4 MW load", linewidth=1.2)
plt.xlabel("Hour rank (0 = highest load)")
plt.ylabel("Aggregated Load [MW]")
plt.title("Load Duration Curve for Grid Area (buses 90,91,92,96) \nBefore and After New Load")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()"""

print("Max load in the system with added load: ", max(aggregated_with_new))

# Inputs expected from your workspace:
# aggregated_with_new  -> pandas Series (hourly aggregated MW after adding new_load_time_series)
# P_lim = 0.637

import numpy as np

P_lim = 0.637

# 1) boolean mask for hours exceeding the limit
mask_over = aggregated_with_new > P_lim

# 2) number of hours per year that exceed the limit
n_hours_over = int(mask_over.sum())

# 3) total excess energy (MWh) = sum of (load - P_lim) over hours where load > P_lim
excess_energy_MWh = (aggregated_with_new[mask_over] - P_lim).sum()

# 4) average excess during overloaded hours (MW)
avg_excess_MW = excess_energy_MWh / n_hours_over if n_hours_over > 0 else np.nan

# 5) If you want the maximum instantaneous overload (peak exceedance)
max_excess_MW = (aggregated_with_new - P_lim).max()
max_excess_MW = max_excess_MW if max_excess_MW > 0 else 0.0

print(f"Number of hours per year where aggregated load > P_lim ({P_lim} MW): {n_hours_over} hours")
print(f"Total excess energy above P_lim over the year: {excess_energy_MWh:.4f} MWh")
print(f"Average excess during those hours: {avg_excess_MW:.4f} MW")
print(f"Maximum instantaneous excess above P_lim: {max_excess_MW:.4f} MW")

#task 13

# Case (c): existing load only
ldc_existing = aggregated_hourly_load.sort_values(ascending=False).reset_index(drop=True)

# Case (b): with time-dependent new load
ldc_time_dependent = aggregated_with_new.sort_values(ascending=False).reset_index(drop=True)

# Case (a): with constant new load (0.4 MW flat)
aggregated_with_const_new = aggregated_hourly_load + 0.4
ldc_constant = aggregated_with_const_new.sort_values(ascending=False).reset_index(drop=True)

#Plot the load duration curves
plt.figure(figsize=(10,6))
plt.plot(ldc_existing.values, label="(c) Existing loads only", linewidth=1.2)
plt.plot(ldc_time_dependent.values, label="(b) With 0.4 MW time-dependent load", linewidth=1.2)
plt.plot(ldc_constant.values, label="(a) With 0.4 MW constant load", linewidth=1.2)

plt.xlabel("Hour rank (0 = highest load)")
plt.ylabel("Aggregated Load [MW]")
plt.title("Comparison of Load Duration Curves in Grid Area")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()


# Task 14: Compare utilization times and coincidence factors for cases (a)–(c)

def calc_metrics(load_df, individual_series_list):
    """
    load_df: pandas Series with aggregated hourly load time series [MW]
    individual_series_list: list of pandas Series with each individual load [MW]
    """
    # Annual energy [MWh]
    E_tot = load_df.sum()
    # Maximum aggregated load [MW]
    P_max = load_df.max()
    # Utilization time [h]
    T_util = E_tot / P_max if P_max > 0 else np.nan
    # Sum of individual maximums
    P_individual_sum = sum(s.max() for s in individual_series_list)
    # Coincidence factor
    CF = P_max / P_individual_sum if P_individual_sum > 0 else np.nan

    return E_tot, P_max, T_util, CF


# Case (c): existing loads only
E_c, P_c, T_c, CF_c = calc_metrics(
    aggregated_hourly_load,
    [load_time_series_mapped[bus] for bus in bus_i_subset]
)

# Case (b): with time-dependent new load
E_b, P_b, T_b, CF_b = calc_metrics(
    aggregated_with_new,
    [load_time_series_mapped[bus] for bus in bus_i_subset] + [new_load_time_series]
)

# Case (a): with constant 0.4 MW new load
aggregated_with_const_new = aggregated_hourly_load + 0.4
E_a, P_a, T_a, CF_a = calc_metrics(
    aggregated_with_const_new,
    [load_time_series_mapped[bus] for bus in bus_i_subset] + [pd.Series([0.4] * len(aggregated_hourly_load))]
)

# Put results in a table
comparison_df = pd.DataFrame({
    "Case": ["(c) Existing only", "(b) With time-dependent new load", "(a) With constant new load"],
    "Annual energy [MWh]": [E_c, E_b, E_a],
    "Max load [MW]": [P_c, P_b, P_a],
    "Utilization time [h]": [T_c, T_b, T_a],
    "Coincidence factor [-]": [CF_c, CF_b, CF_a]
})

print(comparison_df.round(3))

# Power flow limit
P_lim = 0.637  # MW

# Case (c): existing only
n_over_c = (aggregated_hourly_load > P_lim).sum()

# Case (b): with time-dependent new load
n_over_b = (aggregated_with_new > P_lim).sum()

# Case (a): with constant new load
aggregated_with_const_new = aggregated_hourly_load + 0.4
n_over_a = (aggregated_with_const_new > P_lim).sum()

# Print results
print("Number of hours per year exceeding 0.637 MW:")
print(f"(c) Existing only:           {n_over_c} h")
print(f"(b) Time-dependent new load: {n_over_b} h")
print(f"(a) Constant 0.4 MW load:    {n_over_a} h")
