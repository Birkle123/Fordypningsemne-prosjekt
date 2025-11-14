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

#Maximum power imported from the grid
P_lim = 6  # kW

# Create Pyomo model
model = en.ConcreteModel()
model.T = en.RangeSet(1, len(Hours))  # Time index set
model.dt = 1  # Time step in hours

# Decision variables
model.P_charge = en.Var(model.T, within=en.NonNegativeReals, bounds=(0, charging_power_limit))  # Charging power (kW)
model.P_discharge = en.Var(model.T, within=en.NonNegativeReals, bounds=(0, discharging_power_limit))  # Discharging power (kW)
model.E = en.Var(model.T, within=en.NonNegativeReals, bounds=(0, capacity))  # Energy stored in battery (kWh)   
model.P_grid = en.Var(model.T, within=en.Reals, bounds=(None, P_lim))  # Power drawn from/supplied to the grid (kW)

# Objective: Minimize total cost of electricity over the time horizon
def objective_rule(model):
    return sum(model.P_grid[t] * dict_Prices[t] * model.dt for t in model.T)
model.Objective = en.Objective(rule=objective_rule, sense=en.minimize)

# Constraints
def energy_balance_rule(model, t):
    if t == 1:
        return model.E[t] == 0 + (model.P_charge[t] * charging_efficiency - model.P_discharge[t] / discharging_efficiency) * model.dt
    else:
        return model.E[t] == model.E[t-1] + (model.P_charge[t] * charging_efficiency - model.P_discharge[t] / discharging_efficiency) * model.dt
model.EnergyBalance = en.Constraint(model.T, rule=energy_balance_rule)

def power_balance_rule(model, t):
    return model.P_grid[t] + model.P_discharge[t] + dict_PV_prod[t] == dict_Base_load[t] + model.P_charge[t]
model.PowerBalance = en.Constraint(model.T, rule=power_balance_rule)    

#%% Solve the model
solver = SolverFactory('gurobi')  
start_time = time.time()
results = solver.solve(model, tee=True)
end_time = time.time()
print(f"Solved in {end_time - start_time:.2f} seconds")

#%% Extract and plot results
P_charge = [en.value(model.P_charge[t]) for t in model.T]
P_discharge = [en.value(model.P_discharge[t]) for t in model.T]
E = [en.value(model.E[t]) for t in model.T]
P_grid = [en.value(model.P_grid[t]) for t in model.T]
total_cost = en.value(model.Objective)
print(f"Total cost of electricity: {total_cost:.2f} NOK")    


#%% Plot battery charge/discharge schedule
plt.figure(figsize=(10, 6))
plt.plot(Hours, P_charge, label='Charging Power (kW)', color='blue')
plt.plot(Hours, P_discharge, label='Discharging Power (kW)', color='orange')
plt.title('Battery Charge/Discharge Schedule')
plt.xlabel('Hours')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid()
plt.show()

#%% Plot the net profile load before battery operation
Net_load_before = [dict_Base_load[t] - dict_PV_prod[t] for t in model.T]
plt.figure(figsize=(10, 6))
plt.plot(Hours, Net_load_before, label='Net Load before Battery Operation', color='red')
plt.title('Net Load Profile before Battery Operation')
plt.xlabel('Hours')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid()
plt.show()

#%% Plot the net profile load after battery operation
Net_load = [dict_Base_load[t] - dict_PV_prod[t] + P_charge[t-1] - P_discharge[t-1] for t in model.T]
plt.figure(figsize=(10, 6))
plt.plot(Hours, Net_load, label='Net Load after Battery Operation', color='purple')
plt.title('Net Load Profile after Battery Operation')
plt.xlabel('Hours')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid()
plt.show()      

# %% Plot price profile
plt.figure(figsize=(10, 6))
plt.plot(Hours, Price, label='Electricity Price', color='green')
plt.title('Electricity Price Profile')
plt.xlabel('Hours')
plt.ylabel('Price (NOK/kWh)')
plt.legend()
plt.grid()
plt.show()

#%% Plot base load and PV production profiles
plt.figure(figsize=(10, 6))
plt.plot(Hours, Base_load, label='Base Load (kW)', color='brown')
plt.plot(Hours, PV_prod, label='PV Production (kW)', color='yellow')
plt.title('Base Load and PV Production Profiles')
plt.xlabel('Hours')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid()
plt.show()

# %% Plot battery energy state
plt.figure(figsize=(10, 6))
plt.plot(Hours, E, label='Battery Energy State (kWh)', color='cyan')
plt.title('Battery Energy State over Time')
plt.xlabel('Hours')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid()
plt.show()
