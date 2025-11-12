"""
Created on 2024-08-10

@author: ivespe, merkebud

Script for Exercise 1 ("Modelling flexibility resources") in specialization 
course module "Flexibility in power grid operation and planning" at NTNU (TET4565/TET4575) 
"""

# %% Define dependencies and EWH model function 

from math import exp
import matplotlib.pyplot as plt
import numpy as np


def make_load_profile_ewh(time_steps,P,T,S,T_a,C,R,T_min,T_max,t_act,S_act):
    """
    Generate load time series for electric water heater (EWH).
    
    Inputs:
        time_steps: Number of time steps (minutes) (int)
        P: Initial load demand of EWH in kW
        T: Initial temperature of EWH in degrees Celsius
        S: Initial EWH status (1 is heating, 0 is not)
        T_a: Ambient temperature in degrees Celsius
        C: Thermal capacitance in kWh/deg C
        R: Thermal resistance in (degrees Celsius)/kW
        T_min: Minimum allowed water temperature in degrees Celsius
        T_max: Maximum allowed water temperature in degrees Celsius        
        t_act: Time of flexibility activation (minutes)
        S_act: EWH thermostat status after activating flexibility; 1 turns all EWHs on; 
            0 turns all EWHs off
    
    Outputs:
        P_list: Load demand time series of EWH, list with one element per time step
        T_list: Temperature time series of EWH in degrees Celsius, 
            with one element per time step
        S_list: EWH status time series, list with one element per time step    
    """

    P_list = [P]
    T_list = [T]
    S_list = [S]

    for t in range(1,time_steps):        
        T_prev = T
        S_prev = S

        # Solve differential equation for the change of temperature for the next time step
        T = T_a - exp(-(1/60)/(C*R))*(T_a + P_m * R * S_prev  - T_prev) + P_m * R * S_prev 

        if (T <= T_min) & (S_prev == 0):
            # Turn EWH on if the temperature becomes too low
            S = 1
        elif (T > T_max) & (S_prev == 1):
            # Turn EWH off if the temperature becomes too high
            S = 0
        else:
            S = S_prev

        if (t == t_act) & (S_act is not None):
            # Activate flexibility
            S = S_act

        # The EWH operates at full power capacity if turned on
        P = S_prev * P_m

        P_list.append(P)
        T_list.append(T)
        S_list.append(S)

    return P_list, T_list, S_list

def make_flexibility_characteristics_plot_with_rebound(S_list, P_list, T_list, time_steps, t_act=None):
    """
    Create a simplified plot showing EWH flexibility with rebound effect for single water heater.
    Shows 2kW spike during activation, then -83W until end of simulation.
    
    Inputs:
        S_list: EWH status time series (1 = heating, 0 = not heating)
        P_list: Power consumption time series in kW
        T_list: Temperature time series
        time_steps: Number of time steps (minutes)
        t_act: Time of flexibility activation (minutes)
    """
    
    # Create time axis in minutes
    time_axis = np.arange(time_steps)
    
    # Create simplified power profile with rebound
    P_modified = np.zeros(time_steps)
    
    if t_act is not None:
        # Find when the activation period ends (when heater turns off after activation)
        activation_end = None
        for i in range(t_act, len(S_list)):
            if i > t_act and S_list[i] == 0:  # First time heater turns off after activation
                activation_end = i
                break
        
        # If we find the end of activation, apply the rebound
        if activation_end is not None:
            # During activation: 2kW
            for i in range(t_act, activation_end):
                P_modified[i] = 2.0
            
            # After activation until end: -83W = -0.083kW
            for i in range(activation_end, time_steps):
                P_modified[i] = -0.083
        else:
            # Fallback: if activation doesn't end, just show activation period
            for i in range(t_act, time_steps):
                P_modified[i] = 2.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the modified power with rebound effect
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Power Consumption (kW)')
    ax.plot(time_axis, P_modified, color='tab:red', linewidth=3, label='Flexibility Power Profile')
    ax.grid(True, alpha=0.3)
    
    # Mark flexibility activation time
    if t_act is not None:
        ax.axvline(x=t_act, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                  label=f'Flexibility Activation (t={t_act})')
    
    # Add horizontal line at zero for reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Set y-axis limits to show both positive and negative values clearly
    ax.set_ylim(-0.15, 2.2)
    
    ax.legend(loc='upper right')
    plt.title('EWH Flexibility: 2kW Activation + 83W Rebound Effect')
    fig.tight_layout()
    plt.show()
    
    return fig

def make_flexibility_characteristics_plot(S_list, P_list, time_steps, t_act=None):
    """
    Create a plot showing total power activated for EWH flexibility.
    
    Inputs:
        S_list: EWH status time series (1 = heating, 0 = not heating) - not used in plot
        P_list: Power consumption time series in kW
        time_steps: Number of time steps (minutes)
        t_act: Time of flexibility activation (optional, for marking on plot)
    """
    
    # Create time axis in minutes
    time_axis = np.arange(time_steps)
    
    # Create figure with single axis for power
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot power consumption
    color = 'tab:blue'
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Total Power Activated (kW)', color=color)
    ax.plot(time_axis, P_list, color=color, linewidth=2, label='Power Consumption')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    # Mark flexibility activation time if provided
    if t_act is not None:
        ax.axvline(x=t_act, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Flexibility Activation (t={t_act})')
        ax.legend(loc='upper left')
    
    # Add title and layout adjustments
    plt.title('EWH Flexibility Characteristics: Power Consumption')
    fig.tight_layout()
    plt.show()
    
    return fig

# %% Initialize Electric Water Heater model

#Rated power of EWH in kW
P_m = 2

#Ambient temperature in degrees Celsius
T_a = 24

#Minimum allowed water temperature in degrees Celsius
T_min = 70

#Maximum allowed water temperature in degrees Celsius
T_max = 75

#Thermal capacitance in kWh/deg C
C = 0.335

#Thermal resistance in deg C /kW
R = 500

# Initializing EWH temperature (only applicable if modelling a single EWH)
T = 73

# Initialize electric water heater(s) to be turned off
S = 0
P = 0

# Time of flexibility activation (minutes from start time); 
# set to None to disable flexibility activation
t_act = 1000

# EWH activation signal that sets the status of the EWHs after activating flexibility; 
# 1 turns all EWHs on; 0 turns all EWHs off; set to None to disable flexibility activation
S_act = 1

# Number of time steps (minutes)
time_steps = 24*60

# Number of EWHs / hot water tanks to model
N_EWH = 1

if N_EWH == 1:
    # If modelling a single EWH, initialize temperature as specified above
    T_init = [T]
elif N_EWH > 1:
    # If modelling multiple EWHs, initialize with random temperature
    rng = np.random.default_rng(seed=42)
    T_init = rng.uniform(70,75,N_EWH)

# Initialize time series for aggregated load demand of all the EWHs 
P_list_all = np.zeros(time_steps)

# Initialize time series for aggregated baseline load demand of all the EWHs, 
# i.e., the expected load demand without flexibility activation
P_list_base_all = np.zeros(time_steps)

# %% Run Electric Water Heater model

# Loop over all the EWHs
for i_EWH in range(N_EWH):
    T = T_init[i_EWH]
    
    # Trick to initialize EWH status randomly approximately according to steady-state distribution
    if i_EWH > 0:        
        S = S_list[-1]
        P = P_list[-1]
    
    # Run EWH model without flexibility activation to obtain the baseline power consumption pattern
    P_list_base, T_list_base, _ = make_load_profile_ewh(time_steps,P,T,S,T_a,C,R,T_min,T_max,None,None)
    
    # Run EWH model with flexibility activation
    P_list, T_list, S_list = make_load_profile_ewh(time_steps,P,T,S,T_a,C,R,T_min,T_max,t_act,S_act)
    
    # Aggregate load time series
    P_list_all += np.array(P_list)
    P_list_base_all += np.array(P_list_base)

#%% Plot results for from Electric Water Heater model

if N_EWH == 1:
    # If running model for a single Electric Water Heater
    fig,ax1 = plt.subplots()
    if (t_act != None) & (S_act != None):
        h_T_base, = plt.plot(T_list_base, 'r--')
    h_T, = plt.plot(T_list, 'r')
    ax1.set_ylim(ymin = T_min * 0.95)
    ax1.set_ylim(ymax = T_max * 1.05)
    color1 = 'tab:red'
    ax1.set_xlabel('minutes')
    ax1.set_ylabel('Temperature (degrees Celsius)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    if (t_act != None) & (S_act != None):
        ax1.legend([h_T_base,h_T], ['without flex.','with flex.'], loc = 'upper left')

    ax2 = ax1.twinx()  # instantiate a minute Axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Electric water heating power consumption (kW)', color=color2)  # we already handled the x-label with ax1
    ax2.plot(P_list_base, color=color2, linestyle='dashed')
    ax2.plot(P_list, color=color2)    
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

elif N_EWH > 1:
    # If running model for multiple Electric Water Heaters
    fig,ax1 = plt.subplots()
    ax1.set_ylim(ymin = 0)
    ax1.set_ylim(ymax = P_list_base_all.max() * 1.05)
    color1 = 'tab:blue'
    color2 = 'tab:red'
    ax1.set_ylabel('Aggregated electric water heating power consumption (kW)', color=color1)
    if (t_act != None) & (S_act != None):
        h_P_base, = ax1.plot(P_list_base_all, color=color1, linestyle='dashed')
    h_P, = ax1.plot(P_list_all, color=color2)
    if (t_act != None) & (S_act != None):
        ax1.legend([h_P_base,h_P], ['without flex.','with flex.'], loc = 'upper left')
    plt.show()

# %% Create flexibility characteristics plot

# For single EWH: plot individual EWH activation status and power
if N_EWH == 1:
    make_flexibility_characteristics_plot(S_list, P_list, time_steps, t_act)
    # Also show the simplified rebound effect plot
    make_flexibility_characteristics_plot_with_rebound(S_list, P_list, T_list, time_steps, t_act)

# For multiple EWHs: show aggregated flexibility characteristics
elif N_EWH > 1:
    # Calculate aggregated activation status (percentage of EWHs active)
    # Note: For multiple EWHs, we'll use the last EWH's S_list as an example
    # In practice, you might want to calculate the average activation across all EWHs
    make_flexibility_characteristics_plot(S_list, P_list_all, time_steps, t_act)
    # Also show the simplified rebound effect plot for aggregated data
    make_flexibility_characteristics_plot_with_rebound(S_list, P_list_all, T_list, time_steps, t_act)