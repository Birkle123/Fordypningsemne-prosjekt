import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt


# --- Parameters ---
T = 48
T1 = 24

# Reservoir & turbine parameters
V0 = 3.0 #Mm3
Vmax = 4.5 #Mm3
Qmax = 100.0 #m3/s
Pmax = 86.5 #MW
alpha = 3.6/1000.0 #Conversion factor from flow (m³/s) over one hour to million cubic meters (Mm³)
E_conv = 0.657 #Energy [kWh] generated per cubic meter of discharged water.
WV_end = 52600.0

P_full_discharge = E_conv * Qmax * 3600 / 1000
Qmax_from_P = Qmax * (Pmax / P_full_discharge)
# kWh/m3 * m3/s * 3600s/h = kWh (3600 * E_conv * m.q[t] * 1000)
q_cap = min(Qmax, Qmax_from_P)  # Max discharge limited by both physical and power constraints

# Price profile (same for all scenarios)
pi = {t: 50.0 + (t+1) for t in range(1,T+1)}

# --- Scenario definitions with names ---
scenario_info = {
    "Very Dry": 0,
    "Dry": 10,
    "Normal": 20,
    "Wet": 30,
    "Very Wet": 40,
}

scenarios = list(scenario_info.keys())
prob = {s: 1.0 / len(scenarios) for s in scenarios}