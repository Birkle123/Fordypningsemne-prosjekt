"""
Centralized Parameters Module

This module imports all parameters from the centralized configuration system.
Use this for backward compatibility or when you need all parameters imported.

For new code, consider importing directly from config:
    from config import config
    # Then use config.T, config.V0, etc.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# Import all parameters from centralized config
from config import (
    config,  # The main config object
    T, T1,   # Time parameters
    V0, Vmax, Qmax, Pmax, E_conv, alpha,  # Physical parameters  
    WV_end, spillage_cost,  # Economic parameters
    pi, scenario_info, scenarios, prob, q_cap  # Derived parameters
)

# For convenience, also expose computed properties
P_full_discharge = config.P_full_discharge
Qmax_from_P = config.Qmax_from_P

# Print configuration summary when imported
if __name__ == "__main__":
    print("=" * 50)
    print("PARAMETER SUMMARY")
    print("=" * 50)
    print(config.summary())
    print(f"\nDerived parameters:")
    print(f"q_cap = {q_cap:.2f} m³/s")
    print(f"P_full_discharge = {P_full_discharge:.2f} MW")
    print(f"Qmax_from_P = {Qmax_from_P:.2f} m³/s")
    print("\nScenario details:")
    for name, inflow in scenario_info.items():
        print(f"  {name}: {inflow} m³/s (prob: {prob[name]:.2f})")