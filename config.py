import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
import pyomo.environ as pyo


@dataclass
class Config:
    """Centralized configuration class with validation and computed properties."""
    
    # Load from YAML file
    _config_data: Dict[str, Any] = field(init=False)
    
    def __post_init__(self):
        """Load configuration from YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as f:
            self._config_data = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['time', 'reservoir', 'turbine', 'conversion', 'economics', 'scenarios', 'solver']
        for section in required_sections:
            if section not in self._config_data:
                raise ValueError(f"Missing required configuration section: {section}")
    
    # Time parameters
    @property
    def T(self) -> int:
        """Total time periods."""
        return self._config_data['time']['T']
    
    @property
    def T1(self) -> int:
        """First stage periods."""
        return self._config_data['time']['T1']
    
    # Physical parameters
    @property
    def V0(self) -> float:
        """Initial reservoir volume (Mm³)."""
        return self._config_data['reservoir']['V0']
    
    @property
    def Vmax(self) -> float:
        """Maximum reservoir volume (Mm³)."""
        return self._config_data['reservoir']['Vmax']
    
    @property
    def Qmax(self) -> float:
        """Maximum turbine discharge (m³/s)."""
        return self._config_data['turbine']['Qmax']
    
    @property
    def Pmax(self) -> float:
        """Maximum power output (MW)."""
        return self._config_data['turbine']['Pmax']
    
    @property
    def E_conv(self) -> float:
        """Energy conversion factor (kWh/m³)."""
        return self._config_data['turbine']['E_conv']
    
    @property
    def alpha(self) -> float:
        """Flow to volume conversion factor (Mm³ per (m³/s·hour))."""
        return self._config_data['conversion']['alpha']
    
    # Economic parameters
    @property
    def WV_end(self) -> float:
        """Water value at end period (EUR/Mm³)."""
        return self._config_data['economics']['WV_end']
    
    @property
    def spillage_cost(self) -> float:
        """Cost of spillage (EUR/Mm³)."""
        return self._config_data['economics']['spillage_cost']
    
    @property
    def base_price(self) -> float:
        """Base electricity price (EUR/MWh)."""
        return self._config_data['economics']['base_price']
    
    # Scenario parameters
    @property
    def scenario_names(self) -> List[str]:
        """List of scenario names."""
        return self._config_data['scenarios']['names']
    
    @property
    def scenario_inflows(self) -> List[float]:
        """Inflow values for each scenario (m³/s)."""
        return self._config_data['scenarios']['inflows']
    
    @property
    def scenario_probabilities(self) -> List[float]:
        """Probability of each scenario."""
        return self._config_data['scenarios']['probabilities']
    
    @property
    def certain_inflow(self) -> float:
        """Inflow for first T1 periods (m³/s)."""
        return self._config_data['scenarios']['certain_inflow']
    
    # Solver parameters
    @property
    def solver_name(self) -> str:
        """Solver name."""
        return self._config_data['solver']['name']
    
    @property
    def solver_options(self) -> Dict[str, Any]:
        """Solver options."""
        return self._config_data['solver']['options']
    
    # Computed properties
    @property
    def P_full_discharge(self) -> float:
        """Power at full discharge (MW)."""
        return self.E_conv * self.Qmax * 3600 / 1000
    
    @property
    def Qmax_from_P(self) -> float:
        """Max discharge limited by power constraint (m³/s)."""
        return self.Qmax * (self.Pmax / self.P_full_discharge)
    
    @property
    def q_cap(self) -> float:
        """Effective discharge capacity (m³/s)."""
        return min(self.Qmax, self.Qmax_from_P)
    
    # Price profile
    @property
    def pi(self) -> Dict[int, float]:
        """Price profile over all time periods (NOK/MWh)."""
        return {t: self.base_price + t * (1) for t in range(1, self.T + 1)}
    
    # Scenario dictionaries
    @property
    def scenario_info(self) -> Dict[str, float]:
        """Mapping from scenario names to inflow values."""
        return dict(zip(self.scenario_names, self.scenario_inflows))
    
    @property
    def scenarios(self) -> List[str]:
        """List of scenario names."""
        return self.scenario_names
    
    @property
    def prob(self) -> Dict[str, float]:
        """Mapping from scenario names to probabilities."""
        return dict(zip(self.scenario_names, self.scenario_probabilities))
    
    # Solver factory
    def get_solver(self):
        """Create and configure solver instance."""
        solver = pyo.SolverFactory(self.solver_name)
        for option, value in self.solver_options.items():
            solver.options[option] = value
        return solver
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        try:
            # Check scenario consistency
            assert len(self.scenario_names) == len(self.scenario_inflows)
            assert len(self.scenario_names) == len(self.scenario_probabilities)
            assert abs(sum(self.scenario_probabilities) - 1.0) < 1e-6
            
            # Check physical constraints
            assert self.V0 >= 0 and self.V0 <= self.Vmax
            assert self.Qmax > 0
            assert self.Pmax > 0
            assert self.T1 < self.T
            
            return True
        except AssertionError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def summary(self) -> str:
        """Return configuration summary."""
        return f"""
Hydropower Model Configuration Summary:
=====================================
Time horizon: {self.T} hours ({self.T1} certain + {self.T - self.T1} uncertain)
Reservoir: {self.V0} - {self.Vmax} Mm³
Turbine: max {self.Qmax} m³/s, {self.Pmax} MW
Scenarios: {len(self.scenarios)} scenarios
Price range: {min(self.pi.values()):.1f} - {max(self.pi.values()):.1f} NOK/MWh
Solver: {self.solver_name}
"""


# Global configuration instance
config = Config()

# Validate configuration on import
config.validate()

# For backward compatibility, expose common parameters at module level
T = config.T
T1 = config.T1
V0 = config.V0
Vmax = config.Vmax
Qmax = config.Qmax
Pmax = config.Pmax
E_conv = config.E_conv
alpha = config.alpha
WV_end = config.WV_end
spillage_cost = config.spillage_cost
pi = config.pi
scenario_info = config.scenario_info
scenarios = config.scenarios
prob = config.prob
q_cap = config.q_cap


if __name__ == "__main__":
    # Print configuration summary when run directly
    print(config.summary())