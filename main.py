""" 
This codebase contains the code used to solve the project in TET4565.
The work was divided into four main tasks, all of which are runnable from this main.py file. 
"""

task1 = False
task2 = True
task3 = False
task4 = False


if __name__ == "__main__":
    if task1:
        from tasks.task1.data_overview import plot_scenario_data
        df_task1 = plot_scenario_data()
    
    if task2:
        # Deterministic equivalent 1: EV
        from tasks.task2.ev_scenarios import run_EV_scenarios
        run_EV_scenarios(plot=True, summary=True)
        
        # Deterministic equivalent 2: Solving each scenario individually
        from tasks.task2.solve_each_scenario import run_individual_scenarios
        run_individual_scenarios(plot=True, summary=True)
        
        # Deterministic equivalent 3: Proper stochastic problem with nonanticipativity
        from tasks.task2.stochastic_problem import run_stochastic_problem
        run_stochastic_problem(plot=True, summary=True)
    
    if task3:
        pass
    
    if task4:
        pass