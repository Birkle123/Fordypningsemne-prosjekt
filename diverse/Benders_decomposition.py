"""
This script was created for the intended use in TET4565 at NTNU, in September 2023

The original author of this script is Alexandra Sheppard
The first iteration was meant to showcase a two-stage optimization problem

As of 20.09.2023, the code has been changed, by Kasper E. Thorvaldsen
This iteration showcases how to perform Benders decomposition on the same two-stage optimization problem


The code is built up of the following:
    InputData: Reads in excel file data and stores it in appropriate structure. Please take note of the structure regarding input data and dealing with indices and DataFrames/dictionaries

"""

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Constants
# constants = {'max_acres':500,
#              'max_sugar_sale':6000,
#              'scenarios':['H_high', 'H_avg', 'H_low'],
#              'probs': {'H_high':0, 'H_avg':1, 'H_low':0}
#              }

def InputData(data_file):
    
    """
    This file reads in specific excel-file and returns data formatted for further use
    """
    
    inputdata = pd.read_excel(f'{data_file}')
    inputdata = inputdata.set_index('Parameter', drop=True)
    inputdata = inputdata.transpose()
    data = {}
    data['sell'] = inputdata[['Price_sell', 'Plant_cost']]
    data['buy'] = inputdata[['Price_buy', 'Min_req']].drop('Sugar')
    data['H_yield'] = inputdata[['H_yield']].to_dict()["H_yield"]
    return data



# Set up model 1st stage
def ModelSetUp_1st(data, constants, Cuts):
    # Instance
    m = pyo.ConcreteModel()
    # Define sets
    m.I = pyo.Set(initialize=list(data['sell'].index))
    #Parameters
    m.C = pyo.Param(m.I, initialize=data['sell']['Plant_cost'])
    m.ML = pyo.Param(initialize=constants['max_acres'])
    #Variables
    m.x = pyo.Var(m.I, within=pyo.NonNegativeReals)
    
    #m.C.display()
    """Cuts_information"""
    #Set for cuts
    m.Cut = pyo.Set(initialize = Cuts["Set"])
    #Parameter for cuts
    m.Phi = pyo.Param(m.Cut, initialize = Cuts["Phi"])
    m.Lambda = pyo.Param(m.Cut, m.I, initialize = Cuts["lambda"])
    m.x_hat = pyo.Param(m.Cut, m.I, initialize = Cuts["x_hat"])
    #Variable for alpha
    m.alpha = pyo.Var(bounds = (-1000000,1000000))
    
    
    """
    The following functions define the objective function and constraints of the first stage of the optimization problem
    """
    # Mathematical formulation 1st stage
    def Obj_1st(m):
        """
        The objective function is to minimize cost of planting seeds, while accounting for future potential of income through linear cuts
        """
        return -sum(m.C[i]*m.x[i] for i in m.I) + m.alpha

    def LandRestriction(m):
        """
        The planted crop cannot exceed the land area available
        """
        return sum(m.x[i] for i in m.I) <= m.ML

    def CreateCuts(m,c):
        
        """
        This function defines the constraint for linear cuts for all indices c in C
        If C is empty, then this will not initiate at all
        
        Pro tip: You are allowed to do many processing steps within a function when defining the constraint.
            So if you want to see that the data is processed correctly, you can print out many parameter data here to see that they are correct
            For instance: print(m.Phi[c], m.Lambda[c,"Wheat"],m.x_hat[c,"Wheat"])

        """
        print("Cut",c,":", m.Phi[c], m.Lambda[c,"Wheat"],m.x_hat[c,"Wheat"])
        return(m.alpha <= m.Phi[c] + sum(m.Lambda[c,i]*(m.x[i]-m.x_hat[c,i]) for i in m.I))


    
    
    """Constraint cut"""
    m.CreateCuts = pyo.Constraint(m.Cut,rule = CreateCuts)
    
    
    """Constraints"""
    m.LandRestriction = pyo.Constraint(rule=LandRestriction)
    
    # Define objective function
    m.obj = pyo.Objective(rule=Obj_1st, sense=pyo.maximize)
    
    #m.display()
    
    return m

# Set up model 2nd stage
def ModelSetUp_2nd(data, constants,X_hat):
    # Instance
    m = pyo.ConcreteModel()
    # Define sets
    m.I = pyo.Set(initialize=list(data['sell'].index))
    m.J = pyo.Set(initialize=list(data['buy'].index))
    # Define parameters
    m.Ps = pyo.Param(m.I, initialize=data['sell']['Price_sell'])
    m.Pb = pyo.Param(m.J, initialize=data['buy']['Price_buy'])
    m.H = pyo.Param(m.I, initialize=data['H_yield'])
    m.B = pyo.Param(m.J, initialize=data['buy']['Min_req'])
    m.MS = pyo.Param(initialize=constants['max_sugar_sale'])
    m.X_hat = pyo.Param(m.I, initialize = X_hat)
    # Define variables
    m.x = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.w = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.J, within=pyo.NonNegativeReals)
    
    """
    The following functions define the objective function and constraints of the second stage of the optimization problem
    """
    def Obj_2nd(m):
        """
        The objective function is to maximize income from selling yield, minus the cost of purchasing food
        """
        
        return + sum(m.Ps[i]*m.w[i] for i in m.I) - sum(m.Pb[j]*m.y[j] for j in m.J)
    def MinReq(m,j):
        
        """
        Some crops have a minimum requirement as we will have to store it for feed
        """
        return m.H[j]*m.x[j] + m.y[j] - m.w[j] >= m.B[j]
    def MaxSugarSale(m):
        """
        Sugar market is limited
        """
        return m.w['Sugar'] <= m.MS
    def MaxSugarYield(m):
        """
        Sugar has a maximum requirement (diabetes-aware optimization)
        """
        return m.w['Sugar'] <= m.H['Sugar'] * m.x['Sugar']
    def Crop_plant(m,i):
        """
        This constraint is meant to connect the parameter from the first stage decision to the second-stage problem
        This is perfect for finding the dual value of the decision variable, so we can generate cuts
        """
        return m.x[i] == m.X_hat[i]

    # Define constraints
    m.MinReq = pyo.Constraint(m.J,rule=MinReq)
    m.MaxSugarSale = pyo.Constraint(rule=MaxSugarSale)
    m.MaxSugarYield = pyo.Constraint(rule=MaxSugarYield)
    m.Crop_plant = pyo.Constraint(m.I, rule = Crop_plant)
    # Define objective function
    m.obj = pyo.Objective(rule=Obj_2nd, sense=pyo.maximize)
    return m

def Solve(m):
    opt = SolverFactory("glpk")
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(m, load_solutions=True)
    return results, m
def DisplayResults(m):
    return print(m.display(), m.dual.display())

# Function for creating new linear cuts for optimization problem
def Cut_manage(Cuts,m):
    """Add new cut to existing dictionary of cut information"""
    
    #Find cut iteration by checking number of existing cuts
    cut = len(Cuts["Set"])
    #Add new cut to list, since 0-index is a thing this works well
    Cuts["Set"].append(cut)
    
    #Find 2nd stage cost result
    Cuts["Phi"][cut] = pyo.value(m.obj)
    #Find lambda x_hat for each type of grain
    for i in m.I:
        Cuts["lambda"][cut,i] = m.dual[m.Crop_plant[i]]
        Cuts["x_hat"][cut,i] = m.X_hat[i]
    return(Cuts)
    
"""
Start running the overall Benders decomposition problem
"""

constants = {'max_acres':500,
             'max_sugar_sale':6000,
             }

data = InputData('Farmers_2stage.xlsx')

"""
Setup for benders decomposition
We perform this for x iterations
"""
Iterations = 10

#Pre-step: Formulate cut input data
Cuts = {}
Cuts["Set"] = []
Cuts["Phi"] = {}
Cuts["lambda"] = {}
Cuts["x_hat"] = {}

#This is the while-loop in principle, but for this case is only a for-loop
for i in range(Iterations):

    #Solve 1st stage problem
    m_1st = ModelSetUp_1st(data, constants, Cuts)
    Solve(m_1st)
    

    #Process 1st stage result
    X_hat = {"Wheat":m_1st.x["Wheat"], "Corn":m_1st.x["Corn"], "Sugar":m_1st.x["Sugar"]}
    
    #Print results 1st stage
    print("Iteration",i)
    for x in X_hat:
        print(x,X_hat[x].value)
    input("Press the 'any' button to continue (Wink wink)")
    
    #Setup and solve 2nd stage problem
    m_2nd = ModelSetUp_2nd(data, constants, X_hat)
    Solve(m_2nd)


    #Create new cuts for 1st stage problem
    Cuts = Cut_manage(Cuts,m_2nd)
    
    #Print results 2nd stage
    print("Objective function:",pyo.value(m_2nd.obj))
    print("Cut information acquired:")
    for component in Cuts:
        if component == "lambda" or component == "x_hat":
            for j in m_2nd.I:
                print(component,j,Cuts[component][i,j])
        else:
            print(component,Cuts[component])
    input("Press the 'any' button to continue (Wink wink)")
    
    #We perform a convergence check
    print("UB:",pyo.value(m_1st.alpha.value),"- LB:",pyo.value(m_2nd.obj))
    input("Press the 'any' button to continue (Wink wink)")
    

