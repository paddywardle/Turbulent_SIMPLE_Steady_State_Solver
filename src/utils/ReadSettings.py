import json

def ReadJSON(filename):

    with open(filename, 'r') as f:
        return json.load(f)

def ReadSettings(filename):

    simulation_sets = ReadJSON(filename)['SIMULATION']
    Re = simulation_sets['Re']
    alpha_u = simulation_sets['alpha_u']
    alpha_p = simulation_sets['alpha_p']
    SIMPLE_tol = simulation_sets['SIMPLE_tol']
    tol_GS = simulation_sets['Gauss-Seidel']['tol']
    maxIts = simulation_sets['Gauss-Seidel']['maxIts']
    
    return Re, alpha_u, alpha_p, SIMPLE_tol, tol_GS, maxIts
