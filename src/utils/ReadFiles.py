import json
import numpy as np

class ReadFiles():

    """
    Class to read files needed for the simulation
    """

    def __init__(self):

        pass

    def ReadJSON(self, filename):

        """
        Function to read JSON files

        Args:
            filename (string): name of json file
        """

        with open(filename, 'r') as f:
            return json.load(f)

    def ReadSettings(self, filename):

        """
        Function to read settings files

        Args:
            filename (string): settings file name
        """

        simulation_sets = self.ReadJSON(filename)['SIMULATION']
        MESH_sets = self.ReadJSON(filename)['MESH']
        Re = simulation_sets['Re']
        alpha_u = simulation_sets['alpha_u']
        alpha_p = simulation_sets['alpha_p']
        conv_scheme = simulation_sets["SIMPLE"]['conv_scheme']
        SIMPLE_tol = simulation_sets["SIMPLE"]['tol']
        SIMPLE_its = simulation_sets["SIMPLE"]["its"]
        tol_GS = simulation_sets['Gauss-Seidel']['tol']
        maxIts = simulation_sets['Gauss-Seidel']['maxIts']
        L = float(MESH_sets['x1']) - float(MESH_sets['x0'])
        MeshFile = MESH_sets["MeshFile"]
        Cmu = simulation_sets["Turbulence"]["Cmu"]
        C1 = simulation_sets["Turbulence"]["C1"]
        C2 = simulation_sets["Turbulence"]["C2"]
        C3 = simulation_sets["Turbulence"]["C3"]
        sigmak = simulation_sets["Turbulence"]["sigmak"]
        sigmaEps = simulation_sets["Turbulence"]["sigmaEps"]

        return Re, alpha_u, alpha_p, conv_scheme, SIMPLE_tol, SIMPLE_its, tol_GS, maxIts, L, MeshFile, Cmu, C1, C2, C3, sigmak, sigmaEps
    
    def ReadInitialConds(self, filename):

        """
        Function to read single initial field file

        Args:
            filename (string): name of field file
        """

        data = []

        with open(filename) as f:
            for line in f.readlines():
                data.append(float(line.strip()))
        
        return np.array(data, dtype=float)