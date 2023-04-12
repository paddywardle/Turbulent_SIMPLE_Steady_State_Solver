import sys
import numpy as np
from ReadFiles import ReadFiles
from MeshParser import MeshParser

def InitialConds():

    read = ReadFiles()
    Re, alpha_u, alpha_p, conv_scheme, SIMPLE_tol, SIMPLE_its, GS_tol, maxIts, L, directory, Cmu, C1, C2, C3, sigmak, sigmaEps = read.ReadSettings('config/config.json')

    mesh = MeshParser(f"MeshFiles/{directory}")
    num_cells = len(mesh.cells)

    u_field = np.zeros((num_cells, 1))
    v_field = np.zeros((num_cells, 1))
    p_field = np.zeros((num_cells, 1))
    k_field = np.ones((num_cells, 1))
    e_field = ((Cmu ** 0.75) * np.power(k_field, 1.5))/0.1
    
    WriteFile(f"InitialConds/{directory}/u_field.txt", u_field)
    WriteFile(f"InitialConds/{directory}/v_field.txt", v_field)
    WriteFile(f"InitialConds/{directory}/p_field.txt", p_field)
    WriteFile(f"InitialConds/{directory}/k_field.txt", k_field)
    WriteFile(f"InitialConds/{directory}/e_field.txt", e_field)

def WriteFile(filename, data):

    with open(filename, "w") as f:
        for i in data:
            f.write(str(i[0]))
            f.write("\n")

if __name__ == "__main__":

    InitialConds()