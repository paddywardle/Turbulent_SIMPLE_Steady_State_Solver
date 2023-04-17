import sys
import numpy as np
from ReadFiles import ReadFiles
from MeshParser import MeshParser

def InitialConds():

    read = ReadFiles()
    Re, alpha_u, alpha_p, conv_scheme, SIMPLE_tol, SIMPLE_its, GS_tol, maxIts, L, directory, Cmu, C1, C2, C3, sigmak, sigmaEps, BC = read.ReadSettings('config/config.json')

    mesh = MeshParser(f"MeshFiles/{directory}")
    num_cells = len(mesh.cells)

    u_field = np.ones((num_cells, 1))
    v_field = np.zeros((num_cells, 1))
    w_field = np.zeros((num_cells, 1))
    U = np.hstack((u_field, v_field, w_field))
    p_field = np.zeros((num_cells, 1))
    k_field = (3/2) * np.square(BC[0]['inlet'][4] * u_field)
    e_field = ((Cmu ** 0.75) * np.power(k_field, 1.5))/0.1
    
    WriteVectorField(f"InitialConds/{directory}/U", U)
    WriteScalarField(f"InitialConds/{directory}/p", p_field)
    WriteScalarField(f"InitialConds/{directory}/k", k_field)
    WriteScalarField(f"InitialConds/{directory}/epsilon", e_field)

def WriteVectorField(filename, data):

    with open(filename, "w") as f:
        f.write(str(len(data))+"\n")
        f.write("(\n")
        for i, (u, v, w) in enumerate(data):
            f.write(f"({u} {v} {w})\n")
        f.write(")\n;\n\n")

def WriteScalarField(filename, data):

    with open(filename, "w") as f:
        for i in data:
            f.write(str(i[0]))
            f.write("\n")

if __name__ == "__main__":

    InitialConds()
