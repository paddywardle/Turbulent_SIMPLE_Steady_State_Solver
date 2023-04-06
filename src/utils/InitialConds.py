import numpy as np
from ReadFiles import ReadFiles


def InitialConds():

    read = ReadFiles()
    Re, alpha_u, alpha_p, conv_scheme, SIMPLE_tol, SIMPLE_its, GS_tol, maxIts, L, directory, Cmu, C1, C2, C3, sigmak, sigmaEps = read.ReadSettings('config/config.json')

    with open("MeshFiles/20x20/cells.txt") as f:
        num_cells = len(f.readlines())

    u_field = np.zeros((num_cells, 1))
    v_field = np.zeros((num_cells, 1))
    p_field = np.zeros((num_cells, 1))
    k_field = np.ones((num_cells, 1))
    e_field = ((Cmu ** 0.75) * np.power(k_field, 1.5))/0.1
    
    WriteFile("InitialConds/20x20/u_field.txt", u_field)
    WriteFile("InitialConds/20x20/v_field.txt", v_field)
    WriteFile("InitialConds/20x20/p_field.txt", p_field)
    WriteFile("InitialConds/20x20/k_field.txt", k_field)
    WriteFile("InitialConds/20x20/e_field.txt", e_field)

def WriteFile(filename, data):

    with open(filename, "w") as f:
        for i in data:
            f.write(str(i[0]))
            f.write("\n")

if __name__ == "__main__":

    InitialConds()