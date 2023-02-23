import numpy as np

def InitialConds():

    with open("MeshFiles/cells.txt") as f:
        num_cells = len(f.readlines())

    u_field = np.zeros((num_cells, 1))
    v_field = np.zeros((num_cells, 1))
    p_field = np.zeros((num_cells, 1))
    
    WriteFile("InitialConds/u_field_big.txt", u_field)
    WriteFile("InitialConds/v_field_big.txt", v_field)
    WriteFile("InitialConds/p_field_big.txt", p_field)

def WriteFile(filename, data):

    with open(filename, "w") as f:
        for i in data:
            f.write(str(i[0]))
            f.write("\n")

if __name__ == "__main__":

    InitialConds()