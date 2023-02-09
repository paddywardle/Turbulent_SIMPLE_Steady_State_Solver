import numpy as np

def InitialConds():

    with open("MeshFiles/cells_test.txt") as f:
        num_cells = len(f.readlines())

    u_field = np.zeros((num_cells, 1))
    v_field = np.zeros((num_cells, 1))
    p_field = np.zeros((num_cells, 1))
    
    WriteFile("InitialConds/u_field.txt", u_field)
    WriteFile("InitialConds/v_field.txt", v_field)
    WriteFile("InitialConds/p_field.txt", p_field)

def WriteFile(filename, data):

    with open(filename, "w") as f:
        for i in data:
            f.write(str(i[0]))
            f.write("\n")

def ReadFile(filename):

    data = []

    with open(filename) as f:
        for line in f.readlines():
            data.append(float(line.strip()))
    
    return np.array(data)

if __name__ == "__main__":

    InitialConds()