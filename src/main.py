from Mesh import Mesh
import numpy as np
import time
from ReadMesh import read_mesh
from LinearSystem import LinearSystem
from SparseMatrixCR import SparseMatrixCR

if __name__ == "__main__":

    # read in mesh and initialise mesh class using data
    points, faces, cells, boundary = read_mesh("points.txt", "faces.txt", "cells.txt", "boundary.txt")
    mesh = Mesh(points, faces, cells, boundary)

    # set initial conditions for the simulation (Ux, Uy, and P) <- assuming fluid is at rest at the start of the simulation
    u_prev = np.zeros((mesh.num_cells(), mesh.num_cells()))
    v_prev = np.zeros((mesh.num_cells(), mesh.num_cells()))
    p_prev = np.zeros((mesh.num_cells(), mesh.num_cells()))

    sys = LinearSystem(mesh, 100)

    A = sys.A_disc(u_prev, 0.05)
    b = sys.b_disc()

    print(A)
    