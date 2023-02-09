from Mesh import Mesh
import numpy as np
from utils.ReadMesh import read_mesh
from LinearSystem import LinearSystem
from utils.ReadSettings import ReadSettings
from utils.InitialConds import ReadFile
from SIMPLE import SIMPLE

if __name__ == "__main__":

    # Read settings
    Re, alpha_u, alpha_p, SIMPLE_tol, GS_tol, maxIts = simulation_sets = ReadSettings('config/config.json')

    # calculate kinematic viscosity
    viscosity = 1/Re

    # read in mesh and initialise mesh class using data
    points, faces, cells, boundary = read_mesh("points_test.txt", "faces_test.txt", "cells_test.txt", "boundary_patches.txt")

    mesh = Mesh(points, faces, cells, boundary)

    # set initial conditions for the simulation (Ux, Uy, and P) <- assuming fluid is at rest at the start of the simulation
    u_field = ReadFile("InitialConds/u_field.txt")
    v_field = ReadFile("InitialConds/v_field.txt")
    p_field = ReadFile("InitialConds/p_field.txt")

    simple = SIMPLE(mesh, viscosity, alpha_u, alpha_p)

    simple.iterate(u_field, v_field, p_field, SIMPLE_tol)