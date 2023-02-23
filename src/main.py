from Mesh import Mesh
import numpy as np
from utils.ReadFiles import ReadFiles
from utils.WriteFiles import WriteFiles
from SIMPLE import SIMPLE
import time

if __name__ == "__main__":

    # read and write classes
    read = ReadFiles()
    write = WriteFiles()

    # Read settings
    Re, alpha_u, alpha_p, SIMPLE_tol, SIMPLE_its, GS_tol, maxIts, L = read.ReadSettings('config/config.json')

    # calculate kinematic viscosity
    viscosity = L/Re

    # read in mesh and initialise mesh class using data
    points, faces, cells, boundary = read.ReadMesh("points_test.txt", "faces_test.txt", "cells_test.txt", "boundary_patches.txt")

    mesh = Mesh(points, faces, cells, boundary)

    # set initial conditions for the simulation (Ux, Uy, and P) <- assuming fluid is at rest at the start of the simulation
    u_field = read.ReadInitialConds("InitialConds/u_field.txt")
    v_field = read.ReadInitialConds("InitialConds/v_field.txt")
    p_field = read.ReadInitialConds("InitialConds/p_field.txt")

    # timing the simulation
    start_time = time.perf_counter()

    simple = SIMPLE(mesh, viscosity, alpha_u, alpha_p)

    u, v, p, SIMPLE_res = simple.iterate(u_field, v_field, p_field, SIMPLE_tol, SIMPLE_its)

    simulation_run_time = round(time.perf_counter() - start_time, 2)

    print("Simulation run time: " + str(simulation_run_time))

    write.WriteResults(u, v, p, SIMPLE_res)
