from Mesh import Mesh
import numpy as np
from utils.ReadMesh import read_mesh
from utils.WriteResults import WriteResults
from utils.ReadSettings import ReadSettings
from utils.InitialConds import ReadFile
from SIMPLE import SIMPLE
import time

if __name__ == "__main__":

    # Read settings
    Re, alpha_u, alpha_p, SIMPLE_tol, SIMPLE_its, GS_tol, maxIts, L = simulation_sets = ReadSettings('config/config.json')

    # calculate kinematic viscosity
    viscosity = L/Re

    # read in mesh and initialise mesh class using data
    points, faces, cells, boundary = read_mesh("points.txt", "faces.txt", "cells.txt", "boundary_patches2.txt")

    mesh = Mesh(points, faces, cells, boundary)

    # set initial conditions for the simulation (Ux, Uy, and P) <- assuming fluid is at rest at the start of the simulation
    u_field = ReadFile("InitialConds/u_field_big.txt")
    v_field = ReadFile("InitialConds/v_field_big.txt")
    p_field = ReadFile("InitialConds/p_field_big.txt")

    # timing the simulation
    start_time = time.perf_counter()

    simple = SIMPLE(mesh, viscosity, alpha_u, alpha_p)

    u, v, p, SIMPLE_res = simple.iterate(u_field, v_field, p_field, SIMPLE_tol, SIMPLE_its)

    simulation_run_time = round(time.perf_counter() - start_time, 2)

    print("Simulation run time: " + str(simulation_run_time))

    WriteResults(u, v, p, SIMPLE_res)
