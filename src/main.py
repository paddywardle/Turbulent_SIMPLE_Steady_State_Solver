from Mesh import Mesh
import numpy as np
from utils.ReadFiles import ReadFiles
from utils.WriteFiles import WriteFiles
from SIMPLE import SIMPLE
import time
import argparse

if __name__ == "__main__":

    # parse in simulation arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--SIM_num",
        type=int
    )

    args = parser.parse_args()

    SIM_num = args.SIM_num

    # read and write classes
    read = ReadFiles()
    write = WriteFiles(SIM_num)

    # Read settings
    Re, alpha_u, alpha_p, SIMPLE_tol, SIMPLE_its, GS_tol, maxIts, L = read.ReadSettings('config/config.json')

    # calculate kinematic viscosity
    viscosity = L/Re

    # files directory
    directory = "20x20"

    # read in mesh and initialise mesh class using data
    points, faces, cells, boundary = read.ReadMesh(directory+"/points.txt", directory+"/faces.txt", directory+"/cells.txt", directory+"/boundary_patches.txt")

    mesh = Mesh(points, faces, cells, boundary)

    # set initial conditions for the simulation (Ux, Uy, and P) <- assuming fluid is at rest at the start of the simulation
    u_field = read.ReadInitialConds("InitialConds/"+directory+"/u_field.txt")
    v_field = read.ReadInitialConds("InitialConds/"+directory+"/v_field.txt")
    p_field = read.ReadInitialConds("InitialConds/"+directory+"/p_field.txt")

    # timing the simulation
    start_time = time.perf_counter()

    simple = SIMPLE(mesh, viscosity, alpha_u, alpha_p)

    u, v, z, p, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure = simple.iterate(u_field, v_field, p_field, SIMPLE_tol, SIMPLE_its)

    simulation_run_time = round(time.perf_counter() - start_time, 2)

    print("Simulation run time: " + str(simulation_run_time))

    write.WriteResults(u, v, z, p, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure, simulation_run_time, directory)