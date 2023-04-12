from Mesh import Mesh
from SIMPLE import SIMPLE
from utils.ReadFiles import ReadFiles
from utils.WriteFiles import WriteFiles
from utils.MeshParser import MeshParser

import numpy as np
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
    Re, alpha_u, alpha_p, conv_scheme, SIMPLE_tol, SIMPLE_its, GS_tol, maxIts, L, directory, Cmu, C1, C2, C3, sigmak, sigmaEps = read.ReadSettings('config/config.json')

    # calculate kinematic viscosity
    viscosity = L/Re

    dim = 10

    mesh = Mesh(f"MeshFiles/{directory}")

    # set initial conditions for the simulation (Ux, Uy, and P) <- assuming fluid is at rest at the start of the simulation
    u_field = read.ReadInitialConds("InitialConds/"+directory+"/u_field.txt")
    v_field = read.ReadInitialConds("InitialConds/"+directory+"/v_field.txt")
    p_field = read.ReadInitialConds("InitialConds/"+directory+"/p_field.txt")
    k_field = read.ReadInitialConds("InitialConds/"+directory+"/k_field.txt")
    e_field = read.ReadInitialConds("InitialConds/"+directory+"/e_field.txt")

    # timing the simulation
    start_time = time.perf_counter()

    simple = SIMPLE(mesh, conv_scheme, viscosity, alpha_u, alpha_p, Cmu, C1, C2, C3, sigmak, sigmaEps)

    u, v, z, p, k, e, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure, iterations = simple.iterate(u_field, v_field, p_field, k_field, e_field, dim, SIMPLE_tol, SIMPLE_its)

    simulation_run_time = round(time.perf_counter() - start_time, 2)

    print("Simulation run time: " + str(simulation_run_time))

    write.WriteResults(u, v, z, p, k, e, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure, simulation_run_time, directory, iterations)
