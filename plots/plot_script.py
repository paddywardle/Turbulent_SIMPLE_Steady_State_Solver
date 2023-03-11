import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from src.utils.InitialConds import ReadFile
import os
import argparse

SIM = "folder"

def ReadFile(filename):

    data = []

    with open(filename) as f:
        for line in f.readlines():
            data.append(float(line.strip()))
    
    return np.array(data, dtype=float)

def ReadFile2(filename):

    data = []

    with open(filename) as f:
        for line in f.readlines():
            data.append((line.strip()))
    
    return data

def SIMPLE_convergence(residuals, SIM_num):

    u_conv = []
    v_conv = []
    for residual in residuals:
        residual_it = residual.strip().strip("[]").split(",")
        u_conv.append(float(residual_it[0]))
        v_conv.append(float(residual_it[1]))

    plt.plot(range(len(residuals)), u_conv, label="U Residuals")
    plt.plot(range(len(residuals)), v_conv, label="V Residuals")
    plt.legend()
    plt.xlabel("SIMPLE Iteration")
    plt.ylabel("Residual")
    plt.title("SIMPLE Outer Loop Convergence")
    plt.savefig(f"Results/{SIM}/SIM {SIM_num}/SIMPLE_conv_curve.png")

def momentum_convergence(residuals_x, residuals_y, SIM_num):

    initial_x = []
    final_x = []
    initial_y = []
    final_y = []
    for i in range(len(residuals_x)):
        residual_x_it = residuals_x[i].strip().strip("[]").split(",")
        residual_y_it = residuals_y[i].strip().strip("[]").split(",")
        initial_x.append(float(residual_x_it[0]))
        final_x.append(float(residual_x_it[1]))
        initial_y.append(float(residual_y_it[0]))
        final_y.append(float(residual_y_it[1]))
    
    plt.plot(range(len(initial_x)), initial_x, label="Initial Residuals")
    plt.plot(range(len(final_x)), final_x, label="Final Residuals")
    plt.legend()
    plt.xlabel("SIMPLE Iteration")
    plt.ylabel("Residual")
    plt.title("U Inner Loop Convergence")
    plt.savefig(f"Results/{SIM}/SIM {SIM_num}/u_inner_loop_conv.png")
    plt.close()

    plt.plot(range(len(initial_y)), initial_y, label="Initial Residuals")
    plt.plot(range(len(final_y)), final_y, label="Final Residuals")
    plt.legend()
    plt.xlabel("SIMPLE Iteration")
    plt.ylabel("Residual")
    plt.title("V Inner Loop Convergence")
    plt.savefig(f"Results/{SIM}/SIM {SIM_num}/v_inner_loop_conv.png")
    plt.close()

def pressure_convergence(residuals, SIM_num):

    initial = []
    final = []
    for residual in residuals:
        residual_it = residual.strip().strip("[]").split(",")
        initial.append(float(residual_it[0]))
        final.append(float(residual_it[1]))

    plt.plot(range(len(residuals)), initial, label="Initial Residuals")
    plt.plot(range(len(residuals)), final, label="Final Residuals")
    plt.legend()
    plt.xlabel("SIMPLE Iteration")
    plt.ylabel("Residual")
    plt.title("Pressure Inner Loop Convergence")
    plt.savefig(f"Results/{SIM}/SIM {SIM_num}/p_inner_loop_conv.png")
    plt.close()

def velocity_field_plot(ux_field, uy_field, uz_field, SIM_num, ncells, d):
    
    quiver_step = 2
    print(ncells, len(ux_field))
    ux_field = np.pad(np.flip(np.reshape(ux_field, (ncells, ncells)), axis=0), (1,1))
    uy_field = np.pad(np.flip(np.reshape(uy_field, (ncells, ncells)), axis=0), (1,1))
    uz_field = np.pad(np.flip(np.reshape(uz_field, (ncells, ncells)), axis=0), (1,1))
    x, y = np.meshgrid(np.linspace(0, ux_field.shape[0], ux_field.shape[1]), np.linspace(0, ux_field.shape[1], ux_field.shape[0]))

    # setting moving wall
    ux_field[0,:] = 1

    mags = np.linalg.norm(np.dstack((ux_field, uy_field)), axis=2)
    axis_positions = np.linspace(0, len(ux_field)-1, 6)
    axis_labels = [round((d/len(axis_positions)), 2) * i for i in range(len(axis_positions))]

    fig, ax = plt.subplots()
    ax.quiver(x[::quiver_step,::quiver_step], y[::quiver_step,::quiver_step], ux_field[::quiver_step,::quiver_step], uy_field[::quiver_step,::quiver_step])
    im = ax.imshow(mags, interpolation="spline16", cmap="jet")
    ax.set_xticks(axis_positions)
    ax.set_xticklabels(axis_labels)
    ax.set_yticks(axis_positions)
    axis_labels.reverse()
    ax.set_yticklabels(axis_labels)
    clb = fig.colorbar(im)
    clb.ax.set_title("m/s")
    ax.set_title("Velocity Field")
    plt.savefig(f"Results/{SIM}/SIM {SIM_num}/velocity_field.png")
    return ax

def field(field, SIM_num, ncells, d, filename):

    if filename == "x":
        field = np.pad(np.flip(np.reshape(field, (ncells, ncells)), axis=0), (1,1))
        field[0,:] = 1
        fig, ax = plt.subplots()
        im = ax.imshow(field, interpolation="spline16", extent=[0, d, 0, d], cmap="jet")
        clb = fig.colorbar(im)
        clb.ax.set_title("U (m/s)")
        ax.set_title("U Field")
        plt.savefig(f"Results/{SIM}/SIM {SIM_num}/u_field.png")
    elif filename == "y":
        field = np.flip(np.reshape(field, (ncells, ncells)), axis=0)
        fig, ax = plt.subplots()
        im = ax.imshow(field, interpolation="spline16", extent=[0, d, 0, d], cmap="jet")
        clb = fig.colorbar(im)
        clb.ax.set_title("V (m/s)")
        ax.set_title("V Field")
        plt.savefig(f"Results/{SIM}/SIM {SIM_num}/v_field.png")
    elif filename == "z":
        field = np.pad(np.flip(np.reshape(field, (ncells, ncells)), axis=0), (1,1))
        fig, ax = plt.subplots()
        im = ax.imshow(field, interpolation="spline16", extent=[0, d, 0, d], cmap="jet")
        clb = fig.colorbar(im)
        clb.ax.set_title("Z (m/s)")
        ax.set_title("Z Field")
        plt.savefig(f"Results/{SIM}/SIM {SIM_num}/z_field.png")
    else:
        field = np.flip(np.reshape(field, (ncells, ncells)), axis=0) * 1000
        fig, ax = plt.subplots()
        im = ax.imshow(field, interpolation="spline16", extent=[0, d, 0, d], cmap="jet")
        clb = fig.colorbar(im)
        clb.ax.set_title("P (Pa)")
        ax.set_title("Pressure field")
        plt.savefig(f"Results/{SIM}/SIM {SIM_num}/p_field.png")

    return ax

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--SIM_num",
        type=int
    )

    args = parser.parse_args()

    SIM_num = args.SIM_num
    ncells = 40

    u_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/u_field.txt")
    v_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/v_field.txt")
    z_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/z_field.txt")
    p_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/p_field.txt")
    res_SIMPLE = ReadFile2(f"Results/{SIM}/SIM {SIM_num}/res_SIMPLE.txt")
    resx_momentum = ReadFile2(f"Results/{SIM}/SIM {SIM_num}/resx_momentum.txt")
    resy_momentum = ReadFile2(f"Results/{SIM}/SIM {SIM_num}/resy_momentum.txt")
    res_pressure = ReadFile2(f"Results/{SIM}/SIM {SIM_num}/res_pressure.txt")

    velocity_ax = velocity_field_plot(u_field, v_field, z_field, SIM_num, ncells, 0.1)
    plt.close()
    pressure_ax = field(p_field, SIM_num, ncells, 0.1, "p")
    plt.close()
    u_ax = field(u_field, SIM_num, ncells, 0.1, "x")
    plt.close()
    v_ax = field(v_field, SIM_num, ncells, 0.1, "y")
    plt.close()
    z_ax = field(z_field, SIM_num, ncells, 0.1, "z")
    plt.close()
    SIMPLE_convergence(res_SIMPLE, SIM_num)
    plt.close()
    momentum_convergence(resx_momentum, resy_momentum, SIM_num)
    plt.close()
    pressure_convergence(res_pressure, SIM_num)
