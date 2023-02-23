import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from src.utils.InitialConds import ReadFile
import os

def ReadFile(filename):

    data = []

    with open(filename) as f:
        for line in f.readlines():
            data.append(float(line.strip()))
    
    return np.array(data, dtype=float)

def convergence(residuals):

    plt.plot(range(len(residuals)), residuals)
    plt.xlabel("Iterations")
    plt.ylabel("Residual")
    plt.title("Gauss-Seidel Convergence Plot")
    plt.show()

def velocity_field_plot(ux_field, uy_field, ncells, d):

    x, y = np.meshgrid(np.linspace(0, ncells, ncells), np.linspace(0, ncells, ncells))

    ux_field = np.flip(np.reshape(ux_field, (ncells, ncells)), axis=0)

    uy_field = np.flip(np.reshape(uy_field, (ncells, ncells)), axis=0)

    mags = np.linalg.norm(np.dstack((ux_field, uy_field)), axis=2)
    axis_positions = np.linspace(0, ncells, 6)

    ax = plt.gca()
    ax.quiver(x, y, ux_field, uy_field)
    ax.imshow(mags, interpolation="spline16")
    #ax.set_xticks(axis_positions)
    #ax.set_xticklabels(axis_labels)
    plt.show()

def pressure_field(p_field, ncells, d):

    p_field = np.flip(np.reshape(p_field, (ncells, ncells)), axis=0)
    fig, ax = plt.subplots()
    im = ax.imshow(p_field, interpolation="spline16", extent=[0, d, 0, d])
    fig.colorbar(im)
    return ax

if __name__ == "__main__":

    u_field = ReadFile("Results/u_field.txt")
    v_field = ReadFile("Results/v_field.txt")
    p_field = ReadFile("Results/p_field.txt")

    velocity_ax = velocity_field_plot(u_field, p_field, 2, 0.1)
    pressure_ax = pressure_field(p_field, 2, 0.1)
    plt.show()



