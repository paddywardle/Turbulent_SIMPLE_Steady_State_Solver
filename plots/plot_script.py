import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def convergence(residuals):

    plt.plot(range(len(residuals)), residuals)
    plt.xlabel("Iterations")
    plt.ylabel("Residual")
    plt.title("Gauss-Seidel Convergence Plot")
    plt.show()

def velocity_field_quiver_plot(ux_field, uy_field, ncells, d):

    x, y = np.meshgrid(np.linspace(0, d, ncells), np.linspace(0, d, ncells))

    ux_field = np.reshape(ux_field, (ncells, ncells))

    uy_field = np.reshape(uy_field, (ncells, ncells))
    
    plt.quiver(x, y, ux_field, uy_field)
    plt.show()

def field_plot(data, ncells):

    data = np.reshape(data, (ncells, ncells))

    sns.heatmap(data)
    plt.show()