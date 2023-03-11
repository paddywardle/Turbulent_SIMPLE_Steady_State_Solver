import numpy as np
import matplotlib.pyplot as plt

def ReadFile(filename):

    data = []

    with open(filename) as f:
        for line in f.readlines():
            data.append(float(line.strip()))
    
    return np.array(data, dtype=float)

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
    plt.savefig(f"Results/SIM {SIM_num}/p_inner_loop_conv.png")
    plt.close()

def velocity_field_plot(ux_field, uy_field, uz_field, SIM_num, ncells, d):
    
    quiver_step = 2
    ux_field = np.pad(np.flip(np.reshape(ux_field, (ncells, ncells)), axis=0), (1,1))
    uy_field = np.pad(np.flip(np.reshape(uy_field, (ncells, ncells)), axis=0), (1,1))
    uz_field = np.pad(np.flip(np.reshape(uz_field, (ncells, ncells)), axis=0), (1,1))
    x, y = np.meshgrid(np.linspace(0, ux_field.shape[0], ux_field.shape[1]), np.linspace(0, ux_field.shape[1], ux_field.shape[0]))
    # setting moving wall
    ux_field[0,:] = 1

    mags = np.linalg.norm(np.dstack((ux_field, uy_field)), axis=2)
    axis_positions = np.linspace(0, ncells-1, 6)
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

    return ax

if __name__ == "__main__":

    SIM_num = 6
    ncells = 40
    d=0.1
    SIM = "under_relax_study"

    u_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/u_field.txt")
    v_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/v_field.txt")
    z_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/z_field.txt")
    p_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/p_field.txt")

    fig, ax = plt.subplots(2, 2, figsize=(18, 18))

    u_field = np.pad(np.flip(np.reshape(u_field, (ncells, ncells)), axis=0), (1,1))
    u_field[0,:] = 1
    im = ax[0,0].imshow(u_field, interpolation="spline16", extent=[0, d, 0, d], cmap="jet")
    clb = fig.colorbar(im)
    clb.ax.set_title("U (m/s)")
    ax[0,0].set_title("U Field")

    v_field = np.flip(np.reshape(v_field, (ncells, ncells)), axis=0)
    im = ax[0,1].imshow(v_field, interpolation="spline16", extent=[0, d, 0, d], cmap="jet")
    clb = fig.colorbar(im)
    clb.ax.set_title("V (m/s)")
    ax[0,1].set_title("V Field")

    p_field = np.flip(np.reshape(p_field, (ncells, ncells)), axis=0) * 1000
    im = ax[1,0].imshow(p_field, interpolation="spline16", extent=[0, d, 0, d], cmap="jet")
    clb = fig.colorbar(im)
    clb.ax.set_title("P (Pa)")
    ax[1,0].set_title("Pressure field")

    u_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/u_field.txt")
    v_field = ReadFile(f"Results/{SIM}/SIM {SIM_num}/v_field.txt")

    u_field = np.pad(np.flip(np.reshape(u_field, (ncells, ncells)), axis=0), (1,1))
    v_field = np.pad(np.flip(np.reshape(v_field, (ncells, ncells)), axis=0), (1,1))
    quiver_step = 2
    x, y = np.meshgrid(np.linspace(0, u_field.shape[0], u_field.shape[1]), np.linspace(0, u_field.shape[1], u_field.shape[0]))
    # setting moving wall
    u_field[0,:] = 1
    mags = np.linalg.norm(np.dstack((u_field, v_field)), axis=2)
    axis_positions = np.linspace(0, len(u_field)-1, 6)
    axis_labels = [round((d/len(axis_positions)), 2) * i for i in range(len(axis_positions))]

    ax[1,1].quiver(x[::quiver_step,::quiver_step], y[::quiver_step,::quiver_step], u_field[::quiver_step,::quiver_step], v_field[::quiver_step,::quiver_step])
    im = ax[1,1].imshow(mags, interpolation="spline16", cmap="jet")
    clb = fig.colorbar(im)
    clb.ax.set_title("m/s")

    ax[1,1].set_xticks(axis_positions)
    ax[1,1].set_xticklabels(axis_labels)
    ax[1,1].set_yticks(axis_positions)
    axis_labels.reverse()
    ax[1,1].set_yticklabels(axis_labels)
    ax[1,1].set_title("Velocity Field")
    plt.tight_layout()

    plt.savefig(f"Results/{SIM}/SIM {SIM_num}/summary_fields.png")
