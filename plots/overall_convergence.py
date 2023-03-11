import matplotlib.pyplot as plt

def ReadFile2(filename):

    data = []

    with open(filename) as f:
        for line in f.readlines():
            data.append((line.strip()))
    
    return data

def SIMPLE_convergence(residuals, label):

    u_conv = []
    v_conv = []
    for residual in residuals:
        residual_it = residual.strip().strip("[]").split(",")
        u_conv.append(float(residual_it[0]))

    plt.plot(range(len(residuals)), u_conv, label=label)

def overall_convergence():

    directory = "100x100 after UR study"

    p_95_2 = ReadFile2(f"Results/{directory}/SIM 1/res_SIMPLE.txt")
    p_95_4 = ReadFile2(f"Results/{directory}/SIM 2/res_SIMPLE.txt")
    p_95_6 = ReadFile2(f"Results/{directory}/SIM 3/res_SIMPLE.txt")
    p_90_4 = ReadFile2(f"Results/{directory}/SIM 4/res_SIMPLE.txt")
    p_98_4 = ReadFile2(f"Results/{directory}/SIM 5/res_SIMPLE.txt")
    p_98_6 = ReadFile2(f"Results/{directory}/SIM 6/res_SIMPLE.txt")

    SIMPLE_convergence(p_95_2, "Re=50, Upwind")
    SIMPLE_convergence(p_95_4, "Re=50, Centered")
    SIMPLE_convergence(p_95_6, "Re=100, Upwind")
    SIMPLE_convergence(p_90_4, "Re=100, Centred")
    SIMPLE_convergence(p_98_4, "Re=200, Upwind")
    SIMPLE_convergence(p_98_6, "Re=200, Centred")
    #SIMPLE_convergence(p_90_6, "alpha_u=0.90, alpha_p=0.60")

    plt.xlabel("SIMPLE Iteration")
    plt.ylabel("Residual")
    plt.title("SIMPLE Convergence")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    overall_convergence()



