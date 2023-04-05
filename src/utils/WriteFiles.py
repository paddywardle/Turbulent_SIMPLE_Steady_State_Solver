import os

class WriteFiles():

    """
    Class to write results from simulation
    """

    def __init__(self, SIM_num):

        self.SIM_num = SIM_num

    def WriteFile(self, filename, field):

        """
        Function to write individual files

        Args:
            filename (string): filename to write to
            field (np.array): field to write to file
        """
    
        with open(f"Results/SIM {self.SIM_num}/"+filename+".txt", "w") as f:
            for i in field:
                f.write(str(i) + "\n")

    def WriteResults(self, u_field, v_field, z_field, p_field, k_field, e_field, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure, sim_time, mat_coeffs, resolution, iterations):

        """
        Function to write out the velocity fields, pressure field and residuals

        Args:
            u_field (np.array): x velocity field
            v_field (np.array): y velocity field
            p_field (np.array): pressure field
            SIMPLE_residuals (np.array): SIMPLE residuals
        """
        if not os.path.exists(f"Results/SIM {self.SIM_num}"):
            os.mkdir(f"Results/SIM {self.SIM_num}")

        self.WriteFile("u_field", u_field)
        self.WriteFile("v_field", v_field)
        self.WriteFile("z_field", z_field)
        self.WriteFile("p_field", p_field)
        self.WriteFile("k_field", k_field)
        self.WriteFile("e_field", e_field)
        self.WriteFile("res_SIMPLE", res_SIMPLE_ls)
        self.WriteFile("resx_momentum", resx_momentum_ls)
        self.WriteFile("resy_momentum", resy_momentum_ls)
        self.WriteFile("res_pressure", res_pressure)
        self.WriteFile("mat_coeffs", mat_coeffs)

        with open(f"Results/SIM {self.SIM_num}/"+"SIM_time"+".txt", "w") as f:
            f.write("Simulation Time (seconds): " + str(sim_time)+"\n")
            f.write("Mesh Resolution: "+ resolution + "\n")
            f.write("Iterations: " + str(iterations) )
