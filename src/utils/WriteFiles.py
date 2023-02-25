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
            for i in range(len(field)):
                f.write(str(field[i]) + "\n")

    def WriteResults(self, u_field, v_field, p_field, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure):

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
        self.WriteFile("p_field", p_field)
        self.WriteFile("res_SIMPLE", res_SIMPLE_ls)
        self.WriteFile("resx_momentum", resx_momentum_ls)
        self.WriteFile("resy_momentum", resy_momentum_ls)
        self.WriteFile("res_pressure", res_pressure)