class WriteFiles():

    """
    Class to write results from simulation
    """

    def __init__(self):

        pass

    def WriteFile(self, filename, field):

        """
        Function to write individual files

        Args:
            filename (string): filename to write to
            field (np.array): field to write to file
        """
    
        with open("Results/"+filename+".txt", "w") as f:
            for i in range(len(field)):
                f.write(str(field[i]) + "\n")

    def WriteResults(self, u_field, v_field, p_field, SIMPLE_residuals):

        """
        Function to write out the velocity fields, pressure field and residuals

        Args:
            u_field (np.array): x velocity field
            v_field (np.array): y velocity field
            p_field (np.array): pressure field
            SIMPLE_residuals (np.array): SIMPLE residuals
        """

        self.WriteFile("u_field", u_field)
        self.WriteFile("v_field", v_field)
        self.WriteFile("p_field", p_field)
        self.WriteFile("residuals", SIMPLE_residuals)