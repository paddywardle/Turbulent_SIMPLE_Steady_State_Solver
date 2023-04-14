import os
import shutil
import gzip

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

    def CreateFileStructure(self, MeshDir):

        if not os.path.exists(f"Results/SIM {self.SIM_num}"):
            os.mkdir(f"Results/SIM {self.SIM_num}")
            os.mkdir(f"Results/SIM {self.SIM_num}/constant")
            os.mkdir(f"Results/SIM {self.SIM_num}/constant/polyMesh")

        shutil.copy(f"MeshFiles/{MeshDir}/blockMeshDict", f"Results/SIM {self.SIM_num}/constant/polyMesh")
        shutil.copy(f"MeshFiles/{MeshDir}/boundary", f"Results/SIM {self.SIM_num}/constant/polyMesh")
        shutil.copy(f"MeshFiles/{MeshDir}/faces", f"Results/SIM {self.SIM_num}/constant/polyMesh")
        shutil.copy(f"MeshFiles/{MeshDir}/neighbour", f"Results/SIM {self.SIM_num}/constant/polyMesh")
        shutil.copy(f"MeshFiles/{MeshDir}/owner", f"Results/SIM {self.SIM_num}/constant/polyMesh")
        shutil.copy(f"MeshFiles/{MeshDir}/points", f"Results/SIM {self.SIM_num}/constant/polyMesh")

    def WriteVolVectorField(self, u_field, v_field, z_field, filename, it):

        with open(f"Results/SIM {self.SIM_num}/0/"+filename, "r") as f:
            boundaries = f.readlines()

        with gzip.open(f"Results/SIM {self.SIM_num}/{it}/"+filename+".gz", "wb") as f:
            f.write(f"{len(u_field)}\n".encode("utf-8"))
            f.write("(\n".encode("utf-8"))
            for i, (u, v, z) in enumerate(zip(u_field, v_field, z_field)):
                f.write(f"({u} {v} {z})\n".encode("utf-8"))
            f.write(")\n;\n\n".encode("utf-8"))

            for bound in boundaries:
                f.write(bound.encode("utf-8"))

    def WriteVolScalarField(self, field, filename, it):

        if filename != "phi":
            with open(f"Results/SIM {self.SIM_num}/0/"+filename, "r") as f:
                boundaries = f.readlines()

        with gzip.open(f"Results/SIM {self.SIM_num}/{it}/"+filename+".gz", "wb") as f:
            f.write(f"{len(field)}\n".encode("utf-8"))
            f.write("(\n".encode("utf-8"))
            for i, val in enumerate(field):
                f.write(f"{val}\n".encode("utf-8"))
            f.write(")\n;\n\n".encode("utf-8"))

            if filename != "phi":
                for bound in boundaries:
                    f.write(bound.encode("utf-8"))

    def WriteVectorBoundaries(self, BC):

        with open(f"Results/SIM {self.SIM_num}/0/"+"U", "w") as f:
            f.write("boundaryField\n{\n")
            for key in BC[1].keys():
                f.write(f"\t{key}\n\t{{\n")
                if BC[1][key][0] == "fixedValue":
                    f.write(f"\t\ttype\t\t{BC[1][key][0]};\n")
                    f.write(f"\t\tvalue\t\tuniform ({BC[0][key][0]} {BC[0][key][1]} {BC[0][key][2]});\n\t}}\n")
                    continue
                f.write(f"\t\ttype\t\t{BC[1][key][0]};\n\t}}\n")
            f.write("}")

    def WriteScalarBoundaries(self, BC, filename):

        if filename == "p":
            var = 3
        elif filename == "k":
            var = 4
        elif filename == "epsilon":
            var = 5
    
        with open(f"Results/SIM {self.SIM_num}/0/"+filename, "w") as f:
            f.write("boundaryField\n{\n")
            for key in BC[1].keys():
                f.write(f"\t{key}\n\t{{\n")
                if BC[1][key][var] == "fixedValue":
                    f.write(f"\t\ttype\t\t{BC[1][key][var]};\n")
                    f.write(f"\t\tvalue\t\tuniform {BC[0][key][var]};\n\t}}\n")
                    continue
                f.write(f"\t\ttype\t\t{BC[1][key][0]};\n\t}}\n")
            f.write("}")

    def WriteBoundaries(self, BC):
        
        if not os.path.exists(f"Results/SIM {self.SIM_num}/0"):
            os.mkdir(f"Results/SIM {self.SIM_num}/0")

        self.WriteVectorBoundaries(BC)
        self.WriteScalarBoundaries(BC, "p")
        self.WriteScalarBoundaries(BC, "k")
        self.WriteScalarBoundaries(BC, "epsilon")

    def WriteIteration(self, u_field, v_field, z_field, p_field, k_field, e_field, F, it):

        if not os.path.exists(f"Results/SIM {self.SIM_num}"):
            os.mkdir(f"Results/SIM {self.SIM_num}")
        
        if not os.path.exists(f"Results/SIM {self.SIM_num}/{it}"):     
            os.mkdir(f"Results/SIM {self.SIM_num}/{it}")

        self.WriteVolVectorField(u_field, v_field, z_field, "U", it)
        self.WriteVolScalarField(p_field, "p", it)
        self.WriteVolScalarField(k_field, "k", it)
        self.WriteVolScalarField(e_field, "epsilon", it)
        self.WriteVolScalarField(F, "phi", it)

    def WriteResults(self, u_field, v_field, z_field, p_field, k_field, e_field, F, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure, sim_time, resolution, iterations):

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

        self.WriteVolVectorField(u_field, v_field, z_field, "U")
        self.WriteVolScalarField(p_field, "p")
        self.WriteVolScalarField(k_field, "k")
        self.WriteVolScalarField(e_field, "e")
        self.WriteVolScalarField(F, "phi")
        self.WriteFile("res_SIMPLE", res_SIMPLE_ls)
        self.WriteFile("resx_momentum", resx_momentum_ls)
        self.WriteFile("resy_momentum", resy_momentum_ls)
        self.WriteFile("res_pressure", res_pressure)

        with open(f"Results/SIM {self.SIM_num}/"+"SIM_time"+".txt", "w") as f:
            f.write("Simulation Time (seconds): " + str(sim_time)+"\n")
            f.write("Mesh Resolution: "+ resolution + "\n")
            f.write("Iterations: " + str(iterations) )
