import json
import numpy as np

class ReadFiles():

    """
    Class to read files needed for the simulation
    """

    def __init__(self):

        pass

    def ReadJSON(self, filename):

        """
        Function to read JSON files

        Args:
            filename (string): name of json file
        """

        with open(filename, 'r') as f:
            return json.load(f)

    def ReadSettings(self, filename):

        """
        Function to read settings files

        Args:
            filename (string): settings file name
        """

        simulation_sets = self.ReadJSON(filename)['SIMULATION']
        MESH_sets = self.ReadJSON(filename)['MESH']
        Re = simulation_sets['Re']
        alpha_u = simulation_sets['alpha_u']
        alpha_p = simulation_sets['alpha_p']
        conv_scheme = simulation_sets["SIMPLE"]['conv_scheme']
        SIMPLE_tol = simulation_sets["SIMPLE"]['tol']
        SIMPLE_its = simulation_sets["SIMPLE"]["its"]
        tol_GS = simulation_sets['Gauss-Seidel']['tol']
        maxIts = simulation_sets['Gauss-Seidel']['maxIts']
        L = float(MESH_sets['x1']) - float(MESH_sets['x0'])
        MeshFile = MESH_sets["MeshFile"]
        Cmu = simulation_sets["Turbulence"]["Cmu"]
        C1 = simulation_sets["Turbulence"]["C1"]
        C2 = simulation_sets["Turbulence"]["C2"]
        C3 = simulation_sets["Turbulence"]["C3"]
        sigmak = simulation_sets["Turbulence"]["sigmak"]
        sigmaEps = simulation_sets["Turbulence"]["sigmaEps"]

        return Re, alpha_u, alpha_p, conv_scheme, SIMPLE_tol, SIMPLE_its, tol_GS, maxIts, L, MeshFile, Cmu, C1, C2, C3, sigmak, sigmaEps

    def ReadMesh(self, points_filename, faces_filename, cells_filename, owners_filename, neighbours_filename, boundary_filename):

        """
        Function to read different mesh files

        Args:
            points_filename (string): filename of the points file
            faces_filename (string): filename of the string file
            cells_filename (string): filename of the cells file
            boundary_filename (string): filename of the boundary patches file
        """

        # call read_file function for each mesh characteristic and return arrays
        points = np.asarray(self.ReadMeshFile(points_filename, boundary_filename))
        faces = np.asarray(self.ReadMeshFile(faces_filename, boundary_filename), dtype=int)
        cells = np.asarray(self.ReadMeshFile(cells_filename, boundary_filename), dtype=int)
        owners = np.asarray(self.ReadMeshFile(owners_filename, boundary_filename), dtype=int)
        neighbours = np.asarray(self.ReadMeshFile(neighbours_filename, boundary_filename), dtype=int)
        boundary = np.asarray(self.ReadMeshFile(boundary_filename, boundary_filename), dtype=object)

        return points, faces, cells, owners, neighbours, boundary

    def ReadMeshFile(self, filename, boundary_filename):

        """
        Function to read mesh single file

        Args:
            filename (string): name of file to be read
        """

        array = []

        # read each file for the mesh, appropriately format and return array of mesh characteristic
        with open("MeshFiles/" + filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                ls = line.strip('()').split()
                if filename != boundary_filename:
                    ls = [float(i) for i in ls]
                array.append(ls)

        return array
    
    def ReadInitialConds(self, filename):

        """
        Function to read single initial field file

        Args:
            filename (string): name of field file
        """

        data = []

        with open(filename) as f:
            for line in f.readlines():
                data.append(float(line.strip()))
        
        return np.array(data, dtype=float)