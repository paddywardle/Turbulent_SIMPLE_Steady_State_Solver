import numpy as np

class LaplacianBCs:

    """
    Class to discretise the pressure laplacian to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
    
    def LaplacianMatPressureBCs(self, Ap, bp, F, raP, BC):

        """
        This function discretises the pressure laplacian boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            Ap (np.array): pressure matrix
            bp (np.array): pressure RHS
            F (np.array): flux array
            raP (np.array): reciprocal of momentum diagonal coefficients
            BC (dict): boundary conditions
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        Ap = Ap.copy()
        bp = bp.copy()

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):
            if neighbour == -1:
                face_area_vector = face_area_vectors[i]
                face_centre = face_centres[i]
                cell_centre = cell_centres[cell]
                face_mag = np.linalg.norm(face_area_vector)
                d_mag = np.linalg.norm(cell_centre - face_centre)
                
                if i in self.mesh.boundaries['inlet']:
                    bp[cell] -= raP[i] * face_mag * BC['inlet'][3]
                elif i in self.mesh.boundaries['outlet']:
                    Ap[cell, cell] -= (face_mag / d_mag) * raP[i]
                    bp[cell] -= (face_mag / d_mag) * raP[i] * BC['outlet'][3]
                elif i in self.mesh.boundaries['upperWall']:
                    bp[cell] -= raP[i] * face_mag * BC['upperWall'][3]
                elif i in self.mesh.boundaries['lowerWall']:
                    bp[cell] -= raP[i] * face_mag * BC['lowerWall'][3]
                elif i in self.mesh.boundaries['frontAndBack']:
                    bp[cell] -= raP[i] * face_mag * BC['frontAndBack'][3]

        return Ap, bp
