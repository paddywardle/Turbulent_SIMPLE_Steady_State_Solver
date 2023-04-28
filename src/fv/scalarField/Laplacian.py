import numpy as np
from fv.scalarField.LaplacianBCs import LaplacianBCs

class Laplacian(LaplacianBCs):

    """
    Class to discretise the pressure laplacian to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
    
    def LaplacianMatPressure(self, F, raP, BC):

        """
        This function discretises the pressure laplacian to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            Ap (np.array): pressure matrix
            bp (np.array): pressure RHS
            F (np.array): flux array
            raP (np.array): reciprocal of momentum diagonal coefficients
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        N = len(self.mesh.cells)
        Ap = np.zeros((N, N))
        bp = np.zeros((N, 1)).flatten()

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                continue

            face_area_vector = face_area_vectors[i]
            cell_centre = cell_centres[cell]
            face_mag = np.linalg.norm(face_area_vector)
            FN_cell = F[i]
            FN_neighbour = -F[i]

            neighbour_centre = cell_centres[neighbour]
            d_mag = np.linalg.norm(cell_centre - neighbour_centre)

            # diffusive diag contributions
            Ap[cell, cell] -= (face_mag / d_mag) * raP[i]
            Ap[neighbour, neighbour] -= (face_mag / d_mag) * raP[i]

            # diffusive off-diag contributions
            Ap[cell, neighbour] = (face_mag / d_mag) * raP[i]
            Ap[neighbour, cell] = (face_mag / d_mag) * raP[i]

            bp[cell] += FN_cell
            bp[neighbour] += FN_neighbour

        Ap, bp = self.LaplacianMatPressureBCs(Ap, bp, F, raP, BC)

        return Ap, bp
    
    def PressureDisc(self, F, raP_face, BC):

        """
        This function discretises the pressure laplacian to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            F (np.array): flux array
            raP_face (np.array): reciprocal of momentum diagonal coefficients
            BC (int): boundary conditions
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        Ap, bp = self.LaplacianMatPressure(F, raP_face, BC)

        # set reference point
        #Ap[0,0] *= 1.1

        return Ap, bp
