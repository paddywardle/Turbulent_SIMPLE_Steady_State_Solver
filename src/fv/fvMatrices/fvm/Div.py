import numpy as np

class Div():

    """
    Class to discretise the convective term of the Incompressible Navier-Stokes equation to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme):

        self.mesh = mesh
        self.conv_scheme = conv_scheme

    def fvmDiv(self, F):

        """
        This function discretises the convective term to get the diagonal, off-diagonal and source contributions to the linear system for the convection term.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        
        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                continue

            face_area_vector = face_area_vectors[i]
            face_centre = face_centres[i]
            cell_centre = cell_centres[cell]
            face_mag = np.linalg.norm(face_area_vector)

            FN_cell = F[i]
            FN_neighbour = -F[i]

            neighbour_centre = cell_centres[neighbour]
            d_mag = np.linalg.norm(cell_centre - neighbour_centre)

            if self.conv_scheme == "centred":

                fN = np.linalg.norm(neighbour_centre - face_centre)
                fP = np.linalg.norm(cell_centre - face_centre)
                fxO = fP/d_mag
                fxN = fN/d_mag
                
                # convective diag contributions
                A[cell, cell] += fxN * FN_cell
                A[neighbour, neighbour] += fxO * FN_neighbour

                # convective off-diag contributions
                A[cell, neighbour] = (1-fxN) * FN_cell
                A[neighbour, cell] = (1-fxO) * FN_neighbour
            else:
                # convective diag contributions
                A[cell, cell] += max(FN_cell, 0)
                A[neighbour, neighbour] += max(FN_neighbour, 0)

                # convective off-diag contributions
                A[cell, neighbour] += min(FN_cell, 0)
                A[neighbour, cell] += min(FN_neighbour, 0)

        return A, b
