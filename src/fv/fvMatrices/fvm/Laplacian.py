import numpy as np

class Laplacian():

    """
    Class to discretise the laplacian term of the Incompressible Navier-Stokes equation to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh):

        self.mesh = mesh

    def laplacian(self, gamma):

        """
        This function discretises the laplacian term to get the diagonal, off-diagonal and source contributions to the linear system for the diffusive term.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
            gamma (np.array): effective viscosity array
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                continue

            face_area_vector = face_area_vectors[i]
            face_mag = np.linalg.norm(face_area_vector)

            d_mag = np.linalg.norm(cell_centres[cell] - cell_centres[neighbour])

            # diffusive diag contributions
            A[cell, cell] -= gamma[i] * face_mag / d_mag
            A[neighbour, neighbour] -= gamma[i] * face_mag / d_mag

            # diffusive off-diag contributions
            A[cell, neighbour] += gamma[i] * face_mag / d_mag
            A[neighbour, cell] += gamma[i] * face_mag / d_mag
            
        return A, b
