from Mesh import Mesh
from SparseMatrixCR import SparseMatrixCR
import numpy as np

class LinearSystem:

    def __init__(self, mesh, viscosity, alpha_u):

        self.mesh = mesh
        self.viscosity = viscosity
        self.alpha_u = alpha_u

    def momentum_disc(self, u, uface, dim):

        """
        This function discretises the momentum equation to get the diagonal and off-diagonal contributions to the linear system.

        Args:
            u (np.array): current velocity field of the system
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        N = len(self.mesh.cells)

        A = np.zeros((N, N))
        b = np.zeros((N, 1))

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        neighbours = self.mesh.neighbouring_cells()

        for i in range(len(cell_owner_neighbour)):

            cell = cell_owner_neighbour[i][0]
            neighbour = cell_owner_neighbour[i][1]
            face_area_vector = face_area_vectors[i]
            face_centre = face_centres[i]
            cell_centre = cell_centres[cell]
            face_mag = np.linalg.norm(face_area_vector)

            if neighbour == -1:
                if dim == "x":
                    sf = face_area_vector[0]
                else:
                    sf = face_area_vector[1]

                FN = sf * uface[i]
                d = cell_centre - face_centre
                d_mag = np.linalg.norm(d)

                A[cell, cell] += max(FN, 0)
                A[cell, cell] += -self.viscosity * face_mag / d_mag

                b[cell] += min(FN, 0)
                b[cell] += self.viscosity * face_mag / d_mag
                
                continue

            neighbour_centre = cell_centres[neighbour]

            d_mag = np.linalg.norm(cell_centre - neighbour_centre)

            if dim == "x":
                sf_cell = face_area_vector[0]
                sf_neighbour = -face_area_vector[0]
            else:
                sf_cell = face_area_vector[1]
                sf_neighbour = -face_area_vector[1]

            FN_cell = sf_cell * uface[i]
            FN_neighbour = sf_neighbour * uface[i]

            # convective diag contributions
            A[cell, cell] += max(FN_cell, 0)
            A[neighbour, neighbour] += max(FN_neighbour, 0)

            # diffusive diag contributions
            A[cell, cell] += -self.viscosity * face_mag / d_mag
            A[neighbour, neighbour] += -self.viscosity * face_mag / d_mag

            # convective off-diag contributions
            A[cell, neighbour] = min(FN_cell, 0)
            A[neighbour, cell] = min(FN_neighbour, 0)

            # diffusive off-diag contributions
            A[cell, neighbour] = self.viscosity * face_mag / d_mag
            A[neighbour, cell] = self.viscosity * face_mag / d_mag

        for i in range(len(A)):
            A[i, i] /= self.alpha_u
            b[i] += ((1-self.alpha_u)/self.alpha_u) * u[i] * A[i, i]

        return A, b
    
    def gauss_seidel(self, A, b, u, tol=1e-6, maxIts=200):

        """
        This function uses the Gauss-Seidel algorithm to solve the linear system.

        Args:
            A (np.array): array containing the diagonal and off-diagonal contributions to the linear system.
            b (np.array): array containing the source contributions to the linear system.
            tol (float): tolerance for algorithm convergence.
            maxIts (int): maximum number of iterations that algorithm should run for.
        Returns:
            np.array: solution from Gauss-Seidel algorithm.

        """

        it = 0
        # set initial guess for x and initial residual value
        x_initial = np.reshape(u, b.shape)
        x = x_initial
        x_plus1 = x_initial
        res_initial = np.linalg.norm(b - np.matmul(A, x_initial)) + 1e-6 # Udine and Jasak <- ADD REFERENCE
        res = res_initial / res_initial

        # while number iterations is less than 
        while (it < maxIts) and (res > tol):

            lower_tri = np.tril(A)
            strictly_upper_tri = np.triu(A, 1)
            x_plus1 = np.matmul(np.linalg.inv(lower_tri), (b - np.matmul(strictly_upper_tri, x)))
            x = x_plus1
            res = np.linalg.norm(b - np.matmul(A, x)) / res_initial # Udine and Jasak <- ADD REFERENCE
            it += 1

        return x_plus1.flatten()
        

