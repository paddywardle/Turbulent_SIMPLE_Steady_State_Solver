from Mesh import Mesh
from SparseMatrixCR import SparseMatrixCR
import numpy as np

class LinearSystem:

    def __init__(self, mesh, viscosity, alpha_u):

        self.mesh = mesh
        self.viscosity = viscosity
        self.alpha_u = alpha_u

    def momentum_disc(self, u, F, BC):

        """
        This function discretises the momentum equation to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            u (np.array): current velocity field of the system
            uface (np.array): current face velocities
            it (int): current iteration counter
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        top_index = []

        for i in range(len(self.mesh.boundary_patches)):
            if self.mesh.boundary_patches[i][0] == "movingWall":
                top_index = self.mesh.boundary_patches[i+1]

        top_index = [int(i) for i in top_index]

        N = len(self.mesh.cells)

        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i in range(len(cell_owner_neighbour)):

            if face_area_vectors[i][2] != 0:
                continue

            cell = cell_owner_neighbour[i][0]
            neighbour = cell_owner_neighbour[i][1]
            face_area_vector = face_area_vectors[i]
            face_centre = face_centres[i]
            cell_centre = cell_centres[cell]
            face_mag = np.linalg.norm(face_area_vector)

            if neighbour == -1:
                FN_cell = F[i]
                FN_boundary = -F[i]
                d_mag = np.linalg.norm(cell_centre - face_centre)
                A[cell, cell] += self.viscosity * face_mag / d_mag
                if i in top_index:
                    b[cell] += FN_boundary * BC
                    b[cell] += (self.viscosity * face_mag / d_mag) * BC
                continue

            neighbour_centre = cell_centres[neighbour]

            d_mag = np.linalg.norm(cell_centre - neighbour_centre)

            FN_cell = F[i]
            FN_neighbour = -F[i]
            # convective diag contributions
            A[cell, cell] += max(FN_cell, 0)
            A[neighbour, neighbour] += max(FN_neighbour, 0)

            # diffusive diag contributions
            A[cell, cell] += self.viscosity * face_mag / d_mag
            A[neighbour, neighbour] += self.viscosity * face_mag / d_mag

            # convective off-diag contributions
            A[cell, neighbour] += min(FN_cell, 0)
            A[neighbour, cell] += min(FN_neighbour, 0)

            # diffusive off-diag contributions
            A[cell, neighbour] += -self.viscosity * face_mag / d_mag
            A[neighbour, cell] += -self.viscosity * face_mag / d_mag

        for i in range(len(A)):
            A[i, i] /= self.alpha_u
            b[i] += ((1-self.alpha_u)/self.alpha_u) * u[i] * A[i, i]

        return A, b
    
    def pressure_laplacian(self, Fpre, Au):

        """
        This function discretises the pressure laplacian to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            uFpre (np.array): x face flux
            vFpre (np.array): y face flux
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        N = len(self.mesh.cells)
        Ap = np.zeros((N, N))
        bp = np.zeros((N, 1))

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()

        for i in range(len(cell_owner_neighbour)):

            cell = cell_owner_neighbour[i][0]
            neighbour = cell_owner_neighbour[i][1]
            face_area_vector = face_area_vectors[i]
            cell_centre = cell_centres[cell]
            face_mag = np.linalg.norm(face_area_vector)

            if neighbour == -1:

                bp[cell] += Fpre[i]
                continue

            neighbour_centre = cell_centres[neighbour]

            d_mag = np.linalg.norm(cell_centre - neighbour_centre)

            # diffusive diag contributions
            Ap[cell, cell] += (self.viscosity * face_mag / d_mag)  * (1 / Au[cell, cell])
            Ap[neighbour, neighbour] += (self.viscosity * face_mag / d_mag)  * (1 / Au[neighbour, neighbour])

            # diffusive off-diag contributions
            Ap[cell, neighbour] += -(self.viscosity * face_mag / d_mag) * (1 / Au[cell, cell])
            Ap[neighbour, cell] += -(self.viscosity * face_mag / d_mag) * (1 / Au[neighbour, neighbour])

        return Ap, bp
    
    def gauss_seidel(self, A, b, u, tol=1e-6, maxIts=1000):

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

        for k in range(maxIts):
            for i in range(len(A)):
                u_new = b[i]
                for j in range(len(A)):
                    if (j != i):
                        u_new -= A[i][j] * u[j]
                u[i] = u_new / A[i][i]

        return u, 1
    
# if __name__ == "__main__":
#     ls = LinearSystem(1, 1, 1)

#     # x = np.array([0, 0, 0], dtype=float)                        
#     # A = np.array([[4, 1, 2],[3, 5, 1],[1, 1, 3]], dtype=float)
#     # b = np.array([4,7,3], dtype=float)
#     # A = np.array([[4, 1, 2],[3, 5, 1],[1, 1, 3]])
#     # b = np.array([4,7,3])
#     # u = np.array([0, 0, 0])
#     # print(ls.gauss_seidel(A, b, u))
#     # A = np.array([[4, 1, 2],[3, 5, 1],[1, 1, 3]])
#     # b = np.array([4,7,3])
#     # u = np.array([0, 0, 0])
#     # print(np.linalg.solve(A, b))

#     b = np.array([[0.e+00],[0.e+00],[2.e-05],[2.e-05]], dtype=float)
#     A = np.array([[ 6.31578947e-05, -1.00000000e-05, -1.00000000e-05,  0.00000000e+00], [-1.00000000e-05,  6.31578947e-05 , 0.00000000e+00 ,-1.00000000e-05], [-1.00000000e-05 , 0.00000000e+00  ,6.31578947e-05, -1.00000000e-05],[ 0.00000000e+00 ,-1.00000000e-05 ,-1.00000000e-05 , 6.31578947e-05]], dtype=float)
#     print(np.linalg.solve(A, b))