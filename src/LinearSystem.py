from Mesh import Mesh
from SparseMatrixCR import SparseMatrixCR
import numpy as np

class LinearSystem:

    def __init__(self, mesh, viscosity):

        self.mesh = mesh
        self.viscoity = viscosity

    def A_disc(self, u, dx):

        """
        This function discretises the momentum equation to get the diagonal and off-diagonal contributions to the linear system.

        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        N = len(self.mesh.cells)

        A = np.zeros((N, N))

        for i in range(len(self.mesh.cells)):

            diag_cont = 0
            neighbours = self.mesh.neighbouring_cells()[i]

            # THIS DOESN'T GET THE CORRECT FACE AREA VECTOR!
            area_mag = np.linalg.norm(self.mesh.face_area_vectors()[i])

            # ADD LOGIC FOR THE FACE OWNER!

            for j in neighbours:

                off_diag = 0

                # second order postulation for face area velocity
                u_face = u[i] + dx * abs(u[i] - u[j])

                # convective contribution
                diag_cont += max(area_mag * u_face, 0)
                off_diag += min(area_mag * u_face, 0)

                # diffusion contributions
                diag_cont += -self.viscoity * area_mag / dx
                off_diag += self.viscoity * area_mag / dx

                A[i, j] = off_diag
            
            A[i, i] = diag_cont

        #A_sparse = SparseMatrixCR(N, N).from_dense(A)

        return A

    def b_disc(self):

        """
        This function discretises the momentum equation to get the source contributions to the linear system.

        Returns:
            np.array: N x 1 matrix defining the source contributions to the linear system.

        """

        # steady state simulation so will be a vector of zeros
        N = len(self.mesh.cells)

        return np.zeros((N, 1)) 

    def gauss_seidel(self, A, b, tol=1e-1, maxIts=1000):

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

        x_initial = np.ones(b.shape)
        x = x_initial
        res = np.linalg.norm(x) / np.linalg.norm(x_initial)
        lower_tri = np.tril(A)
        strictly_upper_tri = np.triu(A, 1)
        
        while (it < maxIts) and (res > tol):

            lower_tri = np.tril(A)
            strictly_upper_tri = np.triu(A, 1)
            x_plus1 = np.matmul(np.linalg.inv(lower_tri), (b - np.matmul(strictly_upper_tri, x)))
            x = x_plus1
            res = np.linalg.norm(x) / np.linalg.norm(x_initial)
            it += 1
        
        print(f"Gauss-Seidel Final Iterations = {it}")

        return x_plus1
        

