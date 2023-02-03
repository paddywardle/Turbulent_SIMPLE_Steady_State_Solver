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

        Args:
            u (np.array): current velocity field of the system
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        N = len(self.mesh.cells)

        A = np.zeros((N, N))

        for i in range(len(self.mesh.cells)):

            diag_cont = 0
            neighbours = self.mesh.neighbouring_cells()[i]
            face_area_vectors = self.mesh.face_area_vectors()

            # ADD LOGIC FOR THE FACE OWNER!
            cell_faces = self.mesh.cells[i]

            for j in neighbours:

                off_diag = 0

                neighbour_faces = self.mesh.cells[j]
                shared_face = list(set(cell_faces).intersection(neighbour_faces))[0]
                owner_neighbour = self.mesh.cell_owner_neighbour()[shared_face]

                if owner_neighbour[0] != i:
                    sf = face_area_vectors[shared_face]
                else:
                    sf = -face_area_vectors[shared_face]

                # second order postulation for face area velocity
                u_face = u[i][i] + sf[0] * abs(u[i][i] - u[i][j])

                # convective contribution
                diag_cont += max(sf[1] * u_face, 0)
                off_diag += min(sf[1] * u_face, 0)

                # diffusion contributions
                diag_cont += -self.viscoity * abs(sf[1]) / dx#sf[1] <- FIX THIS
                off_diag += self.viscoity * abs(sf[1]) / dx#sf[1] <- FIX THIS

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

    def gauss_seidel(self, A, b, tol=1e-6, maxIts=1000):

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
        x_initial = np.ones(b.shape)
        x = x_initial
        res_initial = np.abs(b - np.matmul(A, x_initial)).sum() # Udine and Jasak <- ADD REFERENCE
        res = np.abs(b - np.matmul(A, x)).sum() / res_initial
        
        # while number iterations is less than 
        while (it < maxIts) and (res > tol):

            lower_tri = np.tril(A)
            strictly_upper_tri = np.triu(A, 1)
            x_plus1 = np.matmul(np.linalg.inv(lower_tri), (b - np.matmul(strictly_upper_tri, x)))
            x = x_plus1
            res = np.abs(b - np.matmul(A, x)).sum() / res_initial # Udine and Jasak <- ADD REFERENCE
            it += 1
        
        print(f"Gauss-Seidel Final Iterations = {it}")

        return x_plus1
        

