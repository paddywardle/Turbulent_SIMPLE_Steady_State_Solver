from Mesh import Mesh
from SparseMatrixCR import SparseMatrixCR
import numpy as np

class LinearSystem:

    def __init__(self, mesh, viscosity):

        self.mesh = mesh
        self.viscoity = viscosity

    def A_disc(self, u, dx):

        """
        This function discretises the momentum equation to get the contributions to the linear system

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

        # steady state simulation so will be a vector of zeros
        N = len(self.mesh.cells)

        return np.zeros((N, 1)) 

    def solver(self):

        pass

