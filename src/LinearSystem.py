from Mesh import Mesh
from SparseMatrixCR import SparseMatrixCR
import numpy as np
from scipy import sparse

class LinearSystem:

    """
    Function to discretise the Incompressible Navier-Stokes equation and the pressure laplacian to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme, viscosity, alpha_u):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
        self.viscosity = viscosity
        self.alpha_u = alpha_u

    def momentum_disc(self, u, F, BC, format="dense"):

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

            cell = cell_owner_neighbour[i][0]
            neighbour = cell_owner_neighbour[i][1]
            face_area_vector = face_area_vectors[i]
            face_centre = face_centres[i]
            cell_centre = cell_centres[cell]
            face_mag = np.linalg.norm(face_area_vector)

            FN_cell = F[i]
            FN_neighbour = -F[i]

            if neighbour == -1:
                d_mag = np.linalg.norm(cell_centre - face_centre)
                A[cell, cell] += self.viscosity * face_mag / d_mag
                if i in top_index:
                    b[cell] -= FN_cell * BC # CHECK THIS <-MAYBE
                    b[cell] += (self.viscosity * face_mag / d_mag) * BC
                continue

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
                A[cell, neighbour] += (1-fxN) * FN_cell
                A[neighbour, cell] += (1-fxO) * FN_neighbour
            else:
                # convective diag contributions
                A[cell, cell] += max(FN_cell, 0)
                A[neighbour, neighbour] += max(FN_neighbour, 0)

                # convective off-diag contributions
                A[cell, neighbour] += min(FN_cell, 0)
                A[neighbour, cell] += min(FN_neighbour, 0)

            # diffusive diag contributions
            A[cell, cell] += self.viscosity * face_mag / d_mag
            A[neighbour, neighbour] += self.viscosity * face_mag / d_mag

            # diffusive off-diag contributions
            A[cell, neighbour] -= self.viscosity * face_mag / d_mag
            A[neighbour, cell] -= self.viscosity * face_mag / d_mag

        for i in range(len(A)):
            A[i, i] /= self.alpha_u
            b[i] += ((1-self.alpha_u)/self.alpha_u) * u[i] * A[i, i]

        if format == "sparse":
            A = sparse.csr_array(A)
            b = sparse.csr_array(b)

        return A, b
    
    def pressure_laplacian(self, F, raP, BC, format="dense"):

        """
        This function discretises the pressure laplacian to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            Fpre (np.array): face flux
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
        face_centres = self.mesh.face_centres()

        for i in range(len(cell_owner_neighbour)):

            # if face_area_vectors[i][2] != 0:
            #     continue

            cell = cell_owner_neighbour[i][0]
            neighbour = cell_owner_neighbour[i][1]
            face_area_vector = face_area_vectors[i]
            cell_centre = cell_centres[cell]
            face_mag = np.linalg.norm(face_area_vector)
            FN_cell = F[i]
            FN_neighbour = -F[i]

            if neighbour == -1:
                bp[cell] += FN_cell
                continue

            neighbour_centre = cell_centres[neighbour]
            d_mag = np.linalg.norm(cell_centre - neighbour_centre)

            # diffusive diag contributions
            Ap[cell, cell] -= (face_mag / d_mag) * raP[cell]
            Ap[neighbour, neighbour] -= (face_mag / d_mag) * raP[neighbour]

            # diffusive off-diag contributions
            Ap[cell, neighbour] += (face_mag / d_mag) * raP[cell]
            Ap[neighbour, cell] += (face_mag / d_mag) * raP[neighbour]

            bp[cell] += FN_cell
            bp[neighbour] += FN_neighbour

        # set reference point
        Ap[0,0] *= 1.1

        if format == "sparse":
            Ap = sparse.csr_array(Ap)
            bp = sparse.csr_array(Ap)

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
            # forward sweep
            for i in range(A.shape[0]):
                u_new = b[i]
                for j in range(A.shape[0]):
                    if (j != i):
                        u_new -= A[i,j] * u[j]
                u[i] = u_new / A[i,i]
            # backward sweep
            for i in reversed(range(A.shape[0])):
                u_new = b[i]
                for j in reversed(range(A.shape[0])):
                    if (j != i):
                        u_new -= A[i,j] * u[j]
                u[i] = u_new / A[i,i]
            res = np.sum(b - np.matmul(A, u))
            if res < tol:
                break

        return u