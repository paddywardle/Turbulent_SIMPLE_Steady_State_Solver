from Mesh import Mesh
from SparseMatrixCR import SparseMatrixCR
import numpy as np

class LinearSystem:

    def __init__(self, mesh, viscosity, alpha_u):

        self.mesh = mesh
        self.viscosity = viscosity
        self.alpha_u = alpha_u

    def momentum_disc(self, u, uface):

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

        for i in range(len(self.mesh.cells)):

            neighbours = self.mesh.neighbouring_cells()[i]
            face_area_vectors = self.mesh.face_area_vectors()

            cell_faces = self.mesh.cells[i]
            centre_P = self.mesh.cell_centres()[i]
            cell_owner_neighbour = self.mesh.cell_owner_neighbour()

            for face in cell_faces:
                face_owner_neighbour = cell_owner_neighbour[face]
                if face_owner_neighbour[1] == -1:
                    sf = face_area_vectors[face]
                    face_mag = np.linalg.norm(sf)
                    FN = face_mag * uface[face]
                    
                    A[i, i] += max(FN, 0)
                    A[i, i] += -self.viscosity * face_mag / 0.005

            for j in neighbours:

                # get faces in neighbour cell
                neighbour_faces = self.mesh.cells[j]
                # get the shared faces between the two cells
                shared_face = list(set(cell_faces).intersection(neighbour_faces))[0]
                # get the owner of the face
                owner_neighbour = cell_owner_neighbour[shared_face]
                # get centre of the neighbour cell
                centre_N = self.mesh.cell_centres()[j]

                # if cell is the owner of the face
                if owner_neighbour[0] == i:
                    sf = face_area_vectors[shared_face]
                else:
                    sf = -face_area_vectors[shared_face]

                face_mag = np.linalg.norm(sf)

                # calculate face flux
                FN = face_mag * uface[shared_face]

                d = abs(centre_P - centre_N)
                d_mag = np.linalg.norm(d)

                # convection contributions
                A[i, i] += max(FN, 0)
                A[i, j] += min(FN, 0)

                # diffusive contributions
                A[i, i] += -self.viscosity * face_mag / d_mag
                A[i, j] += self.viscosity * face_mag / d_mag

        for i in range(len(A)):
            A[i, i] /= self.alpha_u
            b[i] = ((1-self.alpha_u)/self.alpha_u) * u[i] * A[i, i]

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
        

