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
            uface (np.array): current face velocities
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
                sf = np.linalg.norm(face_area_vector)

                FN = sf * uface[i]
                d_mag = np.linalg.norm(cell_centre - face_centre)

                A[cell, cell] += max(FN, 0)
                A[cell, cell] += -self.viscosity * face_mag / d_mag

                b[cell] += min(FN, 0)
                b[cell] += self.viscosity * face_mag / d_mag
                continue

            neighbour_centre = cell_centres[neighbour]

            d_mag = np.linalg.norm(cell_centre - neighbour_centre)

            sf_cell = np.linalg.norm(face_area_vector)
            sf_neighbour = -np.linalg.norm(face_area_vector)

            FN_cell = sf_cell * uface[i]
            FN_neighbour = sf_neighbour * uface[i]

            # convective diag contributions
            A[cell, cell] += max(FN_cell, 0)
            A[neighbour, neighbour] += max(FN_neighbour, 0)

            # diffusive diag contributions
            A[cell, cell] += -self.viscosity * face_mag / d_mag
            A[neighbour, neighbour] += -self.viscosity * face_mag / d_mag

            # convective off-diag contributions
            A[cell, neighbour] += min(FN_cell, 0)
            A[neighbour, cell] += min(FN_neighbour, 0)

            # diffusive off-diag contributions
            A[cell, neighbour] += self.viscosity * face_mag / d_mag
            A[neighbour, cell] += self.viscosity * face_mag / d_mag

        for i in range(len(A)):
            A[i, i] /= self.alpha_u
            b[i] += ((1-self.alpha_u)/self.alpha_u) * u[i] * A[i, i]

        return A, b
    
    def pressure_laplacian(self, Fpre, Au):

        N = len(self.mesh.cells)
        Ap = np.zeros((N, N))
        bp = np.zeros((N, 1))

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

                d_mag = np.linalg.norm(cell_centre - face_centre)

                Ap[cell, cell] += -(self.viscosity * face_mag / d_mag) * (1 / Au[cell, cell])
                bp[cell] += Fpre[i]
                
                continue

            neighbour_centre = cell_centres[neighbour]

            d_mag = np.linalg.norm(cell_centre - neighbour_centre)

            # diffusive diag contributions
            Ap[cell, cell] += -(self.viscosity * face_mag / d_mag)  * (1 / Au[cell, cell])
            Ap[neighbour, neighbour] += -(self.viscosity * face_mag / d_mag)  * (1 / Au[neighbour, neighbour])

            # diffusive off-diag contributions
            Ap[cell, neighbour] += (self.viscosity * face_mag / d_mag) * (1 / Au[cell, cell])
            Ap[neighbour, cell] += (self.viscosity * face_mag / d_mag) * (1 / Au[neighbour, neighbour])

        return Ap, bp
    
    def gauss_seidel(self, A, b, u, tol=1e-10, maxIts=200):

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

        res_initial = np.sum(b - np.matmul(A, u))
        resRel = res_initial / res_initial

        res_ls = [res_initial]
        resRel_ls = [resRel]

        # Iterate 
        for i in range(maxIts):
            # forward sweep
            for j in range(len(u)):
                u[j] = (b[j] - np.dot(A[j, :j], u[:j]) - np.dot(A[j, j+1:], u[j+1:])) / A[j, j]
            # backward sweep
            for j in reversed(range(len(u))):
                u[j] = (b[j] - np.dot(A[j, :j], u[:j]) - np.dot(A[j, j+1:], u[j+1:])) / A[j, j]
            
            res = np.sum(b - np.matmul(A, u))
            resRel = res / res_initial
            res_ls.append(res)
            resRel_ls.append(resRel)
            if np.linalg.norm(np.matmul(A, u)- b) < tol:
                break

        return u, res_ls, resRel_ls