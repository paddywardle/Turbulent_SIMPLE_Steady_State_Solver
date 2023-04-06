from Mesh import Mesh
from SparseMatrixCR import SparseMatrixCR
import numpy as np
from scipy import sparse

class TurbulenceModel:

    """
    Class to discretise the k-e turbulence model equations to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme, viscosity, alpha_u, Cmu, C1, C2, C3, sigmak, sigmaEps):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
        self.viscosity = viscosity
        self.alpha_u = alpha_u
        self.Cmu = Cmu
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.sigmak = sigmak
        self.sigmaEps = sigmaEps

    def effective_visc(self, k_arr, e_arr, sigma):

        veff = np.zeros((len(self.mesh.cells),))

        for i, (k, e) in enumerate(zip(k_arr, e_arr)):

            vt = self.Cmu * ((k**2)/e)

            veff[i] = self.viscosity + vt / sigma

        return veff.flatten()

    def k_mat(self, A, b, F, veff, BC):

        """
        This function discretises the turbulence KE equation to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            u (np.array): current velocity field of the system
            uface (np.array): current face velocities
            it (int): current iteration counter
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i in range(len(cell_owner_neighbour)):

            cell = cell_owner_neighbour[i][0]
            neighbour = cell_owner_neighbour[i][1]

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
            A[cell, cell] += veff[cell] * face_mag / d_mag
            A[neighbour, neighbour] += veff[cell] * face_mag / d_mag

            # diffusive off-diag contributions
            A[cell, neighbour] -= veff[cell] * face_mag / d_mag
            A[neighbour, cell] -= veff[cell] * face_mag / d_mag

        return A, b
    
    def k_boundary_mat(self, A, b, F, veff, BC):

        """
        This function discretises the turbulence KE equation boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

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

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i in range(len(cell_owner_neighbour)):

            if cell_owner_neighbour[i][1] == -1:
                cell = cell_owner_neighbour[i][0]
                face_area_vector = face_area_vectors[i]
                face_centre = face_centres[i]
                cell_centre = cell_centres[cell]
                face_mag = np.linalg.norm(face_area_vector)

                FN_cell = F[i]
                d_mag = np.linalg.norm(cell_centre - face_centre)
                A[cell, cell] += veff[cell] * face_mag / d_mag
                if i in top_index:
                    b[cell] -= FN_cell * BC
                    b[cell] += (veff[cell] * face_mag / d_mag) * BC
        
        return A, b
    
    def e_mat(self, A, b, F, veff, BC):

        """
        This function discretises the epsilon equation to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            u (np.array): current velocity field of the system
            uface (np.array): current face velocities
            it (int): current iteration counter
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i in range(len(cell_owner_neighbour)):

            cell = cell_owner_neighbour[i][0]
            neighbour = cell_owner_neighbour[i][1]

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
            A[cell, cell] += veff[cell] * face_mag / d_mag
            A[neighbour, neighbour] += veff[cell] * face_mag / d_mag

            # diffusive off-diag contributions
            A[cell, neighbour] -= veff[cell] * face_mag / d_mag
            A[neighbour, cell] -= veff[cell] * face_mag / d_mag

        return A, b
    
    def e_boundary_mat(self, A, b, F, veff, BC):

        """
        This function discretises the epsilon equation boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

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

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i in range(len(cell_owner_neighbour)):

            if cell_owner_neighbour[i][1] == -1:
                cell = cell_owner_neighbour[i][0]
                face_area_vector = face_area_vectors[i]
                face_centre = face_centres[i]
                cell_centre = cell_centres[cell]
                face_mag = np.linalg.norm(face_area_vector)

                FN_cell = F[i]
                d_mag = np.linalg.norm(cell_centre - face_centre)
                A[cell, cell] += veff[cell] * face_mag / d_mag
                if i in top_index:
                    b[cell] -= FN_cell * BC
                    b[cell] += (veff[cell] * face_mag / d_mag) * BC
        
        return A, b
    
    def ke_UR(self, A, b, x):

        for i in range(len(A)):
            A[i, i] /= self.alpha_u
            b[i] += ((1-self.alpha_u)/self.alpha_u) * x[i] * A[i, i]

        return A, b
    
    def k_disc(self, k, e, F, BC):

        N = len(self.mesh.cells)

        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        veffk = self.effective_visc(k, e, self.sigmak)

        A, b = self.k_mat(A, b, F, veffk, BC)

        A, b = self.k_boundary_mat(A, b, F, veffk, BC)

        A, b = self.ke_UR(A, b, k)

        return A, b
    
    def e_disc(self, k, e, F, BC):

        N = len(self.mesh.cells)

        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        veffEps = self.effective_visc(k, e, self.sigmaEps)

        A, b = self.e_mat(A, b, F, veffEps, BC)

        A, b = self.e_boundary_mat(A, b, F, veffEps, BC)

        A, b = self.ke_UR(A, b, e)
        
        return A, b