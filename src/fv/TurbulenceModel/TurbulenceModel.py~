import numpy as np
from TurbulenceModelBCs import TurbulenceModelBCs

class TurbulenceModel(TurbulenceModelBCs):

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

    def TurbulentVisc(self, k_arr, e_arr):

        vt = np.zeros((len(self.mesh.cells),))

        for i, (k, e) in enumerate(zip(k_arr, e_arr)):

            vt[i] = self.Cmu * ((k**2)/e)

        return vt.flatten()

    def EffectiveVisc(self, k_arr, e_arr, sigma):

        veff = self.viscosity * np.ones((self.mesh.num_cells(),))# + self.TurbulentVisc(k_arr, e_arr) / sigma

        return veff.flatten()

    def veff_face(self, veff):

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        veff_face = np.zeros((self.mesh.num_faces(),))

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                # going to zero gradient for now <- CHECK THIS
                veff_face[i] = veff[owner]
                continue

            fN_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
            PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[owner])
            fx = fN_mag / PN_mag;
            veff_face[i] = fx * veff[owner] + (1 - fx) * veff[neighbour]

        return veff_face

    def k_mat(self, A, b, F, veff):

        """
        This function discretises the turbulence KE equation to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            A (np.array): k matrix
            b (np.array): k RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

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
            A[cell, cell] += veff[i] * face_mag / d_mag
            A[neighbour, neighbour] += veff[i] * face_mag / d_mag

            # diffusive off-diag contributions
            A[cell, neighbour] -= veff[i] * face_mag / d_mag
            A[neighbour, cell] -= veff[i] * face_mag / d_mag

        return A, b
    
    def e_mat(self, A, b, F, veff):

        """
        This function discretises the epsilon equation to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            A (np.array): e matrix
            b (np.array): e RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

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
            A[cell, cell] += veff[i] * face_mag / d_mag
            A[neighbour, neighbour] += veff[i] * face_mag / d_mag

            # diffusive off-diag contributions
            A[cell, neighbour] -= veff[i] * face_mag / d_mag
            A[neighbour, cell] -= veff[i] * face_mag / d_mag

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

        veffk = self.EffectiveVisc(k, e, self.sigmak)

        veffk_face = self.veff_face(veffk)

        A, b = self.k_mat(A, b, F, veffk_face)

        A, b = self.k_boundary_mat(A, b, F, veffk_face, BC)

        A, b = self.ke_UR(A, b, k)

        return A, b
    
    def e_disc(self, k, e, F, BC):

        N = len(self.mesh.cells)

        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        veffEps = self.EffectiveVisc(k, e, self.sigmaEps)

        veffEps_face = self.veff_face(veffEps)

        A, b = self.e_mat(A, b, F, veffEps_face)

        A, b = self.e_boundary_mat(A, b, F, veffEps_face, BC)

        A, b = self.ke_UR(A, b, e)
        
        return A, b
