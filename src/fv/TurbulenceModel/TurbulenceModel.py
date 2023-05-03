import numpy as np
from fv.TurbulenceModel.TurbulenceModelBCs import TurbulenceModelBCs
from fv.scalarField.Grad import Grad
from fv.fvMatrix import fvMatrix

class TurbulenceModel(TurbulenceModelBCs, fvMatrix):

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

    def ConvMatKE(self, F, veff, BC):

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

        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()
        
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

        A, b = self.ConvMatKEBCs(A, b, F, veff, BC)

        return A, b

    def DiffMatKE(self, F, veff, BC):

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

        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                continue

            face_area_vector = face_area_vectors[i]
            face_mag = np.linalg.norm(face_area_vector)
            
            d_mag = np.linalg.norm(cell_centres[cell] - cell_centres[neighbour])

            # diffusive diag contributions
            A[cell, cell] -= veff[i] * face_mag / d_mag
            A[neighbour, neighbour] -= veff[i] * face_mag / d_mag

            # diffusive off-diag contributions
            A[cell, neighbour] += veff[i] * face_mag / d_mag
            A[neighbour, cell] += veff[i] * face_mag / d_mag

        A, b = self.DiffMatKEBCs(A, b, F, veff, BC)

        return A, b
    
    def KEUR(self, A, b, x):

        A = A.copy()
        b = b.copy()

        diagold = np.diag(A).copy()
        
        for i in range(len(A)):
            A[i, i] /= self.alpha_u

        b += (np.diag(A) - diagold) * x

        return A, b
    
    def KDisc(self, k, e, F, veffk, BC):
        
        #veffk = self.EffectiveVisc(k, e, self.sigmak)

        veff_face = self.veff_face(veffk)

        Aconv, bconv = self.ConvMatKE(F, veff_face, BC)
        Adiff, bdiff = self.DiffMatKE(F, veff_face, BC)

        A = Aconv - Adiff
        b = bconv - bdiff

        A, b = self.KEUR(A, b, k)

        return A, b

    def EDisc(self, k, e, F, veffe, BC):
        
        #veffe = self.EffectiveVisc(k, e, self.sigmaEps)

        veff_face = self.veff_face(veffe)

        Aconv, bconv = self.ConvMatKE(F, veff_face, BC)
        Adiff, bdiff = self.DiffMatKE(F, veff_face, BC)

        A = Aconv - Adiff
        b = bconv - bdiff

        A, b = self.KEUR(A, b, e)

        return A, b
