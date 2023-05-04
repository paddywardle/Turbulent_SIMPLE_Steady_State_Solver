import numpy as np

class TurbulenceModelBCs:

    """
    Class to discretise the k-e turbulence model equations to produce a linear system for the boundaries, using a finite volume discretisation approach.
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
    
    def ConvMatKEBCs(self, A, b, F, BC, idx):

        """
        This function discretises the turbulence KE equation boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            A (np.array): k matrix
            b (np.array): k RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
            BC (float): boundary condition value
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        A = A.copy()
        b = b.copy()

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                face_area_vector = face_area_vectors[i]
                face_centre = face_centres[i]
                cell_centre = cell_centres[cell]
                face_mag = np.linalg.norm(face_area_vector)

                FN_cell = F[i]
                d_mag = np.linalg.norm(cell_centre - face_centre)

                if i in self.mesh.boundaries['inlet']:
                    b[cell] -= FN_cell * BC['inlet'][idx]
                elif i in self.mesh.boundaries['outlet']:
                    # need to alter as it would be neumann <- CHECK THESE
                    A[cell, cell] += FN_cell # CHECK THIS
                    b[cell] -= FN_cell * d_mag * BC['outlet'][idx]
                elif i in self.mesh.boundaries['upperWall']:
                    b[cell] -= FN_cell * BC['upperWall'][idx]
                elif i in self.mesh.boundaries['lowerWall']:
                    b[cell] -= FN_cell * BC['lowerWall'][idx]
                elif i in self.mesh.boundaries['frontAndBack']:
                    b[cell] -= FN_cell * BC['frontAndBack'][idx]
        
        return A, b

    def DiffMatKEBCs(self, A, b, F, veff, BC, idx):

        """
        This function discretises the epsilon equation boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            A (np.array): e matrix
            b (np.array): e RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
            BC (float): boundary condition value
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        A = A.copy()
        b = b.copy()
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                face_area_vector = face_area_vectors[i]
                face_centre = face_centres[i]
                cell_centre = cell_centres[cell]
                face_mag = np.linalg.norm(face_area_vector)

                FN_cell = F[i]
                d_mag = np.linalg.norm(cell_centre - face_centre)

                if i in self.mesh.boundaries['inlet']:
                    A[cell, cell] -= veff[i] * face_mag / d_mag
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['inlet'][idx]
                elif i in self.mesh.boundaries['outlet']:
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['outlet'][idx]
                elif i in self.mesh.boundaries['upperWall']:
                    A[cell, cell] -= veff[i] * face_mag / d_mag
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['outlet'][idx]
                elif i in self.mesh.boundaries['lowerWall']:
                    A[cell, cell] -= veff[i] * face_mag / d_mag
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['lowerWall'][idx]
                elif i in self.mesh.boundaries['frontAndBack']:
                    A[cell, cell] -= veff[i] * face_mag / d_mag
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['frontAndBack'][idx]
                    
        return A, b
