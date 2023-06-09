import numpy as np

class MomentumSystemBCs:

    """
    Class to discretise the Incompressible Navier-Stokes equation to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme, viscosity, alpha_u):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
        self.viscosity = viscosity
        self.alpha_u = alpha_u

    def ConvMatMomentumBCs(self, A, b, F, vel_comp, BC):

        """
        This function discretises the momentum equation boundaries to get the diagonal, off-diagonal and source contributions to the linear system for the convective term.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
            BC (dict): boundary conditions
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        A = A.copy()
        b = b.copy()

        if vel_comp == "u":
            idx = 0
        elif vel_comp == "v":
            idx = 1
        else:
            idx = 2

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                FN_cell = F[i]
                face_mag = np.linalg.norm(face_area_vectors[i])
                d_mag = np.linalg.norm(cell_centres[cell] - face_centres[i])

                if i in self.mesh.boundaries['inlet']:
                    b[cell] -= FN_cell * BC['inlet'][idx] # convection
                elif i in self.mesh.boundaries['outlet']:
                    A[cell, cell] += FN_cell # convection
                    b[cell] -= FN_cell * d_mag * BC['outlet'][idx] # convection
                elif i in self.mesh.boundaries['upperWall']:
                    b[cell] -= FN_cell * BC['upperWall'][idx]
                elif i in self.mesh.boundaries['lowerWall']:
                    b[cell] -= FN_cell * BC['lowerWall'][idx]
                elif i in self.mesh.boundaries['frontAndBack']:
                    pass
                    #b[cell] -= FN_cell * BC['frontAndBack'][idx]
        
        return A, b

    def DiffMatMomentumBCs(self, A, b, F, veff, vel_comp, BC):

        """
        This function discretises the momentum equation boundaries to get the diagonal, off-diagonal and source contributions to the linear system for the diffusive term.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
            BC (dict): boundary conditions
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        A = A.copy()
        b = b.copy()

        if vel_comp == "u":
            idx = 0
        elif vel_comp == "v":
            idx = 1
        else:
            idx = 2

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
                    A[cell, cell] -= veff[i] * face_mag / d_mag # diffusion
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['inlet'][idx] # diffusion
                elif i in self.mesh.boundaries['outlet']:
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['outlet'][idx] # diffusion
                elif i in self.mesh.boundaries['upperWall']:
                    A[cell, cell] -= veff[i] * face_mag / d_mag
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['upperWall'][idx]
                elif i in self.mesh.boundaries['lowerWall']:
                    A[cell, cell] -= veff[i] * face_mag / d_mag
                    b[cell] -= (veff[i] * face_mag / d_mag) * BC['lowerWall'][idx]
                elif i in self.mesh.boundaries['frontAndBack']:
                    pass
                    #A[cell, cell] -= veff[i] * face_mag / d_mag
                    #b[cell] -= (veff[i] * face_mag / d_mag) * BC['frontAndBack'][idx]
        
        return A, b
