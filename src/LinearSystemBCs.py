import numpy as np

class LinearSystemBCs:

    """
    Class to discretise the Incompressible Navier-Stokes equation and the pressure laplacian to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme, viscosity, alpha_u):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
        self.viscosity = viscosity
        self.alpha_u = alpha_u

    def momentum_boundary_mat(self, A, b, F, veff, vel_comp, BC):

        """
        This function discretises the momentum equation boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

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
                    A[cell, cell] += veff[i] * face_mag / d_mag
                    b[cell] -= FN_cell * BC['inlet'][idx]
                    b[cell] += (veff[i] * face_mag / d_mag) * BC['inlet'][idx]
                elif i in self.mesh.boundaries['outlet']:
                    # need to alter as it would be neumann <- CHECK THESE
                    A[cell, cell] += 1 # CHECK THIS
                    b[cell] -= d_mag * BC['outlet'][idx]
                    b[cell] += (veff[i] * face_mag / d_mag) * BC['outlet'][idx]
                elif i in self.mesh.boundaries['upperWall']:
                    A[cell, cell] += veff[i] * face_mag / d_mag
                    b[cell] -= FN_cell * BC['upperWall'][idx]
                    b[cell] += (veff[i] * face_mag / d_mag) * BC['upperWall'][idx]
                elif i in self.mesh.boundaries['lowerWall']:
                    A[cell, cell] += veff[i] * face_mag / d_mag
                    b[cell] -= FN_cell * BC['lowerWall'][idx]
                    b[cell] += (veff[i] * face_mag / d_mag) * BC['lowerWall'][idx]
                elif i in self.mesh.boundaries['frontAndBack']:
                    A[cell, cell] += veff[i] * face_mag / d_mag
                    b[cell] -= FN_cell * BC['frontAndBack'][idx]
                    b[cell] += (veff[i] * face_mag / d_mag) * BC['frontAndBack'][idx]
        
        return A, b
    
    def pressure_boundary_mat(self, Ap, bp, F, raP, BC):

        """
        This function discretises the pressure laplacian boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            Ap (np.array): pressure matrix
            bp (np.array): pressure RHS
            F (np.array): flux array
            raP (np.array): reciprocal of momentum diagonal coefficients
            BC (dict): boundary conditions
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        Ap = Ap.copy()
        bp = bp.copy()

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
                d_mag = np.linalg.norm(cell_centre - face_centre)

                if i in self.mesh.boundaries['outlet']:
                    Ap[cell, cell] -= (face_mag / d_mag) * raP[i]
                    bp -= (face_mag * BC['outlet'][3] / d_mag) * raP[i]

                bp[cell] += F[i]

        return Ap, bp
