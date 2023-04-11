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

    def momentum_boundary_mat(self, A, b, F, veff, BC):

        """
        This function discretises the momentum equation boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
            BC (int): boundary condition
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
    
    def pressure_boundary_mat(self, Ap, bp, F):

        """
        This function discretises the pressure laplacian boundaries to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            Ap (np.array): pressure matrix
            bp (np.array): pressure RHS
            F (np.array): flux array
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()

        for i in range(len(cell_owner_neighbour)):
            if cell_owner_neighbour[i][1] == -1:
                bp[cell_owner_neighbour[i][0]] += F[i]

        return Ap, bp