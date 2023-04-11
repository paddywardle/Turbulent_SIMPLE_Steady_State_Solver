import numpy as np
from LinearSystemBCs import LinearSystemBCs

class LinearSystem(LinearSystemBCs):

    """
    Class to discretise the Incompressible Navier-Stokes equation and the pressure laplacian to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme, viscosity, alpha_u):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
        self.viscosity = viscosity
        self.alpha_u = alpha_u

    def momentum_mat(self, A, b, F, veff):

        """
        This function discretises the momentum equation to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
            F (np.array): flux array
            veff (np.array): effective viscosity array
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
    
    def momentum_UR(self, A, b, u):

        """
        This function under-relaxes the momentum system.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
            u (np.array): cell-centred velocity array
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        for i in range(len(A)):
            A[i, i] /= self.alpha_u
            b[i] += ((1-self.alpha_u)/self.alpha_u) * u[i] * A[i, i]

        return A, b
    
    def momentum_disc(self, u, F, veff, BC):

        """
        This function discretises the momentum equation to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            u (np.array): cell-centred velocity array
            F (np.array): flux array
            veff (np.array): effective viscosity array
            BC (int): boundary condition
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        N = len(self.mesh.cells)

        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        A, b = self.momentum_mat(A, b, F, veff)

        A, b = self.momentum_boundary_mat(A, b, F, veff, BC)

        A, b = self.momentum_UR(A, b, u) 

        return A, b
    
    def pressure_mat(self, Ap, bp, F, raP):

        """
        This function discretises the pressure laplacian to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            Ap (np.array): pressure matrix
            bp (np.array): pressure RHS
            F (np.array): flux array
            raP (np.array): reciprocal of momentum diagonal coefficients
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()

        for i in range(len(cell_owner_neighbour)):

            cell = cell_owner_neighbour[i][0]
            neighbour = cell_owner_neighbour[i][1]

            if neighbour == -1:
                continue

            face_area_vector = face_area_vectors[i]
            cell_centre = cell_centres[cell]
            face_mag = np.linalg.norm(face_area_vector)
            FN_cell = F[i]
            FN_neighbour = -F[i]

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

        return Ap, bp
    
    def pressure_disc(self, F, raP, BC):

        """
        This function discretises the pressure laplacia nto get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            F (np.array): flux array
            raP (np.array): reciprocal of momentum diagonal coefficients
            BC (int): boundary conditions
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        N = len(self.mesh.cells)

        Ap = np.zeros((N, N))
        bp = np.zeros((N, 1)).flatten()

        Ap, bp = self.pressure_mat(Ap, bp, F, raP)

        Ap, bp = self.pressure_boundary_mat(Ap, bp, F)    

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