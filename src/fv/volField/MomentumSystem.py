import numpy as np
from fv.volField.MomentumSystemBCs import MomentumSystemBCs

class MomentumSystem(MomentumSystemBCs):

    """
    Class to discretise the Incompressible Navier-Stokes equation to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme, viscosity, alpha_u):

        self.mesh = mesh
        self.conv_scheme = conv_scheme
        self.viscosity = viscosity
        self.alpha_u = alpha_u

    def ConvMatMomentum(self, F, veff, vel_comp, BC):

        """
        This function discretises the momentum equation to get the diagonal, off-diagonal and source contributions to the linear system for the convection term.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
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
                A[cell, neighbour] = (1-fxN) * FN_cell
                A[neighbour, cell] = (1-fxO) * FN_neighbour
            else:
                # convective diag contributions
                A[cell, cell] += max(FN_cell, 0)
                A[neighbour, neighbour] += max(FN_neighbour, 0)

                # convective off-diag contributions
                A[cell, neighbour] = min(FN_cell, 0)
                A[neighbour, cell] = min(FN_neighbour, 0)

        # Add boundary contributions
        A, b = self.ConvMatMomentumBCs(A, b, F, veff, vel_comp, BC)

        return A, b

    def DiffMatMomentum(self, F, veff, vel_comp, BC):

        """
        This function discretises the momentum equation to get the diagonal, off-diagonal and source contributions to the linear system for the diffusive term.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
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
            A[cell, neighbour] = veff[i] * face_mag / d_mag
            A[neighbour, cell] = veff[i] * face_mag / d_mag
            
        A, b = self.DiffMatMomentumBCs(A, b, F, veff, vel_comp, BC)
        
        return A, b
    
    def MomentumUR(self, A, b, u):

        """
        This function under-relaxes the momentum system.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum RHS
            u (np.array): cell-centred velocity array
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        A = A.copy()
        b = b.copy()

        for i in range(self.mesh.num_cells()):
            A[i, i] /= self.alpha_u
            b[i] += ((1-self.alpha_u)/self.alpha_u) * u[i] * A[i, i]

        return A, b
    
    def MomentumDisc(self, u, F, veff, vel_comp, BC):

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

        Aconv, bconv = self.ConvMatMomentum(F, veff, vel_comp, BC)
        Adiff, bdiff = self.DiffMatMomentum(F, veff, vel_comp, BC)

        A = Aconv - Adiff
        b = bconv - bdiff

        #for i in range(len(A)):
        #    print("diag: ", A[i,i], " off-diag: ", A[i,:].sum() - A[i,i])

        A, b = self.MomentumUR(A, b, u)

        return A, b
    
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