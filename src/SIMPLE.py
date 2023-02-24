import numpy as np
from scipy.sparse.linalg import bicg
from LinearSystem import LinearSystem

class SIMPLE(LinearSystem):

    """
    Class to hold all the functionality for the Semi-Implicit Algorithm for Pressure-Linked Equations (SIMPLE)
    """

    def __init__(self, mesh, viscosity, alpha_u, alpha_p):
        
        LinearSystem.__init__(self, mesh, viscosity, alpha_u)
        self.alpha_u = alpha_u
        self.alpha_p = alpha_p
    
    def face_velocity(self, u, BC):

        """
        Function to calculate face velocity

        Args:
            u (np.array): velocity field
            BC (float): boundary condition
        Returns:
            uface (np.array): face velocity field

        """

        uface = np.zeros((self.mesh.num_faces(), 1))

        owner_neighbours = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        top_index = []

        # gets movingWall faces from boundary_patches file
        for i in range(len(self.mesh.boundary_patches)):
            if self.mesh.boundary_patches[i][0] == "movingWall":
                top_index = self.mesh.boundary_patches[i+1]

        top_index = [int(i) for i in top_index]

        # loops through owner neighbour pairs
        # applies boundary condition if neighbour = -1
        # linearly interpolates velocity onto the face otherwise
        for i in range(len(owner_neighbours)):

            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]
            
            if (neighbour == -1):
                if (i in top_index):
                    uface[i] = BC
            else:
                PF_mag = np.linalg.norm(face_centres[i] - cell_centres[cell])
                PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[cell])
                uface[i] = u[cell] + (PF_mag * (u[neighbour]-u[cell])) / PN_mag

        return uface
    
    def face_flux(self, uface, vface, zface):

        """
        Function to calculate face flux

        Args:
            uface (np.array): face x velocity field
            vface (np.array): face y velocity field
            zface (np.array): face z velocity field
        Returns:
            F (np.array): face flux field

        """

        F = []
        face_area_vectors = np.squeeze(self.mesh.face_area_vectors())
        
        # horizontally stack x, y and z face velocity values
        face_velocity = np.squeeze(np.hstack((uface, vface, zface)))

        # loop through and dot product face velocities with face area vectors to get face flux
        for i in range(len(face_velocity)):
            F_current = np.dot(face_area_vectors[i], face_velocity[i])
            F.append(F_current)

        F = np.asarray(F)

        return F
    
    def face_pressure(self, p_field):

        """
        Function to calculate face pressure gradient.

        Args:
            p_field (np.array): pressure field
        Returns:
            delta_p_face (np.array): face pressure gradient

        """

        delta_p_face = np.zeros((self.mesh.num_faces(), 1))
        owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()

        # loops through owner neighbour pairs
        for i in range(len(owner_neighbour)):

            cell = owner_neighbour[i][0]
            neighbour = owner_neighbour[i][1]

            # zero gradient boundary condition
            if neighbour == -1:
                delta_p_face[i] = 0
                continue

            # calculates face pressure gradient
            cell_centre = cell_centres[cell]
            neighbour_centre = cell_centres[neighbour]
            face_mag = np.linalg.norm(face_area_vectors[i])
            d_mag = np.linalg.norm(cell_centre - neighbour_centre)
            delta_p_face[i] = ((p_field[neighbour] - p_field[cell]) / d_mag) * face_mag
        
        return delta_p_face
    
    def face_ap(self, A):

        """
        Function to calculate face value of momentum coefficients

        Args:
            A (np.array): momentum matrix
        Returns:
            ap_face (np.array): face diagonal momentum values

        """

        ap_face = np.zeros((self.mesh.num_faces(), 1))
        owner_neighbour = self.mesh.cell_owner_neighbour()
        face_centres = self.mesh.face_centres()
        cell_centres = self.mesh.cell_centres()

        # loops through owner neighbour pairs and linearly interpolates ap onto the face
        for i in range(len(owner_neighbour)):
            cell = owner_neighbour[i][0]
            neighbour = owner_neighbour[i][1]

            if neighbour == -1:
                ap_face[i] = A[cell, cell]
            else:
                PF_mag = np.linalg.norm(face_centres[i] - cell_centres[cell])
                PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[cell])
                ap_face[i] = A[cell, cell] + (PF_mag * (A[neighbour, neighbour]-A[cell, cell])) / PN_mag
        
        return ap_face
    
    def face_flux_correction(self, F, raP, p_field):

        """
        Function to correct face flux field.

        Args:
            F (np.array): face flux field
            raP (np.array): reciprocal of momentum matrix diagonal
            p_field (np.array): pressure field
        Returns:
            F (np.array): corrected face flux field

        """

        owner_neighbours = self.mesh.cell_owner_neighbour()
        face_centres = self.mesh.face_centres()
        cell_centres = self.mesh.cell_centres()
        face_area_vectors = self.mesh.face_area_vectors()

        # loops through owner neighbour pairs and corrected face fluxes
        for i in range(len(owner_neighbours)):
            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]
            face_area_vector = face_area_vectors[i]
            face_mag = np.linalg.norm(face_area_vector)

            # nothing happens at boundary due to 0 gradient boundary conditions
            if neighbour == -1:
                d_mag = np.linalg.norm(cell_centres[cell] - face_centres[i])
                aPN = (raP[cell] * face_mag) / d_mag
                # zero gradient boundary condition
                F[i] -= aPN * 0
                continue
            
            d_mag = np.linalg.norm(cell_centres[cell] - cell_centres[neighbour])

            # pressure coefficent
            aPN = (raP[cell] * face_mag) / d_mag

            # correct face flux
            F[i] -= aPN * (p_field[neighbour] - p_field[cell])

        return F

    def cell_centre_correction(self, raP, u, v, p_field):

        """
        Function to correct cell centred velocities

        Args:
            raP (np.array): reciprocal of momentum matrix diagonal
            u (np.array): x velocity field
            v (np.array): y velocity field
            p_field (np.array): pressure field
        Returns:
            u (np.array): corrected x velocity field
            v (np.array): corrected y velocity field
        """

        owner_neighbours = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()

        # loops through owner neighbour pairs and corrects fields - skips if boundary as 0 gradient Neumann boundary conditions
        for i in range(len(owner_neighbours)):

            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]

            if neighbour == -1:
                # neumann 0 grad boundary conditions so skip
                continue
            else:
                d_mag = np.linalg.norm(cell_centres[cell] - cell_centres[neighbour])

                u[cell] -= ((p_field[neighbour]-p_field[cell]) * raP[cell] / d_mag)
                v[cell] -= ((p_field[neighbour]-p_field[cell]) * raP[cell] / d_mag)

        return u, v
    
    def raP(self, A):

        """
        Function to calculate reciprocal of momentum diagonal.

        Args:
            A (np.array): momentum matrix
        Returns:
            np.array: array of reciprocals
        """
        
        raP = []

        for i in range(len(A)):

            raP.append(1/A[i, i])

        return np.array(raP) 
    
    def face_flux_check(self, F):

        """
        Function to check total flux for each cell

        Args:
            F (np.array): Face fluxes
        Returns:
            total_flux (np.array): total flux for each cell
        """

        owner_neighbour = self.mesh.cell_owner_neighbour()
        total_flux = np.zeros((self.mesh.num_cells(), 1))

        # loops through owner neighbour pairs and adds fluxes to owners and neighbours - skips neighbour if boundary
        for i in range(len(owner_neighbour)):
            cell = owner_neighbour[i][0]
            neighbour = owner_neighbour[i][1]

            total_flux[cell] += F[i]

            if neighbour == -1:
                continue
            total_flux[neighbour] -= F[i]

        return total_flux
    
    def residual(self, Ax, bx, Ay, by, u, v):

        """
        Function to calculate residual for SIMPLE.

        Args:
            Ax (np.array): x momentum matrix
            bx (np.array): x momentum source
            Ay (np.array): y momentum matrix
            by (np.array): y momentum source
            u (np.array): current x velocity field
            v (np.array): current y velocity field
        Returns:
            SIMPLE_res (float): SIMPLE residual
        """

        SIMPLE_res_x = np.linalg.norm(bx - np.matmul(Ax, u))
        SIMPLE_res_y = np.linalg.norm(by - np.matmul(Ay, v))
        SIMPLE_res = np.linalg.norm([SIMPLE_res_x, SIMPLE_res_y])

        return SIMPLE_res
        
    def SIMPLE_loop(self, u, v, F, p, it, format="dense"):

        """
        Function to simulate singular SIMPLE loop that can be repeatedly called.

        Args:
            u (np.array): x velocity field
            v (np.array): y velocity field
            F (np.array): face flux field
            p (np.array): pressure field
            it (int): iteration number
            format (string): matrix format
        Returns:
            u (np.array): corrected cell-centred x velocity field
            v (np.array): corrected cell-centred y velocity field
            Fcorr (np.array): corrected face flux field
            p_field (np.array): updated pressure field
            SIMPLE_res (float): resiudal of SIMPLE loop
            GS_res_x (float): final residual of x Gauss-seidel loop
            GS_res_y (float): final residual of y Gauss-seidel loop
        """

        # Momentum Predictor
        Ax, bx = self.momentum_disc(u, F, 1, format)
        Ay, by = self.momentum_disc(v, F, 0, format)

        uplus1, GS_res_x = self.gauss_seidel(Ax, bx, u)
        vplus1, GS_res_y = self.gauss_seidel(Ay, by, v)

        SIMPLE_res = self.residual(Ax, bx, Ay, by, uplus1, vplus1)

        if it == 0:
            # Fpre calculation
            uface_plus1 = self.face_velocity(uplus1, 1)
            vface_plus1 = self.face_velocity(vplus1, 0)
            zface = np.zeros_like(vface_plus1)
            Fpre = self.face_flux(uface_plus1, vface_plus1, zface)
        else:
            Fpre = F

        # reciprocal of diagonal coefficients
        raP = self.raP(Ax)

        # Pressure corrector
        Ap, bp = self.pressure_laplacian(Fpre, raP, 0)

        p_field, exitcode = bicg(Ap, bp)

        # Face flux correction
        Fcorr = self.face_flux_correction(Fpre, raP, p_field)

        # total_flux for each cell check - uncomment if needed
        #total_flux = self.face_flux_check(Fcorr)

        # Explicit pressure under-relaxation
        p_field = p + self.alpha_p * (p_field - p)

        # Cell-centred correction
        u, v = self.cell_centre_correction(raP, u, v, p_field)

        return u, v, Fcorr, p_field, SIMPLE_res, GS_res_x, GS_res_y

    def iterate(self, u, v, p, tol=1e-10, maxIts=100):
    
        """
        SIMPLE algorithm loop.

        Args:
            u (np.array): x velocity field
            v (np.array): y velocity field
            p (np.array): pressure field
            tol (float): algorithm tolerance
            maxIts (int): maximum number of iterations
        Returns:
            u (np.array): final cell-centred x velocity field
            v (np.array): final cell-centred y velocity field
            p_field (np.array): final pressure field
            res_SIMPLE_ls (list): list of SIMPLE residuals
        """ 
        # Initial flux to feed in
        uface = self.face_velocity(u, 1)
        vface = self.face_velocity(v, 0)
        zface = np.zeros_like(vface)
        F = self.face_flux(uface, vface, zface)

        # Lists to store residuals
        res_SIMPLE_ls = []
        res_SIMPLE_y_ls = []
        resX_GS_ls = []
        resY_GS_ls = []

        it = 0

        # SIMPLE loop - will break if residual is less than tolerance
        for i in range(maxIts):
            print("Iteration: " + str(it+1))
            u, v, F, p, SIMPLE_res, resX_GS, resY_GS = self.SIMPLE_loop(u, v, F, p, it, "dense")
            it += 1
            res_SIMPLE_ls.append(SIMPLE_res)
            resX_GS_ls.append(resX_GS)
            resY_GS_ls.append(resY_GS)
            if SIMPLE_res < tol:
                print(f"Simulation converged in {it} iterations")
                break
        
        return u, v, p, res_SIMPLE_ls