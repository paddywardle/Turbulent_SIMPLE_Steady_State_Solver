import numpy as np
from scipy.sparse.linalg import bicg
from LinearSystem import LinearSystem
from TurbulenceModel import TurbulenceModel
import sys

class SIMPLE(LinearSystem, TurbulenceModel):

    """
    Class to hold all the functionality for the Semi-Implicit Algorithm for Pressure-Linked Equations (SIMPLE)
    """

    def __init__(self, writer, mesh, conv_scheme, viscosity, alpha_u, alpha_p, Cmu, C1, C2, C3, sigmak, sigmaEps):
        
        self.writer = writer
        LinearSystem.__init__(self, mesh, conv_scheme, viscosity, alpha_u)
        TurbulenceModel.__init__(self, mesh, conv_scheme, viscosity, alpha_u, Cmu, C1, C2, C3, sigmak, sigmaEps)
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

        # loops through owner neighbour pairs
        # applies boundary condition if neighbour = -1
        # linearly interpolates velocity onto the face otherwise
        for i in range(len(owner_neighbours)):

            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]
            
            if (neighbour == -1):
                if i in self.mesh.boundaries['inlet']:
                    uface[i] = BC['inlet'][0]
                elif i in self.mesh.boundaries['outlet']:
                    uface[i] = BC['outlet'][0]
                elif i in self.mesh.boundaries['upperWall']:
                    uface[i] = BC['upperWall'][0]
                elif i in self.mesh.boundaries['lowerWall']:
                    uface[i] = BC['lowerWall'][0]
                else:
                    uface[i] = BC['frontAndBack'][0]
            else:
                PF_mag = np.linalg.norm(face_centres[i] - cell_centres[cell])
                PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[cell])
                uface[i] = u[cell] + (PF_mag * (u[neighbour]-u[cell])) / PN_mag

        return uface
    
    def face_flux(self, u, v, z, BC):

        """
        Function to calculate face flux

        Args:
            u (np.array): x velocity field
            v (np.array): y velocity field
            z (np.array): z velocity field
        Returns:
            F (np.array): face flux field

        """

        F = []

        uface = self.face_velocity(u, BC)
        vface = self.face_velocity(v, BC)
        zface = self.face_velocity(z, BC)
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

        # loops through owner neighbour pairs
        for i in range(len(owner_neighbour)):

            cell = owner_neighbour[i][0]
            neighbour = owner_neighbour[i][1]

            # zero gradient boundary condition
            if neighbour == -1:
                delta_p_face[i] = 0
                continue

            delta_p_face[i] = (p_field[neighbour] - p_field[cell])
        
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

        F = F.copy()

        owner_neighbours = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_area_vectors = self.mesh.face_area_vectors()
        delta_p_face = self.face_pressure(p_field)

        # loops through owner neighbour pairs and corrected face fluxes
        for i in range(len(owner_neighbours)):
            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]
            face_area_vector = face_area_vectors[i]
            face_mag = np.linalg.norm(face_area_vector)

            # nothing happens at boundary due to 0 gradient boundary conditions
            if neighbour == -1:
                # zero gradient boundary condition
                F[i] -= 0
                continue
            
            d_mag = np.linalg.norm(cell_centres[cell] - cell_centres[neighbour])

            # pressure coefficent
            aPN = (face_mag / d_mag) * raP[cell]

            # correct face flux
            F[i] -= aPN * delta_p_face[i]

        return F

    def cell_centre_correction(self, raP, u, v, z, p_field):

        """
        Function to correct cell centred velocities

        Args:
            raP (np.array): reciprocal of momentum matrix diagonal
            u (np.array): x velocity field (HbyAx operator)
            v (np.array): y velocity field (HbyAy operator)
            p_field (np.array): pressure field
        Returns:
            u (np.array): corrected x velocity field
            v (np.array): corrected y velocity field
        """
        u = u.copy()
        v = v.copy()
        z = z.copy()

        #delta_px, delta_py = self.cell_centre_pressure(p_field)
        delta_px, delta_py, delta_pz = self.cell_pressure_backward(p_field)

        # cell volumes
        cell_vols = self.mesh.cell_volumes()

        for cell in range(self.mesh.num_cells()):

            u[cell] -= delta_px[cell] * raP[cell]# * cell_vols[cell]
            v[cell] -= delta_py[cell] * raP[cell]# * cell_vols[cell]
            z[cell] -= delta_pz[cell] * raP[cell]# * cell_vols[cell]

        return u, v, z

    def cell_pressure_centred(self, p_field):

        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        delta_px = np.zeros_like(p_field)
        delta_py = np.zeros_like(p_field)
        delta_pz = np.zeros_like(p_field)
        owner_neighbour = self.mesh.cell_owner_neighbour()
        d_mag = np.linalg.norm(cell_centres[owner_neighbour[0][0]] - cell_centres[owner_neighbour[0][1]])

        for i, owner_neighbour in enumerate(owner_neighbour):

            owner = owner_neighbour[0]
            neighbour = owner_neighbour[1]
            sf = face_area_vectors[i]

            if neighbour == -1:
                # zero gradient boundary
                continue
            elif sf[0] != 0:
                delta_px[owner] += (p_field[neighbour]-p_field[owner])
                delta_px[neighbour] -= (p_field[neighbour]-p_field[owner])
            elif sf[1] != 0:
                delta_py[owner] += (p_field[neighbour]-p_field[owner])
                delta_py[neighbour] -= (p_field[neighbour]-p_field[owner])

        delta_px /= (2*d_mag)
        delta_py /= (2*d_mag)

        return delta_px, delta_py, delta_pz
    
    def cell_pressure_forward(self, p_field):

        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        delta_px = np.zeros_like(p_field)
        delta_py = np.zeros_like(p_field)
        delta_pz = np.zeros_like(p_field)
        owner_neighbour = self.mesh.cell_owner_neighbour()
        d_mag = np.linalg.norm(cell_centres[owner_neighbour[0][0]] - cell_centres[owner_neighbour[0][1]])

        for i, owner_neighbour in enumerate(owner_neighbour):

            owner = owner_neighbour[0]
            neighbour = owner_neighbour[1]
            sf = face_area_vectors[i]

            if neighbour == -1:
                # zero gradient boundary
                continue
            elif sf[0] != 0:
                delta_px[owner] += (p_field[neighbour]-p_field[owner])
            elif sf[1] != 0:
                delta_py[owner] += (p_field[neighbour]-p_field[owner])

        delta_px /= d_mag
        delta_py /= d_mag

        return delta_px, delta_py, delta_pz
    
    def cell_pressure_backward(self, p_field):

        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        delta_px = np.zeros_like(p_field)
        delta_py = np.zeros_like(p_field)
        delta_pz = np.zeros_like(p_field)
        owner_neighbour = self.mesh.cell_owner_neighbour()
        d_mag = np.linalg.norm(cell_centres[owner_neighbour[0][0]] - cell_centres[owner_neighbour[0][1]])

        for i, owner_neighbour in enumerate(owner_neighbour):

            owner = owner_neighbour[0]
            neighbour = owner_neighbour[1]
            sf = face_area_vectors[i]

            if neighbour == -1:
                # zero gradient boundary
                continue
            elif sf[0] != 0:
                delta_px[neighbour] += (p_field[neighbour]-p_field[owner])
            elif sf[1] != 0:
                delta_py[neighbour] += (p_field[neighbour]-p_field[owner])

        delta_px /= d_mag
        delta_py /= d_mag
 
        return delta_px, delta_py, delta_pz
    
    def cell_pressure(self, p_field):

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
    
    def cell_centre_pressure2(self, A, b, u, raP):

        H = self.H(A, b, u)
        delta_p = H - (1/raP) * u
        
        return delta_p
    
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
    
    def H(self, A, b, u):

        """
        Function to calculate H operator

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum source
            u (np.array): velocity field
        Returns:
            H (np.array): H operator
        """

        H = b.copy()
        owner_neighbours = self.mesh.cell_owner_neighbour()

        for i in range(len(owner_neighbours)):

            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]

            if neighbour == -1:
                continue
            H[cell] -= A[cell, neighbour] * u[neighbour]
            H[neighbour] -= A[neighbour, cell] * u[cell]

        return H
    
    def HbyA(self, A, b, u, raP):

        """
        Function to calculate HbyA operator to enforce divergence free velocity

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum source
            u (np.array): velocity field
            raP (np.array): reciprocal of diagonal coefficients
        Returns:
            HbyA (np.array): HbyA operator
        """

        HbyA = self.H(A, b, u)

        HbyA *= raP

        return HbyA
    
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

        return total_flux.flatten()

    def residuals_combined(self, Ax, bx, Ay, by, u, v):

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
    
    def residual(self, A, b, u):

        """
        Function to calculate residual for SIMPLE.

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum source
            u (np.array): current velocity field
        Returns:
            res (float): residual
        """

        return np.linalg.norm(b - np.matmul(A, u))
    
    def SIMPLE_loop(self, u, v, z, p, k, e, veff, F, BC):

        """
        Function to simulate singular SIMPLE loop that can be repeatedly called.

        Args:
            u (np.array): x velocity field
            v (np.array): y velocity field
            z (np.array): z velocity field
            p (np.array): pressure field
            k (np.array): turbulence kinetic energy field
            e (np.array): turbulence kinetic energy dissipation field
            veff (np.array): effective viscosity field
            F (np.array): face flux field
            BC (dict): boundary conditions
        Returns:
            u (np.array): corrected cell-centred x velocity field
            v (np.array): corrected cell-centred y velocity field
            p_field (np.array): updated pressure field
            k (np.array): updated turbulence kinetic energy field
            e (np.array): updated turbulence kinetic energy dissipation field
            veff (np.array): updated effective viscosity field
            Fcorr (np.array): corrected face flux field
            SIMPLE_res (float): resiudal of SIMPLE loop
            GS_res_x (float): final residual of x Gauss-seidel loop
            GS_res_y (float): final residual of y Gauss-seidel loop
        """

        #avoiding numpy behaviour
        u = u.copy()
        v = v.copy()
        z = z.copy()
        F = F.copy()
        p = p.copy()

        # Momentum Predictor
        Ax, bx = self.momentum_disc(u, F, veff, 'u', BC)
        Ay, by = self.momentum_disc(v, F, veff, 'v', BC)
        Az, bz = self.momentum_disc(z, F, veff, 'w', BC)

        # get momentum coefficients for report
        num_cells = self.mesh.num_cells()

        uplus1, exitcode = bicg(Ax, bx, x0=u, maxiter=200)
        vplus1, exitcode = bicg(Ay, by, x0=v, maxiter=200)
        zplus1, exitcode = bicg(Az, bz, x0=z, maxiter=200)

        resx_momentum = [self.residual(Ax, bx, u), self.residual(Ax, bx, uplus1)]
        resy_momentum = [self.residual(Ay, by, v), self.residual(Ay, by, vplus1)]

        # reciprocal of diagonal coefficients
        raP = self.raP(Ax)

        # HbyA operators
        HbyAx = self.HbyA(Ax, bx, uplus1, raP) # u velocity
        HbyAy = self.HbyA(Ay, by, vplus1, raP) # v velocity
        HbyAz = self.HbyA(Az, bz, zplus1, raP) # z velocity

        Fpre = self.face_flux(HbyAx, HbyAy, HbyAz, BC)

        # Pressure corrector
        Ap, bp = self.pressure_disc(Fpre, raP, BC)
        p_field, exitcode = bicg(Ap, bp, x0=p, maxiter=200)
        res_pressure = [self.residual(Ap, bp, p), self.residual(Ap, bp, p_field)]

        # Face flux correction
        Fcorr = self.face_flux_correction(Fpre, raP, p_field)

        # Explicit pressure under-relaxation
        p_field = p + self.alpha_p * (p_field - p)

        # Cell-centred correction
        uplus1, vplus1, zplus1 = self.cell_centre_correction(raP, uplus1, vplus1, zplus1, p_field)

        # turbulence systems
        Ak, bk = self.k_disc(k, e, F, BC)
        k_field, exitcode = bicg(Ak, bk, x0=k, maxiter=200)
        Ae, be = self.e_disc(k, e, F, BC)
        e_field, exitcode = bicg(Ae, be, x0=e, maxiter=200)

        # recalculating turbulent parameters
        veff = self.EffectiveVisc(k, e, 1)

        #res_SIMPLE = [self.residual(Ax, bx, uplus1), self.residual(Ay, bx, vplus1)]
        res_SIMPLE = [np.linalg.norm(u-uplus1), np.linalg.norm(v-vplus1)]

        return uplus1, vplus1, zplus1, p_field, k_field, e_field, veff, F, res_SIMPLE, resx_momentum, resy_momentum, res_pressure
    
    def iterate(self, u, v, w, p, k, e, BC, tol=1e-6, maxIts=100):
    
        """
        SIMPLE algorithm loop.

        Args:
            u (np.array): x velocity field
            v (np.array): y velocity field
            p (np.array): pressure field
            k (np.array): turbulence kinetic energy field
            e (np.array): turbulence kinetic energy dissipation field
            BC (dict): boundary conditions
            tol (float): algorithm tolerance
            maxIts (int): maximum number of iterations
        Returns:
            u (np.array): final cell-centred x velocity field
            v (np.array): final cell-centred y velocity field
            p_field (np.array): final pressure field
            k (np.array): final turbulence kinetic energy field
            e (np.array): final turbulence kinetic energy dissipation field
            res_SIMPLE_ls (list): list of SIMPLE residuals
        """ 

        # avoiding numpy behaviour
        u = u.copy()
        v = v.copy()
        p = p.copy()

        # Initial flux to feed in
        F = self.face_flux(u, v, w, BC)

        # Lists to store residuals
        res_SIMPLE_ls = []
        resx_momentum_ls = []
        resy_momentum_ls = []
        res_pressure_ls = []
        its = 0

        veff = self.EffectiveVisc(k, e, 1)

        # SIMPLE loop - will break if residual is less than tolerance
        for i in range(maxIts):
            print("Iteration: " + str(i+1))
            u, v, w, p, k, e, veff, F, res_SIMPLE, resx_momentum, resy_momentum, res_pressure = self.SIMPLE_loop(u, v, w, p, k, e, veff, F, BC)
            res_SIMPLE_ls.append(res_SIMPLE)
            resx_momentum_ls.append(resx_momentum)
            resy_momentum_ls.append(resy_momentum)
            res_pressure_ls.append(res_pressure)
            self.writer.WriteIteration(u, v, w, p, k, e, F, i+1)
            its += 1
            if (i+1 > 10):
                if res_SIMPLE[0] < tol and res_SIMPLE[1] < tol:
                    print(f"Simulation converged in {i+1} iterations")
                    break

        return u, v, w, p, k, e, F, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure_ls, its