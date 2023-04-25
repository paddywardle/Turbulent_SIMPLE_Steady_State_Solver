import numpy as np
from scipy.sparse.linalg import bicg, bicgstab, cg, spilu
from scipy.sparse import csc_matrix
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
    
    def face_velocity(self, u, BC, vel_comp):

        """
        Function to calculate face velocity

        Args:
            u (np.array): velocity field
            BC (float): boundary condition
        Returns:
            uface (np.array): face velocity field

        """

        if vel_comp == "u":
            idx = 0
        elif vel_comp == "v":
            idx = 1
        else:
            idx = 2

        uface = np.zeros((self.mesh.num_faces(), 1))

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        # loops through owner neighbour pairs
        # applies boundary condition if neighbour = -1
        # linearly interpolates velocity onto the face otherwise
        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):
            
            if (neighbour == -1):
                if i in self.mesh.boundaries['inlet']:
                    uface[i] = BC['inlet'][idx]
                elif i in self.mesh.boundaries['outlet']:
                    uface[i] = u[owner]
                elif i in self.mesh.boundaries['upperWall']:
                    uface[i] = BC['upperWall'][idx]
                elif i in self.mesh.boundaries['lowerWall']:
                    uface[i] = BC['lowerWall'][idx]
                else:
                    uface[i] = BC['frontAndBack'][idx]
                continue

            fN_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
            PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[owner])
            fx = fN_mag / PN_mag;

            #uface[i] = u[owner] + (PF_mag * (u[neighbour]-u[cell])) / PN_mag
            uface[i] = fx * u[owner] + (1 - fx) * u[neighbour]

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

        uface = self.face_velocity(u, BC, "u")
        vface = self.face_velocity(v, BC, "v")
        zface = self.face_velocity(z, BC, "w")
        face_area_vectors = np.squeeze(self.mesh.face_area_vectors())
        
        # horizontally stack x, y and z face velocity values
        face_velocity = np.squeeze(np.hstack((uface, vface, zface)))
        # loop through and dot product face velocities with face area vectors to get face flux
        for i in range(len(face_velocity)):
            F_current = np.dot(face_area_vectors[i], face_velocity[i])
            F.append(F_current)

        F = np.asarray(F)

        return F
    
    def face_pressure(self, p_field, BC):

        """
        Function to calculate face pressure gradient.

        Args:
            p_field (np.array): pressure field
        Returns:
            delta_p_face (np.array): face pressure gradient

        """

        delta_p_face = np.zeros((self.mesh.num_faces(),))
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()

        # loops through owner neighbour pairs
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            # zero gradient boundary condition
            if neighbour == -1:
                if i in self.mesh.boundaries['outlet']:
                    delta_p_face[i] = BC['outlet'][3] - p_field[cell]
                else:
                    delta_p_face[i] = 0
                continue

            delta_p_face[i] = (p_field[neighbour] - p_field[cell])
        
        return delta_p_face
    
    def face_raP(self, raP):

        """
        Function to calculate face value of momentum coefficients

        Args:
            raP (np.array): reciprocal of diagonal matrix coefficients
        Returns:
            raP_face (np.array): reciprocal face diagonal momentum values

        """

        raP_face = np.zeros((self.mesh.num_faces(),))
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_centres = self.mesh.face_centres()
        cell_centres = self.mesh.cell_centres()

        # loops through owner neighbour pairs and linearly interpolates ap onto the face
        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                # zero gradient Neumann
                raP_face[i] = raP[owner]
                continue
            
            fN_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
            PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[owner])
            fx = fN_mag / PN_mag;
            raP_face[i] = fx * raP[owner] + (1 - fx) * raP[neighbour]
        
        return raP_face
    
    def face_flux_correction(self, F, raP, p_field, BC):

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

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        face_area_vectors = self.mesh.face_area_vectors()
        delta_p_face = self.face_pressure(p_field, BC)
        raP_face = self.face_raP(raP)

        # loops through owner neighbour pairs and corrected face fluxes
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):
            face_area_vector = face_area_vectors[i]
            face_mag = np.linalg.norm(face_area_vector)

            # nothing happens at boundary due to 0 gradient boundary conditions
            if neighbour == -1:
                # zero gradient boundary condition
                d_mag = np.linalg.norm(cell_centres[cell] - face_centres[i])
            else:
                d_mag = np.linalg.norm(cell_centres[cell] - cell_centres[neighbour])

            # pressure coefficent
            aPN = (face_mag / d_mag) * raP_face[i]

            # correct face flux
            F[i] -= aPN * delta_p_face[i]

        return F

    def cell_centre_correction(self, raP, u, v, z, p_field, BC):

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

        #delta_px, delta_py, delta_pz = self.cell_pressure_backward(p_field)
        gradP = self.gradP(p_field, BC)

        for cell in range(self.mesh.num_cells()):

            #u[cell] -= delta_px[cell] * raP[cell]# * cell_vols[cell]
            #v[cell] -= delta_py[cell] * raP[cell]# * cell_vols[cell]
            #z[cell] -= delta_pz[cell] * raP[cell]# * cell_vols[cell]

            u[cell] -= gradP[0][cell] * raP[cell]
            v[cell] -= gradP[1][cell] * raP[cell]
            z[cell] -= gradP[2][cell] * raP[cell]

        return u, v, z

    def gradP(self, p_field, BC):
        
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        V = self.mesh.cell_volumes()
        
        p_grad = np.zeros((3, self.mesh.num_cells()))
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                if i in self.mesh.boundaries['outlet']:
                    p_face = BC["outlet"][3]
                else:
                    p_face = p_field[owner]

                cmptGrad = 0
                for cmptSf in range(3):
                    p_grad[cmptGrad][owner] += p_face * face_area_vectors[i][cmptSf]
                    cmptGrad += 1
                    
                continue

            fN_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
            PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[owner])
            fx = fN_mag / PN_mag;

            p_face = fx * p_field[owner] + (1 - fx) * p_field[neighbour]

            cmptGrad = 0
            for cmptSf in range(3):
                p_grad[cmptGrad][owner] += p_face * face_area_vectors[i][cmptSf]

                p_grad[cmptGrad][neighbour] -= p_face * face_area_vectors[i][cmptSf]

                cmptGrad += 1

        for cmpt in range(3):
            p_grad[cmpt] /= V

        return p_grad
        
    def cell_pressure_centred(self, p_field):

        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        
        delta_px = np.zeros_like(p_field)
        delta_py = np.zeros_like(p_field)
        delta_pz = np.zeros_like(p_field)

        dist_x = np.zeros_like(p_field)
        dist_y = np.zeros_like(p_field)
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            sf = face_area_vectors[i]

            if sf[0] != 0:
                if neighbour == -1:
                    delta_px[owner] += p_field[owner]
                    dist_x[owner] += np.linalg.norm(cell_centres[owner] - face_centres[i])
                    continue
                dist_x[owner] += np.linalg.norm(cell_centres[owner] - cell_centres[neighbour])
                dist_x[neighbour] += np.linalg.norm(cell_centres[owner] - cell_centres[neighbour])
                delta_px[owner] += p_field[neighbour]
                delta_px[neighbour] -= p_field[owner]
            elif sf[1] != 0:
                if neighbour == -1:
                    delta_py[owner] += p_field[owner]
                    dist_y[owner] += np.linalg.norm(cell_centres[owner] - face_centres[i])
                    continue
                dist_y[owner] += np.linalg.norm(cell_centres[owner] - cell_centres[neighbour])
                dist_y[neighbour] += np.linalg.norm(cell_centres[owner] - cell_centres[neighbour])
                delta_py[owner] += p_field[neighbour]
                delta_py[neighbour] -= p_field[owner]
                
        print(dist_y[dist_y == 0])
        delta_px /= dist_x
        delta_py /= dist_y

        return delta_px, delta_py, delta_pz
    
    def cell_pressure_forward(self, p_field):

        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        delta_px = np.zeros_like(p_field)
        delta_py = np.zeros_like(p_field)
        delta_pz = np.zeros_like(p_field)
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        d_mag = np.linalg.norm(cell_centres[owner_neighbour[0][0]] - cell_centres[owner_neighbour[0][1]])

        for i, owner_neighbour in enumerate(cell_owner_neighbour):

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
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        d_mag = np.linalg.norm(cell_centres[cell_owner_neighbour[0][0]] - cell_centres[cell_owner_neighbour[0][1]])

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

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
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()

        # loops through owner neighbour pairs
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):
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
    
    def raP(self, A):

        """
        Function to calculate reciprocal of momentum diagonal.

        Args:
            A (np.array): momentum matrix
        Returns:
            np.array: array of reciprocals
        """
        
        raP = []
        
        V = self.mesh.cell_volumes()

        for i in range(len(A)):

            raP.append(1/A[i, i])

        raP *= V

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
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        V = self.mesh.cell_volumes()

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                continue
            
            H[owner] -= A[owner, neighbour] * u[neighbour]
            H[neighbour] -= A[neighbour, owner] * u[owner]

        H /= V
            
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

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        total_flux = np.zeros((self.mesh.num_cells(), 1))

        # loops through owner neighbour pairs and adds fluxes to owners and neighbours - skips neighbour if boundary
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):
            
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
        k = k.copy()
        e = e.copy()

        # Project viscosity onto faces
        veff_face = self.veff_face(veff)

        # Momentum Predictor
        Ax, bx = self.momentum_disc(u, F, veff_face, 'u', BC)
        Ax_sparse = csc_matrix(Ax)
        Mx = spilu(Ax_sparse)
        Mx = (Mx.L @ Mx.U).A
        Ax = Mx @ Ax
        bx = Mx @ bx
        
        Ay, by = self.momentum_disc(v, F, veff_face, 'v', BC)
        Ay_sparse = csc_matrix(Ay)
        My = spilu(Ay_sparse)
        My = (My.L @ My.U).A
        Ay = My @ Ay
        by = My @ by
        
        Az, bz = self.momentum_disc(z, F, veff_face, 'w', BC)
        
        # get momentum coefficients for report
        num_cells = self.mesh.num_cells()

        uplus1, exitcode = bicgstab(Ax, bx, x0=u, maxiter=200, tol=1e-5)
        vplus1, exitcode = bicgstab(Ay, by, x0=v, maxiter=200, tol=1e-5)
        zplus1, exitcode = bicgstab(Az, bz, x0=z, maxiter=200, tol=1e-5)

        resx_momentum = [self.residual(Ax, bx, u), self.residual(Ax, bx, uplus1)]
        resy_momentum = [self.residual(Ay, by, v), self.residual(Ay, by, vplus1)]

        # reciprocal of diagonal coefficients
        raP = self.raP(Ax)
        raP_face = self.face_raP(raP)

        # HbyA operators
        HbyAx = self.HbyA(Ax, bx, uplus1, raP) # u velocity
        HbyAy = self.HbyA(Ay, by, vplus1, raP) # v velocity
        HbyAz = self.HbyA(Az, bz, zplus1, raP) # z velocity

        Fpre = self.face_flux(HbyAx, HbyAy, HbyAz, BC)
        #Fpre = self.face_flux(uplus1, vplus1, zplus1, BC)

        # Pressure corrector
        Ap, bp = self.pressure_disc(Fpre, raP_face, BC)
        Ap_sparse = csc_matrix(Ap)
        Mp = spilu(Ap_sparse)
        Mp = (Mp.L @ Mp.U).A
        Ap = Mp @ Ap
        bp = Mp @ bp
        p_field, exitcode = cg(Ap, bp, x0=p, maxiter=200, tol=1e-6)        
        #p_field = np.linalg.inv(Ap) @ bp
        res_pressure = [self.residual(Ap, bp, p), self.residual(Ap, bp, p_field)]

        # Face flux correction
        Fcorr = self.face_flux_correction(Fpre, raP, p_field, BC)

        # Explicit pressure under-relaxation
        p_field = p + self.alpha_p * (p_field - p)

        # Cell-centred correction
        uplus1, vplus1, zplus1 = self.cell_centre_correction(raP, uplus1, vplus1, zplus1, p_field, BC)

        # turbulence systems
        Ak, bk = self.k_disc(k, e, F, BC)
        k_field, exitcode = bicgstab(Ak, bk, x0=k, maxiter=200, tol=1e-5)
        Ae, be = self.e_disc(k, e, F, BC)
        e_field, exitcode = bicgstab(Ae, be, x0=e, maxiter=200, tol=1e-5)

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
        k = k.copy()
        e = e.copy()

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
