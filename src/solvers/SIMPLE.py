import numpy as np
from scipy.sparse.linalg import bicg, bicgstab, cg, spilu
from scipy.sparse import csc_matrix
import sys

from fv.volField.MomentumSystem import MomentumSystem
from fv.scalarField.Laplacian import Laplacian
from fv.TurbulenceModel.TurbulenceModel import TurbulenceModel
from fv.fvMatrix import fvMatrix

class SIMPLE(MomentumSystem, Laplacian, TurbulenceModel, fvMatrix):

    """
    Class to hold all the functionality for the Semi-Implicit Algorithm for Pressure-Linked Equations (SIMPLE)
    """

    def __init__(self, writer, mesh, conv_scheme, viscosity, alpha_u, alpha_p, Cmu, C1, C2, C3, sigmak, sigmaEps):
        
        self.writer = writer
        MomentumSystem.__init__(self, mesh, conv_scheme, viscosity, alpha_u)
        Laplacian.__init__(self, mesh, conv_scheme)
        TurbulenceModel.__init__(self, mesh, conv_scheme, viscosity, alpha_u, Cmu, C1, C2, C3, sigmak, sigmaEps)
        fvMatrix.__init__(self, mesh)
        self.alpha_u = alpha_u
        self.alpha_p = alpha_p

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

        gradP = self.gradP(p_field, BC)

        for cell in range(self.mesh.num_cells()):

            #u[cell] -= delta_px[cell] * raP[cell]# * cell_vols[cell]
            #v[cell] -= delta_py[cell] * raP[cell]# * cell_vols[cell]
            #z[cell] -= delta_pz[cell] * raP[cell]# * cell_vols[cell]

            u[cell] -= gradP[0][cell] * raP[cell]
            v[cell] -= gradP[1][cell] * raP[cell]
            z[cell] -= gradP[2][cell] * raP[cell]

        return u, v, z

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
        Ax, bx = self.MomentumDisc(u, F, veff_face, 'u', BC)
        Ay, by = self.MomentumDisc(v, F, veff_face, 'v', BC)
        Az, bz = self.MomentumDisc(z, F, veff_face, 'w', BC)
        
        # get momentum coefficients for report
        num_cells = self.mesh.num_cells()

        #uplus1, exitcode = bicgstab(Ax, bx, x0=u, maxiter=200, tol=1e-5)
        #vplus1, exitcode = bicgstab(Ay, by, x0=v, maxiter=200, tol=1e-5)
        #zplus1, exitcode = bicgstab(Az, bz, x0=z, maxiter=200, tol=1e-5)

        uplus1 = np.linalg.solve(Ax, bx)
        vplus1 = np.linalg.solve(Ay, by)
        zplus1 = z
        
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

        # Pressure corrector
        Ap, bp = self.PressureDisc(Fpre, raP_face, BC)

        #p_field, exitcode = bicgstab(Ap, bp, x0=p, maxiter=200, tol=1e-6)
        p_field = np.linalg.solve(Ap, bp)
        
        res_pressure = [self.residual(Ap, bp, p), self.residual(Ap, bp, p_field)]

        # Face flux correction
        Fcorr = Fpre - self.face_pressure(p_field, Ap, raP, BC)

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
