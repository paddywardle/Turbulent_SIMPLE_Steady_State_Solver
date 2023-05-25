import numpy as np
from scipy.sparse.linalg import bicg, bicgstab, cg, spilu
from scipy.sparse import csc_matrix
import sys

from fv.MomentumEq.MomentumSystem import MomentumSystem
from fv.PressureLaplacian.Laplacian import Laplacian
from fv.fvMatrices.fvc.Grad import Grad
from fv.TurbulenceModel.TurbulenceModel import TurbulenceModel
from fv.TurbulenceModel.WallFunctions.WallFunctions import WallFunctions
from fv.fvMatrices.fvMatrix import fvMatrix
from Tensor.Tensor import Tensor

class SIMPLE(MomentumSystem, Laplacian, TurbulenceModel, fvMatrix, Grad, Tensor, WallFunctions):

    """
    Class to hold all the functionality for the Semi-Implicit Algorithm for Pressure-Linked Equations (SIMPLE)
    """

    def __init__(self, writer, mesh, conv_scheme, viscosity, alpha_u, alpha_p, Cmu, C1, C2, C3, sigmak, sigmaEps, kap, E):
        
        self.writer = writer
        MomentumSystem.__init__(self, mesh, conv_scheme, alpha_u)
        #Laplacian.__init__(self, mesh)
        WallFunctions.__init__(self, mesh, Cmu, kap, E)
        TurbulenceModel.__init__(self, mesh, conv_scheme, alpha_u, Cmu, C1, C2, C3, sigmak, sigmaEps)
        fvMatrix.__init__(self, mesh)
        self.viscosity = viscosity
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
        cell_volumes = self.mesh.cell_volumes()

        for cell in range(self.mesh.num_cells()):

            V = cell_volumes[cell]

            u[cell] -= (gradP[0][cell] * raP[cell]) / V
            v[cell] -= (gradP[1][cell] * raP[cell]) / V
            z[cell] -= (gradP[2][cell] * raP[cell]) / V

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

        return np.linalg.norm(b - np.matmul(A, u), ord=1)
    
    def SIMPLE_loop(self, u, v, z, p, k, e, nu_list, F, BC):

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

        nu, nut, nueff_face = nu_list
        nu_face = self.veff_face(nu)
        nut_face = self.veff_face(nut)
        
        # Momentum Predictor
        Ax, bx = self.MomentumDisc(u, F, nueff_face, 1, 'u', BC)
        Ay, by = self.MomentumDisc(v, F, nueff_face, 1, 'v', BC)
        #Az, bz = self.MomentumDisc(z, F, nueff_face, 1, 'w', BC)

        #uplus1, exitcode = bicgstab(Ax, bx, x0=u, tol=1e-7)
        #vplus1, exitcode = bicgstab(Ay, by, x0=v, tol=1e-7)
        # zplus1, exitcode = bicgstab(Az, bz, x0=z, tol=1e-7)

        uplus1 = np.linalg.solve(Ax, bx)
        vplus1 = np.linalg.solve(Ay, by)
        zplus1 = z
        
        resx_momentum = [self.residual(Ax, bx, u), self.residual(Ax, bx, uplus1)]
        resy_momentum = [self.residual(Ay, by, v), self.residual(Ay, by, vplus1)]

        # reciprocal of diagonal coefficients
        raP = self.raP(Ax)
        raP_face = self.face_raP(raP)

        # # HbyA operators
        # HbyAx = self.HbyA(Ax, bx, uplus1, raP) # u velocity
        # HbyAy = self.HbyA(Ay, by, vplus1, raP) # v velocity
        # HbyAz = self.HbyA(Az, bz, zplus1, raP) # z velocity

        # # Face flux correction
        # Fpre = self.face_flux(HbyAx, HbyAy, HbyAz, BC)
        Fpre = self.face_flux(uplus1, vplus1, zplus1, BC)
        
        # Pressure corrector
        Ap, bp = self.PressureDisc(Fpre, raP_face, BC)

        #p_field, exitcode = bicgstab(Ap, bp, x0=p, tol=1e-7)
        p_field = np.linalg.solve(Ap, bp)
        
        res_pressure = [self.residual(Ap, bp, p), self.residual(Ap, bp, p_field)]

        # Face flux correction
        Fcorr = Fpre - self.face_pressure(p_field, Ap, raP, BC)

        # Explicit pressure under-relaxation
        p_field = p + self.alpha_p * (p_field - p)

        # Cell-centred correction
        uplus1, vplus1, zplus1 = self.cell_centre_correction(raP, uplus1, vplus1, zplus1, p_field, BC)
        
        # turbulence systems
        gradU = self.gradU(uplus1, vplus1, zplus1, BC)
        G = 2 * nut * self.magSqr(self.Symm(gradU))
        #G, e = self.kWallFunction(uplus1, vplus1, zplus1, G, k, e, nu_face, nueff_face)

        Ae, be = self.EDisc(k, e, Fpre, nu_face, nut_face, G, 1, BC)
        #Ae, be = self.eWallFunction(Ae, be, e)
        #e_field, exitcode = bicgstab(Ae, be, x0=e, tol=1e-7)
        e_field = np.linalg.solve(Ae, be)

        Ak, bk = self.KDisc(k, e, Fpre, nu_face, nut_face, G, 1, BC)
        #k_field, exitcode = bicgstab(Ak, bk, x0=k, tol=1e-7)
        k_field = np.linalg.solve(Ak, bk)

        # viscosity wall function
        nut = self.TurbulentVisc(k_field, e_field)
        nut_face = self.veff_face(nut)
        #nut_face = self.nutWallFunction(nu_face, nut_face, k, uplus1, vplus1, zplus1)
        nueff_face = nu_face + nut_face
        nu_list = [nu, nut, nueff_face]
        
        res_SIMPLE = np.array([self.residual(Ax, bx, uplus1), self.residual(Ay, by, vplus1), self.residual(Ap, bp, p_field), self.residual(Ak, bk, k_field), self.residual(Ae, be, e_field)])
        #res_SIMPLE = [np.linalg.norm(u-uplus1), np.linalg.norm(v-vplus1)]

        return uplus1, vplus1, zplus1, p_field, k_field, e_field, nu_list, Fcorr, res_SIMPLE, resx_momentum, resy_momentum, res_pressure
    
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

        # Project viscosity onto faces
        nu = self.viscosity * np.ones((self.mesh.num_cells(),))
        nut = self.TurbulentVisc(k, e)
        nut_face = self.veff_face(nut) #self.nutWallFunction(self.veff_face(nu), self.veff_face(nut), k, u, v, w)
        nueff_face = self.veff_face(nu) + nut_face
        nu_list = [nu, nut, nueff_face]

        # SIMPLE loop - will break if residual is less than tolerance
        for i in range(maxIts):
            
            print("Iteration: " + str(i+1))
            
            u, v, w, p, k, e, nu_list, F, res_SIMPLE, resx_momentum, resy_momentum, res_pressure = self.SIMPLE_loop(u, v, w, p, k, e, nu_list, F, BC)

            if i+1 == 1:
                res_SIMPLE_init = res_SIMPLE
            print("Residual: ", res_SIMPLE)
            res_SIMPLE = np.divide(res_SIMPLE, res_SIMPLE_init)
            print("Residual Normalized: ", res_SIMPLE)
            res_SIMPLE_ls.append(res_SIMPLE/res_SIMPLE_init)
            resx_momentum_ls.append(resx_momentum)
            resy_momentum_ls.append(resy_momentum)
            res_pressure_ls.append(res_pressure)
            
            self.writer.WriteIteration(u, v, w, p, k, e, nu_list[1], F, i+1)
            its += 1
            
            if (i+1 > 5):
                if res_SIMPLE[0] < tol and res_SIMPLE[1] < tol:
                    print(f"Simulation converged in {i+1} iterations")
                    break

        return u, v, w, p, k, e, F, res_SIMPLE_ls, resx_momentum_ls, resy_momentum_ls, res_pressure_ls, its
