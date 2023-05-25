import numpy as np
from fv.MomentumEq.MomentumSystemBCs import MomentumSystemBCs
from fv.fvMatrices.fvm.Div import Div as fvmDiv
from fv.fvMatrices.fvm.Ddt import Ddt
from fv.fvMatrices.fvm.Laplacian import Laplacian
from fv.fvMatrices.fvMatrix import fvMatrix

class MomentumSystem(Ddt, fvmDiv, Laplacian, fvMatrix, MomentumSystemBCs):

    """
    Class to discretise the Incompressible Navier-Stokes equation to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh, conv_scheme, alpha_u):

        pass
    
    def MomentumDisc(self, u, F, veff, deltaT, vel_comp, BC):

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
        
        Addt, bddt = self.ddt(u, deltaT)
        Aconv, bconv = self.fvmDiv(F)
        Adiff, bdiff = self.laplacian(veff)

        Aconv, bconv = self.ConvMatMomentumBCs(Aconv, bconv, F, vel_comp, BC)
        Adiff, bdiff = self.DiffMatMomentumBCs(Adiff, bdiff, F, veff, vel_comp, BC)

        A = Addt + Aconv - Adiff
        b = bddt + bconv - bdiff

        A, b = self.relax(A, b, u)

        return A, b
