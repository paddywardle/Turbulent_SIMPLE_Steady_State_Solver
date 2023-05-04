import numpy as np
from fv.PressureLaplacian.LaplacianBCs import LaplacianBCs
from fv.fvMatrices.fvm.Laplacian import Laplacian
from fv.fvMatrices.fvc.Div import Div as fvcDiv

class Laplacian(Laplacian, fvcDiv, LaplacianBCs):

    """
    Class to discretise the pressure laplacian to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh):

        pass
        
    def PressureDisc(self, F, raP_face, BC):

        """
        This function discretises the pressure laplacian to get the diagonal, off-diagonal and source contributions to the linear system.

        Args:
            F (np.array): flux array
            raP_face (np.array): reciprocal of momentum diagonal coefficients
            BC (int): boundary conditions
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """

        Ap, bp = self.laplacian(raP_face)
        bp += self.fvcDiv(F)
        Ap, bp = self.LaplacianMatPressureBCs(Ap, bp, F, raP_face, BC)

        # set reference point
        #Ap[0,0] *= 1.1

        return Ap, bp
