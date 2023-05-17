import numpy as np

class Ddt():

    """
    Class to discretise the temporal term of the Incompressible Navier-Stokes equation to produce a linear system, using a finite volume discretisation approach.
    """

    def __init__(self, mesh):

        self.mesh = mesh

    def fvmDdt(self, x, deltaT):

        """
        This function discretises the temporal term to get the diagonal, off-diagonal and source contributions to the linear system for the convection term.

        Args:
            x (np.array): cell values
            deltaT (float): timestep
        Returns:
            np.array: N x N matrix defining contributions of convective and diffusion terms to the linear system.

        """
        
        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        cell_volumes = self.mesh.cell_volumes()

        for cell in range(self.mesh.num_cells()):

            if neighbour == -1:
                continue

            FN_cell = F[i]
            V = self.mesh.cell_volumes()[cell]

            A[cell, cell] = V / deltaT

            b[cell] = (V * x[cell]) / deltaT

        return A, b
