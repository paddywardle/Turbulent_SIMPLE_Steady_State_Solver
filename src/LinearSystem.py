from Mesh import Mesh
from SparseMatrixCR import SparseMatrixCR
import numpy as np

class LinearSystem:

    def __init__(self, mesh, Re):

        self.mesh = mesh
        self.Re = Re

    def A_disc(self):

        N = self.mesh.cells * self.mesh.cells

        A = np.zeros((N, N))

        #for i in self.mesh.cells:

    def b_disc(self):

        pass

    def solver(self):

        pass

