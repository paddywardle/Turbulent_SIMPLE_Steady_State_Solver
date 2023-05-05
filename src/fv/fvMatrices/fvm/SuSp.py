import numpy as np

class SuSp():

    def __init__(self, mesh):

        self.mesh = mesh

    def Sp(self, sp, vf):

        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        V = self.mesh.cell_volumes()

        for i, (s, v) in enumerate(zip(sp, vf)):
            A[i,i] += s * V[i]

        return A, b

    def Su(self, su, vf):

        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        V = self.mesh.cell_volumes()

        for i, (s, v) in enumerate(zip(su, vf)):
            b[i] -= s * V[i] 

        return A, b

    def SuSp(self, susp, vf):

        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        V = self.mesh.cell_volumes()

        for i, (s, v) in enumerate(zip(susp, vf)):
            A[i,i] += max(s, 0) * V[i]
            b[i] -= (min(s, 0) * v) * V[i] 

        return A, b
