import numpy as np

class SuSp():

    def __init__(self, mesh):

        self.mesh = mesh

    def Sp(self, sp):

        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        V = self.mesh.cell_volumes()

        for i, val in enumerate(sp):
            A[i,i] += val * V[i]

        return A, b

    def SuSp(self, susp, vf):

        N = len(self.mesh.cells)
        A = np.zeros((N, N))
        b = np.zeros((N, 1)).flatten()

        V = self.mesh.cell_volumes()

        for i, (s, v) in enumerate(zip(susp, vf)):
            A[i,i] += max(s, 0) * V[i]
            b[i] -= min(s, 0) * v * V[i] 

        return A, b
