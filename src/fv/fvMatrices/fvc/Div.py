import numpy as np

class Div():

    def __init__(self, mesh):

        self.mesh = mesh

    def fvcDiv(self, F):

        V = self.mesh.cell_volumes()
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()

        ssf = np.zeros((self.mesh.num_cells(),))

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                ssf[owner] += F[i]
                continue

            ssf[owner] += F[i]
            ssf[neighbour] -= F[i]

        #ssf /= V

        return ssf

        
