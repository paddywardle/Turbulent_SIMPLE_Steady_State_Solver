import numpy as np

class WallFunctions():

    def __init__(self, mesh, Cmu, kap):

        self.mesh = mesh
        self.Cmu = Cmu
        self.kap = kap

    def kWallFunction(self, u, v, w, G, k, e, nu, nueff, BC):

        G = G.copy()
        e = e.copy()
        U = np.dstack((u, v, w))[0]
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                
                if (i in self.mesh.boundaries['upperWall']) or (i in self.mesh.boundaries['lowerWall']):

                    n = face_area_vectors[i] / np.linalg.norm(face_area_vectors[i])
                    y = np.linalg.norm(face_centres[i] - cell_centres[cell])
                    yVis = (nu[i] * 11.225) / ((self.Cmu**0.25) * (k[cell] ** 0.5))
                    yPlus = (self.Cmu ** 0.25 * k[cell] ** 0.5 * y) / (nu[i])
                    snGrad = U[cell] - n * np.dot(n, U[cell])
                    magUwGrad = np.linalg.norm(snGrad)
                    if (y > yVis):
                        print("hello in WallFunc")
                        G[cell] = (nueff[i] * magUwGrad) / ((self.Cmu ** 0.25) * self.kap * y)
                        e[cell] = ((self.Cmu ** 0.75) * (k[cell] ** 1.5)) / (self.kap * y)
                    else:
                        G[cell] = 0
                        e[cell] = (2 * nu[i] * k[cell]) / (y ** 0.5)

        return G, e

    def eWallFunction(self, A, b, k, e, nu, BC):

        A = A.copy()
        b = b.copy()

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:

                if (i in self.mesh.boundaries['upperWall']) or (i in self.mesh.boundaries['lowerWall']):
                    y = np.linalg.norm(face_centres[i] - cell_centres[cell])
                    yVis = (nu[i] * 11.225) / ((self.Cmu**0.25) * (k[cell] ** 0.5))
                    A[cell, cell] += (10 ** 10)
                    b[cell] += e[cell] * (10 ** 10)

        return A, b

    def nutWallFunction(self, nu_face, nut_face, k, BC):

        nut_face = nut_face.copy()

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:

                if (i in self.mesh.boundaries['upperWall']) or (i in self.mesh.boundaries['lowerWall']):
                    y = np.linalg.norm(face_centres[i] - cell_centres[cell])
                    yVis = (nu_face[i] * 11.225) / ((self.Cmu**0.25) * (k[cell] ** 0.5))
                    
                    if (y > yVis):
                        nut_face[i] = (self.Cmu ** 0.25) * (k[cell] * y / nu_face[i])
                    else:
                        nut_face[i] = 0

        return nut_face
