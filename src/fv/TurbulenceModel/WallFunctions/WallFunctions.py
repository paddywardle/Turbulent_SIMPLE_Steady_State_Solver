import numpy as np
import math

class WallFunctions():

    def __init__(self, mesh, Cmu, kap, E):

        self.mesh = mesh
        self.Cmu = Cmu
        self.kap = kap
        self.E = E

    def kWallFunction(self, u, v, w, G, k, e, nu, nueff):

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
                    yPlus = ((self.Cmu ** 0.25) * (k[cell] ** 0.5) * y) / (nu[i])
                    snGrad = (U[cell] - n * np.dot(n, U[cell])) / y
                    magSnGrad = np.linalg.norm(snGrad)

                    if (y > yVis):
                        G[cell] = ((nueff[i] * magSnGrad)**2) / ((self.Cmu**0.25)* self.kap * (k[cell] ** 0.5) * y)
                        e[cell] = ((self.Cmu ** 0.75) * (k[cell] ** 1.5)) / (self.kap * y)
                    else:
                        G[cell] = 0
                        e[cell] = (2 * nu[i] * k[cell]) / (y ** 2)

        return G, e

    def eWallFunction(self, A, b, e):

        A = A.copy()
        b = b.copy()

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:

                if (i in self.mesh.boundaries['upperWall']) or (i in self.mesh.boundaries['lowerWall']):
                    A[cell, cell] += (10 ** 10)
                    b[cell] = e[cell] * (10 ** 10)

        return A, b

    def nutWallFunction(self, nu_face, nut_face, k, u, v, w):

        nut_face = nut_face.copy()
        U = np.dstack((u, v, w))[0]

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:

                if (i in self.mesh.boundaries['upperWall']) or (i in self.mesh.boundaries['lowerWall']):
                    y = np.linalg.norm(face_centres[i] - cell_centres[cell])
                    yVis = (nu_face[i] * 11.225) / ((self.Cmu**0.25) * (k[cell] ** 0.5))
                    yPlus = ((self.Cmu ** 0.25) * (k[cell] ** 0.5) * y) / (nu_face[i])
                    n = face_area_vectors[i] / np.linalg.norm(face_area_vectors[i])
                    snGrad = (U[cell] - n * np.dot(n, U[cell])) / y
                    magSnGrad = np.linalg.norm(snGrad)
                    
                    if (y > yVis):
                        nut_face[i] = (nu_face[i] * yPlus * self.kap) / (math.log(self.E * yPlus) - 1)
                    else:
                        nut_face[i] = 0

        return nut_face
