import numpy as np
import sys

class Grad():

    def __init__(self):

        pass

    def gradP(self, p_field, BC):
        
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        V = self.mesh.cell_volumes()
        
        p_grad = np.zeros((3, self.mesh.num_cells()))
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                if i in self.mesh.boundaries['outlet']:
                    p_face = BC["outlet"][3]
                else:
                    p_face = p_field[owner]

                cmptGrad = 0
                for cmptSf in range(3):
                    p_grad[cmptGrad][owner] += p_face * face_area_vectors[i][cmptSf]
                    cmptGrad += 1
                    
                continue

            Nf_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
            Pf_mag = np.linalg.norm(face_centres[i] - cell_centres[owner])
            fx = Nf_mag / (Pf_mag + Nf_mag)

            p_face = fx * p_field[owner] + (1 - fx) * p_field[neighbour]

            cmptGrad = 0
            for cmptSf in range(3):
                p_grad[cmptGrad][owner] += p_face * face_area_vectors[i][cmptSf]

                p_grad[cmptGrad][neighbour] -= p_face * face_area_vectors[i][cmptSf]

                cmptGrad += 1

        for cmpt in range(3):
            p_grad[cmpt] /= V

        return p_grad

    def gradU(self, u, v, w, BC):
        
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        V = self.mesh.cell_volumes()
        noComponents = 3

        U = np.array([u, v, w])
        
        U_grad = np.zeros((noComponents*3, self.mesh.num_cells()))
        
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            uface = np.empty(noComponents)
            cmptGrad = 0

            for cmpt in range(noComponents):

                if neighbour == -1:
                    if i in self.mesh.boundaries['inlet']:
                        uface[cmpt] = BC['inlet'][cmpt]
                    elif i in self.mesh.boundaries['outlet']:
                        uface[cmpt] = U[cmpt][owner]
                    elif i in self.mesh.boundaries['upperWall']:
                        uface[cmpt] = BC['upperWall'][cmpt]
                    elif i in self.mesh.boundaries['lowerWall']:
                        uface[cmpt] = BC['lowerWall'][cmpt]
                    else:
                        uface[cmpt] = BC['frontAndBack'][cmpt]
                        
                    for cmptSf in range(3):
                        U_grad[cmptGrad][owner] += uface * face_area_vectors[i][cmptSf]
                        cmptGrad += 1
                    
                    continue

                Nf_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
                Pf_mag = np.linalg.norm(face_centres[i] - cell_centres[owner])
                fx = Nf_mag / (Pf_mag + Nf_mag)

                uface = fx * U[cmpt][owner] + (1 - fx) * U[cmpt][neighbour]

                for cmptSf in range(3):
                    U_grad[cmptGrad][owner] += uface * face_area_vectors[i][cmptSf]

                    U_grad[cmptGrad][neighbour] -= uface * face_area_vectors[i][cmptSf]

                    cmptGrad += 1

        for cmpt in range(3):
            U_grad[cmpt] /= V

        return U_grad
