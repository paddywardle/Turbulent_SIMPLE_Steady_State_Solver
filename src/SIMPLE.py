import numpy as np
from LinearSystem import LinearSystem

class SIMPLE(LinearSystem):

    def __init__(self, mesh, viscosity, alpha_u, alpha_p):
        
        LinearSystem.__init__(self, mesh, viscosity, alpha_u)
        self.alpha_u = alpha_u
        self.alpha_p = alpha_p

    def initial_pressure(self):

        return np.zeros((self.mesh.num_cells(), self.mesh.num_cells()))
    
    def face_flux(self, u):

        uface = np.zeros((self.mesh.num_faces(), 1))

        for i in range(self.mesh.num_cells()):
            for j in self.mesh.neighbouring_cells()[i]:
                neighbour_faces = self.mesh.cells[j]
                shared_face = list(set(self.mesh.cells[i]).intersection(neighbour_faces))[0]
                uface[shared_face] = (u[i] + u[j])/2

        top_index = []

        for i in range(len(self.mesh.boundary_patches)):
            if self.mesh.boundary_patches[i][0] == "movingWall":
                top_index = self.mesh.boundary_patches[i+1]

        for i in top_index:
            uface[int(i)] = 1
        
        return uface
    
    def iterate(self, u, v, sweeps, tol=1e-6):

        p = self.initial_pressure()

        u_convected = u
        u_convecting = u
        v_convected = v
        v_convecting = v

        uface = self.face_flux(u)
        vface = self.face_flux(v)
        print(uface)

        # Ax, bx = self.momentum_disc(uface, [0, 1])
        # Ay, by = self.momentum_disc(vface, [0, 1])

        # uplus1 = self.gauss_seidel(Ax, bx, u)
        # vplus1 = self.gauss_seidel(Ay, by, v)

        # res_initial = np.linalg.norm(bx - np.matmul(Ax, u)) + 1e-6

        # res = res_initial / res_initial

        # while res > tol:

        #     uface = self.face_flux()
            

        # while not at convergence
        #
        # discretise momentum equation using A_disc and b_disc <- apply under-relaxation of pressure here?
        # 
        # use gauss-seidel to solve to get velocity
        # 
        # calculate face flux Fpre = sf*uf
        # 
        # discretise pressure equation 
        # 
        #  correct face flux and cell centre velocities




