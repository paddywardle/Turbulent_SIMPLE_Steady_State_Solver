import numpy as np
from LinearSystem import LinearSystem

class SIMPLE(LinearSystem):

    def __init__(self, mesh, viscosity, alpha_u, alpha_p):
        
        LinearSystem.__init__(self, mesh, viscosity, alpha_u)
        self.alpha_u = alpha_u
        self.alpha_p = alpha_p
    
    def face_velocity(self, u, BC):

        uface = np.zeros((self.mesh.num_faces(), 1))

        owner_neighbours = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()

        top_index = []

        for i in range(len(self.mesh.boundary_patches)):
            if self.mesh.boundary_patches[i][0] == "movingWall":
                top_index = self.mesh.boundary_patches[i+1]

        top_index = [int(i) for i in top_index]

        for i in range(len(owner_neighbours)):

            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]
            
            if (neighbour == -1):
                if (i in top_index):
                    uface[i] = BC
            else:
                PF_mag = np.linalg.norm(face_centres[i] - cell_centres[cell])
                PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[cell])
                uface[i] = u[cell] + (PF_mag * (u[neighbour]-u[cell])) / PN_mag

        return uface
    
    def face_flux(self, uface, vface, zface):

        F = []
        face_area_vectors = np.squeeze(self.mesh.face_area_vectors())
        
        face_velocity = np.squeeze(np.hstack((uface, vface, zface)))

        for i in range(len(face_velocity)):
            F_current = np.dot(face_area_vectors[i], face_velocity[i])
            F.append(F_current)

        F = np.asarray(F)

        return F
    
    def face_pressure(self, p_field):

        delta_p_face = np.zeros((self.mesh.num_faces(), 1))
        owner_neighbour = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()
        cell_centres = self.mesh.cell_centres()

        for i in range(len(owner_neighbour)):

            cell = owner_neighbour[i][0]
            neighbour = owner_neighbour[i][1]

            if neighbour == -1:
                delta_p_face[i] = 0
                continue
            cell_centre = cell_centres[cell]
            neighbour_centre = cell_centres[neighbour]
            face_mag = np.linalg.norm(face_area_vectors[i])
            d_mag = np.linalg.norm(cell_centre - neighbour_centre)
            delta_p_face[i] = ((p_field[neighbour] - p_field[cell]) / d_mag) * face_mag
        
        return delta_p_face
    
    def face_ap(self, A):

        ap_face = np.zeros((self.mesh.num_faces(), 1))
        owner_neighbour = self.mesh.cell_owner_neighbour()
        face_centres = self.mesh.face_centres()
        cell_centres = self.mesh.cell_centres()

        for i in range(len(owner_neighbour)):
            cell = owner_neighbour[i][0]
            neighbour = owner_neighbour[i][1]

            if neighbour == -1:
                ap_face[i] = A[cell, cell]
            else:
                PF_mag = np.linalg.norm(face_centres[i] - cell_centres[cell])
                PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[cell])
                ap_face[i] = A[cell, cell] + (PF_mag * (A[neighbour, neighbour]-A[cell, cell])) / PN_mag
        
        return ap_face
    
    def face_flux_correction(self, F, A, p_field):

        owner_neighbours = self.mesh.cell_owner_neighbour()
        face_centres = self.mesh.face_centres()
        cell_centres = self.mesh.cell_centres()
        face_area_vectors = self.mesh.face_area_vectors()
        ap_face = self.face_ap(A)

        for i in range(len(owner_neighbours)):
            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]
            face_area_vector = face_area_vectors[i]
            face_mag = np.linalg.norm(face_area_vector)
            ap = A[cell,cell]#ap_face[i]

            if neighbour == -1:
                d_mag = np.linalg.norm(cell_centres[cell] - face_centres[i])
                aPN = ap * face_mag / d_mag
                # zero gradient boundary condition
                F[i] -= aPN * 0
                continue
            
            d_mag = np.linalg.norm(cell_centres[cell] - cell_centres[neighbour])
            aPN = ap * face_mag / d_mag

            F[i] -= aPN * (p_field[neighbour] - p_field[cell])

        return F

    # def pressure(self, p_field):

    #     face_area_vectors = self.mesh.face_area_vectors()
    #     owner_neighbours = self.mesh.cell_owner_neighbour()

    #     delta_px = np.zeros_like(p_field, dtype=float)
    #     delta_py = np.zeros_like(p_field, dtype=float)

    #     for i in range(len(owner_neighbours)):

    #         cell = owner_neighbours[i][0]
    #         neighbour = owner_neighbours[i][1]
    #         face_area_vector = face_area_vectors[i]

    #         for j in range(len(face_area_vector)):
                


    
    def cell_centre_correction(self, Ax, Ay, u, v, p_field):

        owner_neighbours = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()

        for i in range(len(owner_neighbours)):

            cell = owner_neighbours[i][0]
            neighbour = owner_neighbours[i][1]

            if neighbour == -1:
                # neumann 0 grad boundary conditions so skip
                continue
            else:
                d_mag = np.linalg.norm(cell_centres[cell] - cell_centres[neighbour])

                u[cell] -= ((p_field[neighbour]-p_field[cell]) / d_mag) / Ax[cell, cell]
                # u[neighbour] -= (p_field[cell]-p_field[neighbour]) / Ax[cell, cell]
                v[cell] -= ((p_field[neighbour]-p_field[cell]) / d_mag) / Ay[cell, cell]
                # v[neighbour] -= (p_field[cell]-p_field[neighbour]) / Ay[cell, cell]

        return u, v
    
    def face_flux_check(self, F):

        owner_neighbour = self.mesh.cell_owner_neighbour()
        total_flux = np.zeros((self.mesh.num_cells(), 1))

        for i in range(len(owner_neighbour)):
            cell = owner_neighbour[i][0]
            neighbour = owner_neighbour[i][1]

            total_flux[cell] += F[i]
            total_flux[neighbour] -= F[i]

        return total_flux
    
    def residual(self, Ax, bx, Ay, by, u, v):

        SIMPLE_res_x = np.linalg.norm(bx - np.matmul(Ax, u))
        SIMPLE_res_y = np.linalg.norm(by - np.matmul(Ay, v))
        SIMPLE_res = np.linalg.norm([SIMPLE_res_x, SIMPLE_res_y])

        return SIMPLE_res
    
    def initial_residual(self, u, F):

        A, b = self.momentum_disc(u, F, 0)
        res_initial = np.sum(b - np.matmul(A, u))
        if res_initial == 0:
            resRel = 0
        else:
            resRel = res_initial / res_initial

        return res_initial, resRel
        
    def SIMPLE_loop(self, u, v, F, p, it):

        Ax, bx = self.momentum_disc(u, F, 1)
        Ay, by = self.momentum_disc(v, F, 0)

        uplus1, GS_res_x = self.gauss_seidel(Ax, bx, u)
        vplus1, GS_res_y = self.gauss_seidel(Ay, by, v)

        SIMPLE_res = self.residual(Ax, bx, Ay, by, uplus1, vplus1)

        uface_plus1 = self.face_velocity(uplus1, 1)
        vface_plus1 = self.face_velocity(vplus1, 0)
        zface = np.zeros_like(vface_plus1)
        Fpre = self.face_flux(uface_plus1, vface_plus1, zface)

        Ap, bp = self.pressure_laplacian(Fpre, Ax, 0)

        p_field = np.linalg.solve(Ap, bp).flatten() #self.gauss_seidel(Ap, bp, p)
    
        Fcorr = self.face_flux_correction(Fpre, Ax, p_field)
        total_flux = self.face_flux_check(Fcorr)

        p_field_UR = p + self.alpha_p * (p_field - p)

        ucorr, vcorr = self.cell_centre_correction(Ax, Ay, u, v, p_field_UR)

        return ucorr, vcorr, Fcorr, p_field_UR, SIMPLE_res, GS_res_x, GS_res_y

    def iterate(self, u, v, p, tol=1e-10, maxIts=10):
        
        uface = self.face_velocity(u, 1)
        vface = self.face_velocity(v, 0)
        zface = np.zeros_like(vface)
        F = self.face_flux(uface, vface, zface)

        res_SIMPLE_ls = []
        res_SIMPLE_y_ls = []
        resX_GS_ls = []
        resY_GS_ls = []

        it = 0

        for i in range(maxIts):
            print("Iteration: " + str(it+1))
            u, v, F, p, SIMPLE_res, resX_GS, resY_GS = self.SIMPLE_loop(u, v, F, p, it)
            it += 1
            res_SIMPLE_ls.append(SIMPLE_res)
            resX_GS_ls.append(resX_GS)
            resY_GS_ls.append(resY_GS)
            break
            if SIMPLE_res < tol:
                print(f"Simulation converged in {it} iterations")
                break
        
        return u, v, p, res_SIMPLE_ls