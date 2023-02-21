import numpy as np
from LinearSystem import LinearSystem
#from plots.plot_script import convergence, velocity_field_quiver_plot, field_plot

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
                uface[i] = u[cell] + (PF_mag * u[neighbour]) / PN_mag

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

    def cell_centred_pressure(self, p_field):

        delta_px = np.zeros((self.mesh.num_cells(), 1))
        delta_py = np.zeros((self.mesh.num_cells(), 1))
        owner_neighbours = self.mesh.cell_owner_neighbour()
        face_area_vectors = self.mesh.face_area_vectors()

        for i in range(self.mesh.num_cells()):

            faces = self.mesh.cells[i]
            faces_x = []
            faces_y = []

            for face in faces:
                if face_area_vectors[face][0] != 0:
                    faces_x.append(face)
                elif face_area_vectors[face][1] != 0:
                    faces_y.append(face)
            
            owner_neighbour_x = owner_neighbours[faces_x].flatten()
            owner_neighbour_x = owner_neighbour_x[owner_neighbour_x != i]
            owner_neighbour_y = owner_neighbours[faces_y].flatten()
            owner_neighbour_y = owner_neighbour_y[owner_neighbour_y != i]

            if -1 in owner_neighbour_x:
                delta_px[i] = p_field[owner_neighbour_x[owner_neighbour_x != -1]] - p_field[i]
            else:
                delta_px[i] = p_field[owner_neighbour_x[0]] - p_field[owner_neighbour_x[1]]
            if -1 in owner_neighbour_y:
                delta_py[i] = p_field[owner_neighbour_y[owner_neighbour_y != -1]] - p_field[i]
            else:
                delta_py[i] = p_field[owner_neighbour_y[0]] - p_field[owner_neighbour_y[1]]
        
        return delta_px, delta_py
    
    def face_ap(self, A):

        ap_face = np.zeros((self.mesh.num_faces(), 1))
        owner_neighbour = self.mesh.cell_owner_neighbour()

        for i in range(self.mesh.num_cells()):
            for j in self.mesh.neighbouring_cells()[i]:
                neighbour_faces = self.mesh.cells[j]
                shared_face = list(set(self.mesh.cells[i]).intersection(neighbour_faces))[0]
                ap_face[shared_face] = (A[i, i] + A[j, j])/2

        for i in range(len(owner_neighbour)):
            if owner_neighbour[i][1] == -1:
                diag_ind = owner_neighbour[i][0]
                ap_face[i] = A[diag_ind, diag_ind] 
        
        return ap_face
    
    def face_flux_correction(self, Fpre, A, delta_p):

        ap_face = self.face_ap(A)
        face_area_vectors = self.mesh.face_area_vectors()

        Fcorr = np.zeros((self.mesh.num_faces(), 1))

        for i in range(self.mesh.num_faces()):
            face_mag = np.linalg.norm(face_area_vectors[i])
            Fcorr[i] = Fpre[i] - (1/ap_face[i]) * face_mag * delta_p[i]

        return Fcorr
    
    def cell_centre_correction(self, Ax, Ay, u, v, p_field_UR):

        delta_px, delta_py = self.cell_centred_pressure(p_field_UR)

        ucorr = np.zeros((self.mesh.num_cells(), 1))
        vcorr = np.zeros((self.mesh.num_cells(), 1))

        for i in range(self.mesh.num_cells()):
            ucorr[i] = u[i] - delta_px[i]/Ax[i, i]
            vcorr[i] = v[i] - delta_py[i]/Ay[i, i]

        return ucorr, vcorr
    
    def face_flux_check(self, F):

        owner_neighbour = self.mesh.cell_owner_neighbour()
        total_flux = np.zeros((self.mesh.num_cells(), 1))

        for i in range(len(owner_neighbour)):
            cell = owner_neighbour[i][0]
            neighbour = owner_neighbour[i][1]

            total_flux[cell] += F[i]
            total_flux[neighbour] -= F[i]

        return total_flux
        
    def SIMPLE_loop(self, u, v, F, p, it):

        uface = self.face_velocity(u, 1)
        vface = self.face_velocity(v, 0)
        zface = np.zeros_like(vface)
        Ax, bx = self.momentum_disc(u, F, 1)
        Ay, by = self.momentum_disc(v, F, 0)
        print(Ax)
        print(bx)

        uplus1, res, resRel = self.gauss_seidel(Ax, bx, u)
        vplus1, res, resRel = self.gauss_seidel(Ay, by, v)
        print(uplus1)
        print(vplus1)
        
        if it == 0:
            uface_plus1 = self.face_velocity(uplus1, 1)
            vface_plus1 = self.face_velocity(vplus1, 0)
            Fpre = self.face_flux(uface_plus1, vface_plus1, zface)
        else:
            Fpre = F

        Ap, bp = self.pressure_laplacian(Fpre, Ax)

        p_field, res, resRel = self.gauss_seidel(Ap, bp, p)

        delta_p = self.face_pressure(p_field)

        Fcorr = self.face_flux_correction(Fpre, Ax, delta_p)

        total_flux = self.face_flux_check(Fcorr)
        # print(total_flux)

        p_field_UR = p + self.alpha_p * (p_field - p)

        ucorr, vcorr = self.cell_centre_correction(Ax, Ay, u, v, p_field_UR)

        # print(ucorr)

        res = np.linalg.norm(bx - np.matmul(Ax, ucorr))

        return ucorr, vcorr, Fcorr, p, res
    
    def initial_residual(self, u, F):

        A, b = self.momentum_disc(u, F, 0)
        res_initial = np.sum(b - np.matmul(A, u))
        if res_initial == 0:
            resRel = 0
        else:
            resRel = res_initial / res_initial

        return res_initial, resRel

    def iterate(self, u, v, p, tol=1e-10, maxIts=10):
        
        uface = self.face_velocity(u, 1)
        vface = self.face_velocity(v, 0)
        zface = vface
        F = self.face_flux(uface, vface, zface)

        res_initial, resRel = self.initial_residual(u, F)
        res_ls = [res_initial]
        resRel_ls = [resRel]

        it = 0

        for i in range(maxIts):
            u, v, F, p, res = self.SIMPLE_loop(u, v, F, p, it)
            it += 1
            if res_initial == 0:
                resRel = 0
            else:
                resRel = res / res_initial
            res_ls.append(res)
            resRel_ls.append(resRel)
            break
            if res < tol:
                print(f"Simulation converged in {it} iterations")
                break

        # print(u)
        # print(v)
        # print(p)
        # print(F)

