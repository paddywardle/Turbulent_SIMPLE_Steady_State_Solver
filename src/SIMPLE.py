import numpy as np
from LinearSystem import LinearSystem
from utils.GS_convergence_plot import GS_convergence, velocity_field_quiver_plot, field_plot

class SIMPLE(LinearSystem):

    def __init__(self, mesh, viscosity, alpha_u, alpha_p):
        
        LinearSystem.__init__(self, mesh, viscosity, alpha_u)
        self.alpha_u = alpha_u
        self.alpha_p = alpha_p

    def initial_pressure(self):

        return np.zeros((self.mesh.num_cells(), 1))
    
    def face_flux(self, u, BC):

        uface = np.zeros((self.mesh.num_faces(), 1))

        for i in range(self.mesh.num_cells()):
            for j in self.mesh.neighbouring_cells()[i]:
                neighbour_faces = self.mesh.cells[j]
                shared_face = list(set(self.mesh.cells[i]).intersection(neighbour_faces))[0]
                uface[shared_face] = (u[i] + u[j])/2

        for i in range(len(self.mesh.boundary_patches)):
            if self.mesh.boundary_patches[i][0] == "movingWall":
                top_index = self.mesh.boundary_patches[i+1]

        for i in top_index:
            uface[int(i)] = BC
        
        return uface
    
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
    
    def cell_centre_correction(self, u, A, p_field_UR):

        delta_p = self.face_pressure(p_field_UR)

        ucorr = np.zeros((self.mesh.num_cells(), 1))
        vcorr = np.zeros((self.mesh.num_cells(), 1))

        for i in range(self.mesh.num_cells()):
            ucorr[i] = u[i] - delta_p[i]/A[i, i]

        return ucorr

    def SIMPLE_loop(self, u, v, uface, vface, p):

        Ax, bx = self.momentum_disc(u, uface)
        Ay, by = self.momentum_disc(v, vface)
        # print("Ax")
        # print(Ax)
        # print("bx")
        # print(bx)
        # print("Ay")
        # print(Ay)
        # print("by")
        # print(by)

        uplus1, res, resRel = self.gauss_seidel(Ax, bx, u)
        #GS_convergence(resRel, list(range(len(res))))
        vplus1, res, resRel = self.gauss_seidel(Ay, by, v)

        uFpre = self.face_flux(uplus1, 1)
        vFpre = self.face_flux(vplus1, 0)

        Ap, bp = self.pressure_laplacian(uFpre, Ax)

        # print("pressure A")
        # print(Ap)
        # print("pressure b")
        # print(bp)
        
        p_field, res, resRel = self.gauss_seidel(Ap, bp, p)
        #GS_convergence(resRel, list(range(len(resRel))))
        delta_p = self.face_pressure(p_field)

        uFcorr = self.face_flux_correction(uFpre, Ax, delta_p)
        vFcorr = self.face_flux_correction(vFpre, Ay, delta_p)

        p_field_UR = p + self.alpha_p * (p_field - p)

        # # # am I using the correct bits here? <- should be delta_p?
        ucorr = self.cell_centre_correction(u, Ax, p_field_UR)
        vcorr = self.cell_centre_correction(v, Ay, p_field_UR)

        res = np.linalg.norm(bx - np.matmul(Ax, ucorr))

        return ucorr, vcorr, uFcorr, vFcorr, p, res
    
    def iterate(self, u, v, p, tol=1e-6):
        
        u_current = u
        v_current = v
        p_current = p
        uface_current = self.face_flux(u_current, 1)
        vface_current = self.face_flux(v_current, 0)

        res_initial = 1
        resRel = res_initial / res_initial

        # change to while loop for convergence
        for i in range(5):
            u_updated, v_updated, uface_updated, vface_updated, p_updated, res = self.SIMPLE_loop(u_current, v_current, uface_current, vface_current, p_current)

            u_current = u_updated
            v_current = v_updated
            uface_current = uface_updated
            vface_current = vface_updated
            p_current = p_updated

            resRel = res / res_initial
            break
        print("ux")
        print(u_current)
        print("pressure field")
        print(p_updated)

        # velocity_field_quiver_plot(u_current, v_current, 2, 0.1)

        # field_plot(p_current, 2)

        # print(u_current)