import numpy as np
from LinearSystem import LinearSystem

class SIMPLE(LinearSystem):

    def __init__(self, mesh, viscosity, alpha_u, alpha_p):
        
        LinearSystem.__init__(self, mesh, viscosity, alpha_u)
        self.alpha_u = alpha_u
        self.alpha_p = alpha_p

    def initial_pressure(self):

        return np.zeros((self.mesh.num_cells(), 1))
    
    def face_flux(self, u):

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
            uface[int(i)] = 1
        
        return uface
    
    def face_pressure(self, p_field):

        delta_p_face = np.zeros((self.mesh.num_faces(), 1))
        owner_neighbour = self.mesh.cell_owner_neighbour()

        for i in range(self.mesh.num_cells()):
            for j in self.mesh.neighbouring_cells()[i]:
                neighbour_faces = self.mesh.cells[j]
                shared_face = list(set(self.mesh.cells[i]).intersection(neighbour_faces))[0]
                delta_p_face[shared_face] = (p_field[j] + p_field[i])

        for i in range(len(owner_neighbour)):
            if owner_neighbour[i][1] == -1:
                delta_p_face[i] = p_field[owner_neighbour[i][0]] - p_field[owner_neighbour[i][0]]
        
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
    
    def pressure_laplacian(self, Fpre, Au):
        
        N = len(self.mesh.cells)
        Ap = np.zeros((N, N))
        bp = np.zeros((N, 1))

        for i in range(len(self.mesh.cells)):

            neighbours = self.mesh.neighbouring_cells()[i]
            face_area_vectors = self.mesh.face_area_vectors()
            ap = Au[i, i]

            cell_faces = self.mesh.cells[i]
            centre_P = self.mesh.cell_centres()[i]
            cell_owner_neighbour = self.mesh.cell_owner_neighbour()

            for face in cell_faces:
                face_owner_neighbour = cell_owner_neighbour[face]
                if face_owner_neighbour[1] == -1:
                    sf = face_area_vectors[face]
                    face_mag = np.linalg.norm(sf)
                    Ap[i, i] += -self.viscosity * face_mag / (0.005 * ap)
                    bp[i] += Fpre[i]

            for j in neighbours:

                # get faces in neighbour cell
                neighbour_faces = self.mesh.cells[j]
                # get the shared faces between the two cells
                shared_face = list(set(cell_faces).intersection(neighbour_faces))[0]
                # get the owner of the face
                owner_neighbour = cell_owner_neighbour[shared_face]
                # get centre of the neighbour cell
                centre_N = self.mesh.cell_centres()[j]

                # if cell is the owner of the face
                if owner_neighbour[0] == i:
                    sf = face_area_vectors[shared_face]
                    bp[i] += Fpre[i]
                else:
                    sf = -face_area_vectors[shared_face]
                    bp[i] += -Fpre[i]

                face_mag = np.linalg.norm(sf)

                d = abs(centre_P - centre_N)
                d_mag = np.linalg.norm(d)

                # diffusive contributions
                Ap[i, i] += -self.viscosity * face_mag / (d_mag * ap)
                Ap[i, j] += self.viscosity * face_mag / (d_mag * ap)

        return Ap, bp
    
    def face_flux_correction(self, Fpre, A, delta_p):

        delta_p_face = self.face_pressure(delta_p)
        ap_face = self.face_ap(A)
        face_area_vectors = self.mesh.face_area_vectors()

        Fcorr = np.zeros((self.mesh.num_faces(), 1))

        for i in range(self.mesh.num_faces()):
            face_mag = np.linalg.norm(face_area_vectors[i])
            Fcorr[i] = Fpre[i] - (1/ap_face[i]) * face_mag * delta_p_face[i]

        return Fcorr
    
    def cell_centre_correction(self, u, A, delta_p):

        ucorr = np.zeros((self.mesh.num_cells(), 1))

        for i in range(self.mesh.num_cells()):
            ucorr[i] = u[i] - delta_p[i]/A[i, i]

        return ucorr

    def SIMPLE_loop(self, u, v, uface, vface, p):

        Ax, bx = self.momentum_disc(u, uface)
        Ay, by = self.momentum_disc(v, vface)

        uplus1 = self.gauss_seidel(Ax, bx, u)
        vplus1 = self.gauss_seidel(Ay, by, v)

        uFpre = self.face_flux(uplus1)
        vFpre = self.face_flux(vplus1)

        Ap, bp = self.pressure_laplacian(uFpre, Ax)
        
        p_field = self.gauss_seidel(Ap, bp, p)

        uFcorr = self.face_flux_correction(uFpre, Ax, p_field)
        vFcorr = self.face_flux_correction(vFpre, Ay, p_field)

        p_field_UR = p + self.alpha_p * (p_field - p)
        print(p_field.shape, p.shape)

        ucorr = self.cell_centre_correction(u, Ax, p_field_UR)
        vcorr = self.cell_centre_correction(v, Ay, p_field_UR)

        res = np.linalg.norm(bx - np.matmul(Ax, ucorr))

        return ucorr, vcorr, uFcorr, vFcorr, p, res
    
    def iterate(self, u, v, p, tol=1e-6):
        
        u_current = u
        v_current = v
        p_current = p
        uface_current = self.face_flux(u_current)
        vface_current = self.face_flux(v_current)

        res_initial = 1
        resRel = res_initial / res_initial

        for i in range(100):

            u_updated, v_updated, uface_updated, vface_updated, p_updated, res = self.SIMPLE_loop(u_current, v_current, uface_current, vface_current, p_current)

            u_current = u_updated
            v_current = v_updated
            uface_current = uface_updated
            vface_current = vface_updated
            p_current = p_updated

            resRel = res / res_initial
            break

        print(u_current)



