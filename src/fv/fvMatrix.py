import numpy as np
import sys

class fvMatrix():

    """
    Class to hold methods to apply to finite volume matrices.
    """

    def __init__(self, mesh):

        self.mesh = mesh

    def raP(self, A):

        """
        Function to calculate reciprocal of momentum diagonal.

        Args:
            A (np.array): momentum matrix
        Returns:
            np.array: array of reciprocals
        """
        
        raP = []
        
        V = self.mesh.cell_volumes()

        for i in range(len(A)):

            raP.append(1/A[i, i])

        raP *= V

        return np.array(raP) 
    
    def H(self, A, b, u):

        """
        Function to calculate H operator

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum source
            u (np.array): velocity field
        Returns:
            H (np.array): H operator
        """

        H = b.copy()
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        V = self.mesh.cell_volumes()

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                continue
            
            H[owner] -= A[owner, neighbour] * u[neighbour]
            H[neighbour] -= A[neighbour, owner] * u[owner]

        H /= V
            
        return H
    
    def HbyA(self, A, b, u, raP):

        """
        Function to calculate HbyA operator to enforce divergence free velocity

        Args:
            A (np.array): momentum matrix
            b (np.array): momentum source
            u (np.array): velocity field
            raP (np.array): reciprocal of diagonal coefficients
        Returns:
            HbyA (np.array): HbyA operator
        """

        HbyA = self.H(A, b, u)

        HbyA *= raP

        return HbyA


    def face_flux(self, u, v, z, BC):

        """
        Function to calculate face flux

        Args:
            u (np.array): x velocity field
            v (np.array): y velocity field
            z (np.array): z velocity field
        Returns:
            F (np.array): face flux field

        """

        noComponents = 3
        F = np.zeros((self.mesh.num_faces(),))

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        face_area_vectors = self.mesh.face_area_vectors()

        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            for cmpt in range(noComponents):
                
                if (neighbour == -1):
                    if i in self.mesh.boundaries['inlet']:
                        uface = BC['inlet'][cmpt]
                    elif i in self.mesh.boundaries['outlet']:
                        uface = u[cmpt]
                    elif i in self.mesh.boundaries['upperWall']:
                        uface = BC['upperWall'][cmpt]
                    elif i in self.mesh.boundaries['lowerWall']:
                        uface = BC['lowerWall'][cmpt]
                    else:
                        uface = BC['frontAndBack'][cmpt]
                else:
                    fN_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
                    PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[owner])
                    fx = fN_mag / PN_mag;
                    uface = fx * u[owner] + (1 - fx) * u[neighbour]

                F[i] += uface * face_area_vectors[i][cmpt]

        return F
    
    def face_pressure(self, p_field, Ap, raP, BC):

        """
        Function to calculate face pressure gradient.

        Args:
            p_field (np.array): pressure field
        Returns:
            delta_p_face (np.array): face pressure gradient

        """

        delta_p_face = np.zeros((self.mesh.num_faces(),))
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        cell_centres = self.mesh.cell_centres()
        face_centres = self.mesh.face_centres()
        raP_face = self.face_raP(raP)
        face_area_vectors = self.mesh.face_area_vectors()

        # loops through owner neighbour pairs
        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):
            face_area_vector = face_area_vectors[i]
            face_mag = np.linalg.norm(face_area_vector)

            # zero gradient boundary condition
            if neighbour == -1:
                d_mag = np.linalg.norm(cell_centres[owner] - face_centres[i])
                if i in self.mesh.boundaries['outlet']:
                    aPN = raP_face[i] * (face_mag / d_mag)
                    delta_p_face[i] += aPN * (BC['outlet'][3] - p_field[owner])
                else:
                    delta_p_face[i] += 0
                continue

            #delta_p_face[i] = (p_field[neighbour] - p_field[cell])
            delta_p_face[i] += Ap[owner, neighbour] * p_field[neighbour] - Ap[neighbour, owner] * p_field[owner]
        
        return delta_p_face
    
    def face_raP(self, raP):

        """
        Function to calculate face value of momentum coefficients

        Args:
            raP (np.array): reciprocal of diagonal matrix coefficients
        Returns:
            raP_face (np.array): reciprocal face diagonal momentum values

        """

        raP_face = np.zeros((self.mesh.num_faces(),))
        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        face_centres = self.mesh.face_centres()
        cell_centres = self.mesh.cell_centres()

        # loops through owner neighbour pairs and linearly interpolates ap onto the face
        for i, (owner, neighbour) in enumerate(cell_owner_neighbour):

            if neighbour == -1:
                # zero gradient Neumann
                raP_face[i] = raP[owner]
                continue
            
            fN_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
            PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[owner])
            fx = fN_mag / PN_mag;
            raP_face[i] = fx * raP[owner] + (1 - fx) * raP[neighbour]
        
        return raP_face

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

            fN_mag = np.linalg.norm(face_centres[i] - cell_centres[neighbour])
            PN_mag = np.linalg.norm(cell_centres[neighbour] - cell_centres[owner])
            fx = fN_mag / PN_mag;

            p_face = fx * p_field[owner] + (1 - fx) * p_field[neighbour]

            cmptGrad = 0
            for cmptSf in range(3):
                p_grad[cmptGrad][owner] += p_face * face_area_vectors[i][cmptSf]

                p_grad[cmptGrad][neighbour] -= p_face * face_area_vectors[i][cmptSf]

                cmptGrad += 1

        for cmpt in range(3):
            p_grad[cmpt] /= V

        return p_grad
    
    def face_flux_check(self, F):

        """
        Function to check total flux for each cell

        Args:
            F (np.array): Face fluxes
        Returns:
            total_flux (np.array): total flux for each cell
        """

        cell_owner_neighbour = self.mesh.cell_owner_neighbour()
        total_flux = np.zeros((self.mesh.num_cells(), 1))

        # loops through owner neighbour pairs and adds fluxes to owners and neighbours - skips neighbour if boundary
        for i, (cell, neighbour) in enumerate(cell_owner_neighbour):
            
            total_flux[cell] += F[i]

            if neighbour == -1:
                continue
            total_flux[neighbour] -= F[i]

        return total_flux.flatten()
