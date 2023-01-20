from typing import List
import numpy as np
from scipy.spatial import ConvexHull


class Mesh:

    def __init__(self, points, faces, cells, boundary_patches):

        self.points = points
        self.faces = faces
        self.cells = cells
        self.boundary_patches = boundary_patches

    def __repr__(self):

        """
        Dunder method for the unambiguous representation of a mesh instance.

        Returns:
            str: representation of mesh instance
        """

        return f"Mesh(points={self.points},\n faces={self.faces},\n cells={self.cells},\n boundary_patches={self.boundary_patches})"

    def __str__(self):

        """
        Dunder method for the readable string representation of a mesh instance.

        Returns:
            str: string representation of mesh instance
        """

        return f"""Mesh Class with: {self.num_points()} points, {self.num_faces()} faces, {self.num_cells()} cells, {self.num_boundary_patches()} boundary patches"""

    def num_points(self):

        """
        This function returns the number of points in the mesh instance. This is done using the points instance variable.

        Returns:
            int: value representing the number of points in the mesh instance.
        """

        return len(self.points)

    def num_faces(self):

        """
        This function returns the number of faces in the mesh instance. This is done using the faces instance variable.

        Returns:
            int: value representing the number of faces in the mesh instance.
        """

        return len(self.faces)

    def num_cells(self):

        """
        This function returns the number of cells in the mesh instance. This is done using the cells instance variable.

        Returns:
            int: value representing the number of cells in the mesh instance.
        """

        return len(self.cells)

    def num_boundary_patches(self):

        """
        This function returns the number of boundary patches in the mesh instance. This is done using the boundary_patches instance variable.

        Returns:
            int: value representing the number of boundary patches in the mesh instance.
        """

        return len(self.boundary_patches)

    def cell_volumes(self):

        """
        This function returns a numpy array of the volumes of each cell in the mesh.

        Returns:
            np.array: array containing the volume of each cell in the mesh
        """
        # should I calculate cell and face centre lists using other functions and then use these values in the loop rather than calculating each time?
        # empty array for cell volumes to be added to
        cell_vols = np.array([])

        for cell in self.cells:
            # get unique faces that make up the cell
            cell_points_num = np.unique(np.concatenate(self.faces[cell]))
            cell_points = self.points[cell_points_num]
            cell_centre = np.divide(cell_points.sum(axis=0), len(cell_points))
            cell_vol = 0
            for face in cell:
                # get points that make up the face
                face_points = self.points[self.faces[face]]
                face_centre = np.divide(face_points.sum(axis=0), len(face_points))
                for i in range(len(face_points)):
                    if i == (len(face_points)-1):
                        vertices = np.concatenate(([face_points[i]], [face_points[0]], [face_centre], [cell_centre]), axis=0)
                        coord_matrix = np.column_stack((vertices, np.ones((len(vertices), 1))))
                        cell_vol += abs(np.linalg.det(coord_matrix) / 6)
                        continue
                    vertices = np.concatenate(([face_points[i]], [face_points[i+1]], [face_centre], [cell_centre]), axis=0)
                    coord_matrix = np.column_stack((vertices, np.ones((len(vertices), 1))))
                    cell_vol += abs(np.linalg.det(coord_matrix) / 6)
            cell_vols = np.append(cell_vols, cell_vol)
        
        return cell_vols

    def convex_hull_volume(self):

        # empty array for cell volumes to be added to
        cell_vols = np.array([])

        for cell in self.cells:
            # get unique faces that make up the cell
            cell_faces = np.unique(np.concatenate(self.faces[cell]))
            cell_points = self.points[cell_faces]
            cell_vol = ConvexHull(cell_points).volume
            cell_vols = np.append(cell_vols, cell_vol)
        
        return cell_vols

    def face_area_vectors(self):

        """
        This function returns a numpy array of the face area vector for each face in the mesh instance.

        Returns:
            np.array: array containing the face area vector of each cell in the mesh
        """

        # empty array for cell face area vectors
        face_area_vecs = []

        for face in self.faces:
            # get points that make up face
            face_points = self.points[face]

            # sum up points and divide by number of points, to get face centre
            points_sums = face_points.sum(axis=0)
            face_cen = np.divide(points_sums, len(face_points))

            face_area_vec = 0

            for i in range(len(face_points)):
                if i == (len(face_points)-1):
                    face_area_vec += np.cross(face_points[i]-face_cen, face_points[0]-face_cen) / 2
                    continue
                face_area_vec += np.cross(face_points[i]-face_cen, face_points[i+1]-face_cen) / 2
            
            face_area_vecs.append(face_area_vec)

        return np.asarray(face_area_vecs)

    def cell_centres(self):

        """
        This function returns a numpy array of the cell centre for each cell in the mesh instance.

        Returns:
            np.array: array containing the cell centres of each cell in the mesh
        """

        # empty list for cell centre coordinates
        cell_cens = []

        for cell in self.cells:
            # get unique faces that make up the cell
            cell_faces = np.unique(np.concatenate(self.faces[cell]))
            cell_points = self.points[cell_faces]

            # sum up cell coordinates and divide by number of points, to get cell centre
            points_sums = cell_points.sum(axis=0)
            cell_cen = np.divide(points_sums, len(cell_points))
            cell_cens.append(cell_cen)

        cell_cens = np.asarray(cell_cens)

        return cell_cens

    def face_centres(self):

        """
        This function returns a numpy array of the face centre for each face in the mesh instance.

        Returns:
            np.array: array containing the face centres of each face in the mesh
        """

        # empty list for face centre coordinates
        face_cens = []

        for face in self.faces:
            # get points that make up the face
            face_points = self.points[face]

            # sum up points and divide by number of points, to get face centre
            points_sums = face_points.sum(axis=0)
            face_cen = np.divide(points_sums, len(face_points))
            face_cens.append(face_cen)

        face_cens = np.asarray(face_cens)

        return face_cens

    def boundary_face_cells(self):

        """
        This function returns the cells that boundary faces are connected to on a per-patch basis.

        Returns:
            np.array: array containing cells that each boundary patch is connected to
        """
 
        boundary_face_cells = []

        for boundary_patch in self.boundary_patches:
            per_patch = []
            # skipping patch types (testing for values that are strings)
            if type(boundary_patch) == str:
                boundary_face_cells.append(boundary_patch)
                continue
            # looping through faces in boundary patches
            for boundary_face in boundary_patch:
                # getting cells that contain current boundary face
                per_patch += [i for i in range(len(self.cells)) if boundary_face in self.cells[i]]
            # appending unique faces to list (set gets unique values)
            boundary_face_cells.append(set(per_patch))

        return boundary_face_cells

    def neighbouring_cells(self):

        """
        This function returns an array with lists referring to the cells each cell is connected to.

        Returns:
            np.array: array containing lists of cell neighbours
        """

        neighbours = []


        for i in range(len(self.cells)):
            current_neighbours = []
            # looping through cells again
            for j in range(len(self.cells)):
                # skipping as don't want to compare cell to itself
                if i == j:
                    continue
                # test to see if they share faces - if they do append cell number to current_neighbour list
                if len(set(self.cells[i]) & set(self.cells[j])) != 0:
                    current_neighbours.append(j)
            # append current_neighbour to overall neighbours list corresponding to cells label
            neighbours.append(current_neighbours)

        return np.asarray(neighbours)

    def cell_owner_neighbour(self):

        """
        This function returns an array with lists referring to the owner and neighbour of each face of the mesh.

        Returns:
            np.array: array containing face owner and neighbours, where neighbour = -1 for boundary faces
        """

        # lists to store owner and neighbour values
        owner = []
        neighbour = []

        # looping through faces
        for face in range(len(self.faces)):
            # search for indices of face in cell array (they'll be 2 for internal faces, 1 for boundary faces)
            cells_with_face = np.where(self.cells == face)
            # testing for boundary face
            if len(cells_with_face[0]) == 1:
                # appending cell number to owner list
                owner.append(cells_with_face[0][0])
                # appending -1 to neighbour list as it is a boundary face
                neighbour.append(-1)
                continue
            # appending lowest cell label to owner list (convention)
            owner.append(cells_with_face[0][0])
            # appending highest cell label to neighbour list (convention)
            neighbour.append(cells_with_face[0][1])
        
        return np.column_stack((owner, neighbour))
