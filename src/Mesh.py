from typing import List
import numpy as np

class Mesh:

    def __init__(self, points, faces, cells, boundary_patches):

        self.points = points
        self.faces = faces
        self.cells = cells
        self.boundary_patches = boundary_patches

    def __repr__(self):

        return f"Mesh(points={self.points},\n faces={self.faces},\n cells={self.cells},\n boundary_patches={self.boundary_patches})"

    def __str__(self):

        return f"""Mesh Class with: {self.num_points()} points, {self.num_faces()} faces, {self.num_cells()} cells, {self.num_boundary_patches()} boundary patches"""

    def num_points(self):

        return len(self.points)

    def num_faces(self):

        return len(self.faces)

    def num_cells(self):

        return len(self.cells)

    def num_boundary_patches(self):

        return len(self.boundary_patches)

    def cell_volumes(self):

        cell_vols = np.array([])

        for cell in self.cells:
            cell_faces = np.unique(np.concatenate(self.faces[cell]))
            cell_points = self.points[cell_faces]
            cell_min_points = cell_points.min(axis=0)
            cell_max_points = cell_points.max(axis=0)
            cell_vol = np.prod(cell_max_points - cell_min_points)
            cell_vols = np.append(cell_vols, cell_vol)

        return cell_vols

    def face_area_vectors(self):

        face_area_vecs = np.array([])

        for face in self.faces:
            face_points = self.points[face]
            face_min_points = face_points.min(axis=0)
            face_max_points = face_points.max(axis=0)
            point_diff = face_max_points-face_min_points
            face_area = np.prod(point_diff[point_diff > 0])
            face_area_vecs = np.append(face_area_vecs, face_area)

        return face_area_vecs

    def cell_centres(self):

        cell_cens = []

        for cell in self.cells:
            cell_faces = np.unique(np.concatenate(self.faces[cell]))
            cell_points = self.points[cell_faces]
            points_sums = cell_points.sum(axis=0)
            cell_cen = np.divide(points_sums, len(cell_points))
            cell_cens.append(cell_cen)

        cell_cens = np.asarray(cell_cens)

        return cell_cens

    def face_centres(self):

        face_cens = []

        for face in self.faces:
            face_points = self.points[face]
            points_sums = face_points.sum(axis=0)
            face_cen = np.divide(points_sums, len(face_points))
            face_cens.append(face_cen)

        face_cens = np.asarray(face_cens)

        return face_cens

    def boundary_face_cells(self):

        pass

    def neighbouring_cells(self):

        pass

    def cell_owner_neighbour(self):

        pass

