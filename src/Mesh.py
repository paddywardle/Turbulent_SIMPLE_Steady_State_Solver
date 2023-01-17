from typing import List
import numpy as np

class Mesh:

    def __init__(self, points, faces, cells, boundary_patches):
        self.points = points
        self.faces = faces
        self.cells = cells
        self.boundary_patches = boundary_patches

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

        pass

    def cell_centres(self):

        pass

    def face_centres(self):

        pass

    def boundary_face_cells(self):

        pass

    def neighbouring_cells(self):

        pass

    def cell_owner_neighbour(self):

        pass

