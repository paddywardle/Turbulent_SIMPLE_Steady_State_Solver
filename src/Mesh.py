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

            min_vertex = np.array(self.points[self.faces[cell[0]][0]])

            max_vertex = np.array(self.points[self.faces[cell[0]][0]])
            
            for face_num in cell:

                face = self.faces[face_num]

                for vertex_num in face:

                    vertex = self.points[vertex_num]
                    
                    for i in range(len(vertex)):
                        if vertex[i] < min_vertex[i]:
                            min_vertex[i] = vertex[i]
                        elif vertex[i] > max_vertex[i]:
                            max_vertex[i] = vertex[i]
            print(min_vertex, max_vertex)
            volume = np.prod(max_vertex-min_vertex)

            cell_vols = np.append(cell_vols, volume)

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

