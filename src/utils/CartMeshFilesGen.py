import numpy as np
from utils.ReadJSON import ReadJSON

def mesh_files(x0, x1, y0, y1, z0, z1, nxCells, nyCells, nzCells):

    points = get_points(x0, x1, y0, y1, z0, z1, nxCells, nyCells, nzCells)

    faces = get_faces(nxCells, nyCells, nzCells)

    write_file(points, "MeshFiles/points.txt")
    write_file(faces, "MeshFiles/faces.txt")

def get_points(x0, x1, y0, y1, z0, z1, nxCells, nyCells, nzCells):

    points = []

    x_vals = []
    y_vals = []
    z_vals = [z0, z1]
     
    dx = (x1 - x0) / nxCells
    dy = (y1 - y0) / nyCells
    dz = (z1 - z0)

    x = 0
    y = 0

    for i in range(nxCells+1):
        x = round(x, 4)
        x_vals.append(x)
        x += dx

    for i in range(nyCells+1):
        y = round(y, 4)
        y_vals.append(y)
        y += dy

    for k in range(len(z_vals)):
        for j in range(len(y_vals)):
            for i in range(len(x_vals)):
                points.append([x_vals[i], y_vals[j], z_vals[k]])

    return np.array(points)
     
def get_faces(nxCells, nyCells, nzCells):

    faces = []
    faces_back_front = []
    faces_vertical = []
    faces_horizontal = []

    for k in range(nzCells+1):
        face_zlen = len(faces_back_front)
        for j in range(nyCells):
            face_ylen = len(faces_back_front)
            for i in range(nxCells):
                    start_idx = i+face_ylen+face_zlen+j+k
                    face = [start_idx, start_idx+1, start_idx+1+nxCells, start_idx+2+nxCells]
                    faces_back_front.append(face)
                    faces.append(face)

    for j in range(nyCells):
        face_ylen = len(faces_vertical)
        for i in range(nxCells+1):
                vertical_start = face_ylen+i
                bottom_idx = (nxCells+1) * (nyCells+1)
                face_vertical = [vertical_start, vertical_start+1+nxCells, vertical_start+bottom_idx, vertical_start+bottom_idx+nxCells+1]
                faces_vertical.append(face_vertical)
                if (i != 0) and (i != nxCells):
                    faces.insert(0, face_vertical)
                else:
                    faces.append(face_vertical)

    for j in range(nyCells+1):
        face_ylen = len(faces_horizontal)
        for i in range(nxCells):
                horizontal_start = face_ylen+j+i
                bottom_idx = (nxCells+1) * (nyCells+1)
                face_horizontal = [horizontal_start, horizontal_start+1, horizontal_start+bottom_idx, horizontal_start+bottom_idx+1]
                faces_horizontal.append(face_horizontal)
                if (j != 0) and (j != nyCells):
                    faces.insert(0, face_horizontal)
                else:
                    faces.append(face_horizontal)

    return np.array(faces)

def get_cells(faces):

    pass

def write_file(points, filename):

    with open(filename, "w") as file:
        for point in points:
            line = "("
            for i in range(len(point)):
                if (i == len(point)-1):
                    line += f"{point[i]})\n"
                else:
                    line += f"{point[i]},"
            file.writelines(line)

if __name__ == "__main__":

    mesh_settings = ReadJSON('config/config.json')['MESH']

    mesh_files(mesh_settings['x0'], mesh_settings['x1'], mesh_settings['y0'], mesh_settings['y1'], mesh_settings['z0'], 
               mesh_settings['z1'], mesh_settings['nxCells'], mesh_settings['nyCells'], mesh_settings['nzCells'])