import numpy as np

def mesh_files(x0, x1, y0, y1, z0, z1, nxCells, nyCells, nzCells):

    points = get_points(x0, x1, y0, y1, z0, z1, nxCells, nyCells, nzCells)

    faces = get_faces(nxCells, nyCells, nzCells)

    print(len(faces))

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

    for k in range(nzCells+1):
        face_zlen = len(faces)
        for j in range(nyCells):
            face_ylen = len(faces)
            for i in range(nxCells):
                    start_idx = i+face_ylen+face_zlen+j+k
                    face = [start_idx, start_idx+1, start_idx+1+nxCells, start_idx+2+nxCells]
                    faces.append(face)

    for j in range(nyCells):
        face_ylen = len(faces)
        for i in range(nxCells+1):
                vertical_start = i+face_ylen
                bottom_idx = (nxCells+1) * (nyCells+1)
                face_vertical = [vertical_start, vertical_start+1+nxCells, vertical_start+bottom_idx, vertical_start+bottom_idx+nxCells+1]
                faces.append(face_vertical)

    for j in range(nyCells+1):
        face_ylen = len(faces)
        for i in range(nxCells):
                horizontal_start = face_ylen+j+i
                bottom_idx = (nxCells+1) * (nyCells+1)
                face_horizontal = [horizontal_start, horizontal_start+1, horizontal_start+bottom_idx, horizontal_start+bottom_idx+1]
                faces.append(face_horizontal)

    return np.array(faces) 

def write_points(points):

    with open("points.txt", "w") as file:
        for point in points:
            point = f"({point[0]},{point[1]},{point[2]})\n"
            file.writelines(point)

if __name__ == "__main__":

    mesh_files(0, 0.1, 0, 0.1, 0, 1, 2, 2, 1)