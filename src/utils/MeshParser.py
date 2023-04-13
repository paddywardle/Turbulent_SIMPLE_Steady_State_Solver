import numpy as np

class MeshParser:

    def __init__(self, directory):
        
        self.directory = directory
        self.faces = self.Faces()
        self.points = self.Points()
        self.owners, self.neighbours = self.OwnerNeighbours()
        self.cells = self.Cells()
        self.boundaries = self.Boundaries()

    def OwnerNeighbours(self):
        
        owners = []

        with open(f"{self.directory}/owner.txt") as f:
            for line in f.readlines()[21:]:
                if line.strip() == ")":
                    break
                owners.append(int(line))

        neighbours = [-1]*len(owners)

        with open(f"{self.directory}/neighbour.txt") as f:
            for i, line in enumerate(f.readlines()[21:]):
                if line.strip() == ")":
                    break
                neighbours[i] = int(line)

        return np.array(owners), np.array(neighbours)
    
    def Faces(self):

        faces = []

        with open(f"{self.directory}/faces.txt") as f:
            for i, line in enumerate(f.readlines()[20:]):
                if line.strip() == ")":
                    break
                line = line[1:].strip().strip('()').split()
                line = list(map(int, line))

                faces.append(line)

        return np.array(faces)
    
    def Points(self):

        faces = []

        with open(f"{self.directory}/points.txt") as f:
            for i, line in enumerate(f.readlines()[20:]):
                if line.strip() == ")":
                    break
                line = line.strip().strip('()').split()
                line = list(map(float, line))
                faces.append(line)

        return np.array(faces)
    
    def Cells(self):

        num_cells = max(self.owners)
        cells = [[] for i in range(num_cells+1)]

        for i, (owner, neighbour) in enumerate(zip(self.owners, self.neighbours)):
            cells[owner].append(i)
            if neighbour == -1:
                continue
            cells[neighbour].append(i)

        return np.array(cells)

    def Boundaries(self):
        
        boundaries = {}
        with open(f"{self.directory}/boundary.txt") as f:
            patches = f.readlines()[19:]
            for i, line in enumerate(patches):
                if line.strip() == ")":
                    break
                elif line.strip() == '{':
                    nfaces = int(patches[i+2].strip().strip(';').split()[1])
                    startFace = int(patches[i+3].strip().strip(';').split()[1])
                    boundaries[patches[i-1].strip()] = list(range(startFace, startFace+nfaces))

        return boundaries

if __name__ == "__main__":

    mp = MeshParser("MeshFiles/backward_step")
    points = [0,1,41,42,2091,2092,2132,2133]
    print(mp.Points()[[2, 5]])