def read_owner_neighbours(owner_filename, neighbour_filename):
    
    owner_lst = read_file(owner_filename)
    neighbour_lst = read_file(neighbour_filename)
    
    cells = create_cells(owner_lst, neighbour_lst)

    write_file("cells.txt", cells)

def write_file(filename, data):

    with open("MeshFiles/"+filename, "w") as f:
        for i in range(len(data)):
            f.write("(")
            for j in range(len(data[i])):
                if j == len(data[i])-1:
                    f.write(str(data[i][j])+")\n")
                    continue
                f.write(str(data[i][j])+" ")

def read_file(filename):

    lst = []

    with open("MeshFiles/"+filename) as f:
        for line in f.readlines():
            lst.append(int(line))

    return lst

def create_cells(owners, neighbours):

    num_cells = max(owners)
    cells = [[] for i in range(num_cells+1)]

    for i in range(len(owners)):
        cells[owners[i]].append(i)
    
    for i in range(len(neighbours)):
        neighbour = neighbours[i]
        cells[neighbour].append(i)

    return cells

if __name__ == "__main__":

    read_owner_neighbours("owner.txt", "neighbour.txt")