def boundary_parser():

    file = []
    patches = []

    with open("MeshFiles/boundary.txt") as f:
        for line in f.readlines():
            line_stripped = line.strip().strip(';')
            file.append(line_stripped.split())

    for i in range(len(file)):

        if "{" in file[i]:
            patches.append(file[i-1])

        if file[i][0] == "nFaces":
            #print(int(file[i+1][1]), int(file[i+1][1])+int(file[i][1]))
            boundary_faces = list(range(int(file[i+1][1]), int(file[i+1][1])+int(file[i][1])))
            patches.append(boundary_faces)
    
    with open("MeshFiles/boundary_patches.txt", "w") as f:
        for ls in patches:
            if type(ls[0]) == str:
                f.write(ls[0] + "\n")
            else:
                f.write("(")
                for i in range(len(ls)):
                    if i == len(ls)-1:
                        f.write(str(ls[i]) + ")\n")
                        continue
                    f.write(str(ls[i]) + " ")


if __name__ == "__main__":
    boundary_parser()