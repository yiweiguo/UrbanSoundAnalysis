import random

def load_all_data():
    path_1 = "trec/trec05p-1/full/index"
    path_2 = "trec/trec06p/full/index"
    path_3 = "trec/trec07p/full/index"
    target_path = "trec/train/full/index"
    fd_1 = open(path_1, "r")
    fd_2 = open(path_2, "r")
    fd_3 = open(path_3, "r")
    fd = open(target_path, "w")
    lines_1 = fd_1.readlines()
    lines_2 = fd_2.readlines()
    lines_3 = fd_3.readlines()
    while lines_1 or lines_2 or lines_3:
        if lines_1:
            line_1 = lines_1.pop(0).strip(" \n")
            line_1_tokens = line_1.split(" ")
            line_1_path = line_1_tokens[0] + " trec/trec05p-1" + line_1_tokens[1][2:] + "\n"
            fd.write(line_1_path)
        if lines_2:
            line_2 = lines_2.pop(0).strip(" \n")
            line_2_tokens = line_2.split(" ")
            line_2_path = line_2_tokens[0] + " trec/trec06p" + line_2_tokens[1][2:] + "\n"
            fd.write(line_2_path)
        if lines_3:
            line_3 = lines_3.pop(0).strip(" \n")
            line_3_tokens = line_3.split(" ")
            line_3_path = line_3_tokens[0] + " trec/trec07p" + line_3_tokens[1][2:] + "\n"
            fd.write(line_3_path)
    fd_1.close()
    fd_2.close()
    fd_3.close()
    fd.close()

def create_3000():
    target_path = "trec/train/random3000/index"
    full_path = "trec/train/full/index"
    fd = open(target_path, "w")
    fd_read = open(full_path, "r")
    lines = fd_read.readlines()
    random_lines = random.sample(lines, 3000)
    for line in random_lines:
        fd.write(line)
    fd.close()

def main():
    #load_all_data()
    create_3000()

if __name__ == "__main__":
    main()



