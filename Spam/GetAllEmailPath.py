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
            line_1 = lines_1.pop(0)
            line_1_tokens = line_1.split(" ")
            line_1_path = line_1_tokens[0] + " trec/trec05p-1" + line_1_tokens[1][2:]
            fd.write(line_1_path)
        if lines_2:
            line_2 = lines_2.pop(0)
            line_2_tokens = line_2.split(" ")
            line_2_path = line_2_tokens[0] + " trec/trec06p" + line_2_tokens[1][2:]
            fd.write(line_2_path)
        if lines_3:
            line_3 = lines_3.pop(0)
            line_3_tokens = line_3.split(" ")
            line_3_path = line_3_tokens[0] + " trec/trec07p" + line_3_tokens[1][2:]
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

def create_3000_even():
    path_1 = "trec/trec05p-1/full/index"
    path_2 = "trec/trec06p/full/index"
    path_3 = "trec/trec07p/full/index"
    target_path = "trec/train/random3000even/index"
    fd_1 = open(path_1, "r")
    fd_2 = open(path_2, "r")
    fd_3 = open(path_3, "r")
    fd = open(target_path, "w")
    lines_1 = fd_1.readlines()
    lines_2 = fd_2.readlines()
    lines_3 = fd_3.readlines()
    random_lines_1 = random.sample(lines_1, 1000)
    random_lines_2 = random.sample(lines_2, 1000)
    random_lines_3 = random.sample(lines_3, 1000)
    while random_lines_1 or random_lines_2 or random_lines_3:
        if random_lines_1:
            random_line_1 = random_lines_1.pop(0)
            random_line_1_tokens = random_line_1.split(" ")
            random_line_1_path = random_line_1_tokens[0] + " trec/trec05p-1" + random_line_1_tokens[1][2:]
            fd.write(random_line_1_path)
        if random_lines_2:
            random_line_2 = random_lines_2.pop(0)
            random_line_2_tokens = random_line_2.split(" ")
            random_line_2_path = random_line_2_tokens[0] + " trec/trec06p" + random_line_2_tokens[1][2:]
            fd.write(random_line_2_path)
        if random_lines_3:
            random_line_3 = random_lines_3.pop(0)
            random_line_3_tokens = random_line_3.split(" ")
            random_line_3_path = random_line_3_tokens[0] + " trec/trec07p" + random_line_3_tokens[1][2:]
            fd.write(random_line_3_path)
    fd_1.close()
    fd_2.close()
    fd_3.close()
    fd.close()

def create_5000_test(super_path):
    target_path = super_path + "/test/index"
    open_path = super_path + "/full/index"
    fd_full = open(open_path, "r")
    lines = fd_full.readlines()
    random_lines = random.sample(lines, 5000)
    fd_write = open(target_path, "w")
    for random_line in random_lines:
        fd_write.write(random_line)
    fd_full.close()
    fd_write.close()

def main():
    #load_all_data()
    #create_3000()
    #create_3000_even()
    create_5000_test("trec/trec05p-1")
    create_5000_test("trec/trec06p")
    create_5000_test("trec/trec07p")

if __name__ == "__main__":
    main()



