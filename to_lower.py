# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/'

in1 = io_folder_path + 'colocation_32.txt'

tmp = []
with open(in1, 'r') as f:
    for line in f:
        tmp.append(line.lower())

out = in1.split('.')[0] + '_low.' + in1.split('.')[1]

with open(out, 'w') as f:
    for line in tmp:
        f.write(line)