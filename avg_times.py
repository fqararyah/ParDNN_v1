import utils
from os import walk


# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/time_steps/'

# output file
out1 = io_folder_path + 'nodes_average_durations.txt'

files = []
for (dirpath, dirnames, filenames) in walk(io_folder_path):
    files.extend(filenames)
    break

# getting time (weight) info for nodes
nodes_durations = {}
for file in files:
    if 'json' in file:
        analysis_graph = utils.read_profiling_file(io_folder_path + file)
        for node in analysis_graph:
            if node in nodes_durations:
                nodes_durations[node].append(analysis_graph[node].duration)
            else:
                nodes_durations[node] = [analysis_graph[node].duration]

for node, running_times in nodes_durations.items():
    nodes_durations[node].sort()
    if len(nodes_durations[node]) > 4:
        nodes_durations[node] = nodes_durations[node][2:len(
            nodes_durations[node]) - 2]

with open(out1, 'w') as f:
    for node, running_times in nodes_durations.items():
        mean = int(sum(nodes_durations[node]) / len(nodes_durations[node]))
        median = int(nodes_durations[node]
                     [int(len(nodes_durations[node]) / 2)])
        to_write = 0
        if mean >= 1.5 * median or mean <= median / 1.5:
            to_write = int((median + int(nodes_durations[node][int(len(nodes_durations[node]) / 4)]) + int(
                nodes_durations[node][int(3 * len(nodes_durations[node]) / 4)])) / 3)
            if len(nodes_durations[node]) >= 4 and nodes_durations[node][int(2 * len(nodes_durations[node]) / 3)] >= 2 * median:
                print(node + ', ' + str(nodes_durations[node]) + ', The mean is: ' + str(
                    mean) + ', The median is:' + str(median) + ' ' + str(to_write))
        else:
            to_write = int((mean + median) / 2)
        if node.lower() == 'tower_0/mixed_8x8x2048a/branch3x3dbl/conv_1/conv2d':
            print(node + ', ' + str(nodes_durations[node]) + ', The mean is: ' + str(
                    mean) + ', The median is:' + str(median) + ' ' + str(to_write))
                    
        f.write(node.lower() + '::' + str(to_write) + '\n')
