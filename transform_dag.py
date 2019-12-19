import utils
import json

#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_480/'
# input files

io_folder_path = utils.io_folder_path

network_app = utils.network_app

in1 = io_folder_path + network_app + '.dot'
in2 = io_folder_path + 'tensors_sz_32_low.txt'

out = io_folder_path + network_app + 't_low.dot'

all_nodes = {}
graph = {}
# constructing the graph and initializing the nodes levels from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            all_nodes[nodes[0]] = "1"
            all_nodes[nodes[1]] = "1"
            if nodes[0] in graph:
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]

for node in all_nodes.keys():
    if node.startswith('^'):
        normal_node = node[1:]
        if normal_node in all_nodes:
            if normal_node in graph.keys():
                if node not in graph[normal_node]:
                    graph[normal_node].append(node)
            else:
                graph[normal_node] = [node]

with open(out, 'w') as f:
    f.write('digraph{\n')
    for node,adjs in graph.items():
        for adj in adjs:
            f.write('"' + node.lower() + '"->"' + adj.lower() + '"\n')
    f.write('}')

tensors_sizes = {}
# get tensors sizes
with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensors_sizes[splitted[0]] = splitted[1]

print(len(tensors_sizes))

with open(in2, 'w') as f:
    for tensor, size in tensors_sizes.items():
        if not tensor.startswith("^"):
            f.write(tensor+"::"+size+"\n")
            f.write("^"+tensor+"::" + tensors_sizes[node[1:]] + "\n")

