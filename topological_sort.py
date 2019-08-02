import json
import utils
import queue

#folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/'

#input files
in1 = io_folder_path + 'rev_resnet_src_sink.dot' #'rev_inc_A_dot_src_sink.dot'

#output files
out1 = io_folder_path + 'rev_resnet_src_sink_nodes_levels.txt' #'rev_src_snk_nodes_levels.txt'

#will contain the graph as an adgacency list
graph = {}
#will contain the nodes and their levels
nodes_levels = {}
#this queue will be used as a temporary container in the topological sort
visit = queue.Queue()


#constructing the graph and initializing the nodes levels from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes)  > 1:
            if not nodes[0] in nodes_levels:
                nodes_levels[nodes[0]] = -1
            if not nodes[1] in nodes_levels:
                nodes_levels[nodes[1]] = -1
            if nodes[0] in graph:
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]

#topological sort
for node_name in graph.keys():
    if nodes_levels[node_name] == -1:
        nodes_levels[node_name] = 0
        visit.put(node_name)
        while not visit.empty():
            curr_node = visit.get()
            curr_level = nodes_levels[curr_node]
            for adj in graph[curr_node]:
                if curr_level >= nodes_levels[adj]:
                    nodes_levels[adj] = curr_level + 1
                    if adj in graph:
                        visit.put(adj)

#writing results to file          
with open(out1, 'w') as f:
    for node_name, node_level in nodes_levels.items():
        f.write(node_name + "::" + str(node_level) + "\n")
