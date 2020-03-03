import utils
import nodeProps
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import collections
import heapq
import random
import queue
import time
import numpy as np
import statistics

# folder containing the work files
io_folder_path = utils.io_folder_path
network_app = utils.network_app
in1 = io_folder_path + network_app + \
    '_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step17_low.json'
# 'part_8_1799_src_sink_nodes_levels.txt'
in3 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
# 'rev_part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_' + network_app + '_src_sink_nodes_levels_low.txt'
in4_b = io_folder_path + 'rev_' + network_app + '_src_sink_low.dot'
in5 = io_folder_path + 'tensors_sz_32_low.txt'
in6 = io_folder_path + 'memory.txt'
in6_b = io_folder_path + 'res_memory.txt'
in8 = io_folder_path + 'collocations.txt'
in9 = io_folder_path + 'no_ops.txt'
in10 = io_folder_path + 'ref_nodes.txt'
in11 = io_folder_path + 'var_nodes.txt'
in12 = io_folder_path + 'vanilla_cleaned.place'

# output file
out1 = io_folder_path + 'placement.place'

# grouper parameters
no_of_desired_groups = 8
memory_limit_per_group = 30 * 1024 * 1024 * 1024

#tst
comm_latency = 45
average_tensor_size_if_not_provided = 1
comm_transfer_rate = 1000000 / (140 * 1024 * 1024 * 1024)

# will contain the graph as an adgacency list
graph = {}
rev_graph = {}
all_nodes = {}
sink_node_name = 'snk'
source_node_name = 'src'
graph[sink_node_name] = []
rev_graph[source_node_name] = []

# initializing the nodes and adjacencies from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            if not splits[0] in all_nodes:
                all_nodes[splits[0]] = nodeProps.NodeProps()
            if not splits[1] in all_nodes:
                all_nodes[splits[1]] = nodeProps.NodeProps()

            if splits[0] in graph.keys():
                graph[splits[0]].append(splits[1])
            else:
                graph[splits[0]] = [splits[1]]

# constructing the graph and initializing the nodes levels from the dot file
with open(in4_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            if nodes[0] in rev_graph:
                rev_graph[nodes[0]].append(nodes[1])
            else:
                rev_graph[nodes[0]] = [nodes[1]]

no_op_nodes = {}
with open(in9, 'r') as f:
    for line in f:
        no_op_nodes[utils.clean_line(line)] = 1

# getting time (weight) info for nodes
analysis_graph = utils.read_profiling_file(in2, True)

sudo_nodes = {}
for node, node_props in all_nodes.items():
    if node not in analysis_graph:
        analysis_graph[node] = node_props
    if node.startswith('^'):
        sudo_nodes[node] = 1

levels_weights = {}
no_of_levels = 0
# get nodes levels
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            int_node_level = int(node_and_level[1])
            analysis_graph[node_and_level[0]].level = int_node_level
            if int_node_level in levels_weights.keys():
                levels_weights[int_node_level] = levels_weights[int_node_level] + \
                    analysis_graph[node_and_level[0]].duration
            else:
                levels_weights[int_node_level
                               ] = analysis_graph[node_and_level[0]].duration
                no_of_levels = no_of_levels + 1

tensors_sizes = {}
edges_weights = {}
# get tensors sizes
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensor_size = int(splitted[1])
        tensor_name = splitted[0]
        tensors_sizes[tensor_name] = tensor_size
        edge_weight = float(tensor_size) * comm_transfer_rate + comm_latency
        edges_weights[tensor_name] = {}
        if tensor_name in graph:
            for adj_node in graph[tensor_name]:
                if adj_node in no_op_nodes or adj_node in sudo_nodes or adj_node == sink_node_name:
                    edges_weights[tensor_name][adj_node] = comm_latency
                else:
                    edges_weights[tensor_name][adj_node] = edge_weight 

collocations = []
nodes_collocation_groups = {}
indx = 0
with open(in8, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        collocations.append([])
        splits = line.split("::")
        for node in splits:
            collocations[indx].append(node)
            nodes_collocation_groups[node] = indx
        indx += 1

ref_nodes = {}
with open(in10, 'r') as f:
    for line in f:
        ref_nodes[utils.clean_line(line)] = 1

var_nodes = {}
with open(in11, 'r') as f:
    for line in f:
        var_nodes[utils.clean_line(line)] = 1

vanilla_placement = {}
with open(in12, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line).lower()
        splits = line.split(' ')
        vanilla_placement[splits[0]] = splits[1]

t0 = time.time()

# get_node_average_weight
total_nodes_weight = 0
for node, node_props in analysis_graph.items():
    total_nodes_weight = total_nodes_weight + node_props.duration

average_node_weight = total_nodes_weight/len(analysis_graph)

for node in all_nodes:
    if not node in tensors_sizes:
        tensors_sizes[node] = 0
        edges_weights[node] = {}
        for adj_node in graph[node]:
            edges_weights[node][adj_node] = float(comm_latency)

# get nodes in degrees for the topological sort
nodes_in_degrees = {}
for node in all_nodes:
    if node in rev_graph:
        nodes_in_degrees[node] = len(rev_graph[node])
    else:
        nodes_in_degrees[node] = 0
# get reverse nodes in degrees for the topological sort
rev_nodes_in_degrees = {}
for node in all_nodes:
    if node in graph:
        rev_nodes_in_degrees[node] = len(graph[node])
    else:
        rev_nodes_in_degrees[node] = 0

#nodes bottom levels
def get_nodes_weighted_levels(graph, edges_weights, src_nodes = None, previosly_visited = [], grouped = False, nodes_groups = {}, is_rev = True, _nodes_in_degrees = rev_nodes_in_degrees):
    # getting the sources of the graph to start the topological traversal from them
    graph_keys = {}
    nodes_weighted_levels={}
    tmp_nodes_in_degrees = copy.deepcopy(_nodes_in_degrees)
    traversal_queueu = queue.Queue()

    if src_nodes is None:
        for graph_key in graph.keys():
            graph_keys[graph_key] = 0

        for adj_nodes in graph.values():
            for node in adj_nodes:
                if node in graph_keys:
                    graph_keys[node] = 1
        src_nodes = {}
        for node, source_node in graph_keys.items():
            if source_node == 0:
                src_nodes[node] = 1

    for node in src_nodes:
        traversal_queueu.put(node)
    for node in graph.keys():
        nodes_weighted_levels[node] = 0  # analysis_graph[node].duration

    # start the traversal
    while not traversal_queueu.empty():
        current_node = traversal_queueu.get()
        adj_nodes = graph[current_node]
        current_node_level = nodes_weighted_levels[current_node]
        current_node_duration = analysis_graph[current_node].duration
        for adj_node in adj_nodes:
            if adj_node not in previosly_visited:
                #this is correct, might seem confusing, remember we are working with the reversed graph
                if is_rev:
                    edge_weight = edges_weights[adj_node][current_node]
                else:
                    edge_weight = edges_weights[current_node][adj_node]
                if grouped:
                    if nodes_groups[adj_node] == nodes_groups[current_node]:
                        edge_weight = 0
                new_level = current_node_level + edge_weight + (analysis_graph[adj_node].duration if is_rev else current_node_duration)
                tmp_nodes_in_degrees[adj_node] -= 1
                if nodes_weighted_levels[adj_node] < new_level:
                    nodes_weighted_levels[adj_node] = new_level
                if tmp_nodes_in_degrees[adj_node] == 0:
                    traversal_queueu.put(adj_node)
    return nodes_weighted_levels

# extracting all vertical paths in the graph
rev_graph[source_node_name] = []
free_nodes = []
paths = []
current_path = []
visited = {}
src_nodes = {}
groups_weights = []
paths_lengths = []
current_path_weight = 0
current_path_weight_with_comm = 0
num_paths = 0
nodes_paths_mapping = {}
nodes_to_visit = list(all_nodes.keys())
tmp_rev_graph = copy.deepcopy(rev_graph)
tmp_nodes_in_degrees = copy.deepcopy(nodes_in_degrees)

nodes_weighted_levels = get_nodes_weighted_levels(rev_graph, edges_weights, src_nodes=[sink_node_name])


for node, weighted_level in nodes_weighted_levels.items():
    heapq.heappush(free_nodes, (-weighted_level, node))

tmp_rev_nodes_in_degrees = copy.deepcopy(rev_nodes_in_degrees)
while free_nodes:
    current_node = heapq.heappop(free_nodes)[1]
    while current_node in visited and free_nodes:
        current_node = heapq.heappop(free_nodes)[1]

    while current_node !='' and current_node not in visited:
        current_path.append(current_node)
        current_path_weight = current_path_weight + \
            analysis_graph[current_node].duration
        if len(current_path) > 1:
            current_path_weight_with_comm = current_path_weight_with_comm + \
                analysis_graph[current_node].duration + edges_weights[current_path[-2]][current_node]
        visited[current_node] = 1
        src_nodes[current_node] = 1
        max_priority = -1
        next_node = ''
        for adj_node in graph[current_node]:
            if adj_node not in visited and nodes_weighted_levels[adj_node] > max_priority:
                max_priority = nodes_weighted_levels[adj_node]
                next_node = adj_node
        current_node = next_node

    if len(current_path) > 0:
        if current_node != '':
            current_path_weight_with_comm += edges_weights[current_path[-1]][current_node]
        paths.append(current_path)
        groups_weights.append(current_path_weight)
        paths_lengths.append(len(current_path))
        if len(paths) <= no_of_desired_groups or current_path_weight_with_comm >= groups_weights[0]:
            nodes_weighted_levels = get_nodes_weighted_levels(graph = tmp_rev_graph, edges_weights = edges_weights, src_nodes= src_nodes, previosly_visited= visited)
            free_nodes = []
            for node, weighted_level in nodes_weighted_levels.items():
                heapq.heappush(free_nodes, (-weighted_level, node))

        for node in current_path:
            del tmp_rev_nodes_in_degrees[node]
            for adj_node in graph[node]:
                tmp_nodes_in_degrees[adj_node] -= 1
                if adj_node in visited and tmp_nodes_in_degrees[adj_node] == 0:
                    del tmp_rev_graph[adj_node]
                    del src_nodes[adj_node]

        current_path = []
        current_path_weight = 0
        current_path_weight_with_comm = 0
        num_paths = num_paths + 1

# sort paths from shortest to longest
paths_lengths, groups_weights, paths = (list(t) for t in zip(
    *sorted(zip(paths_lengths, groups_weights, paths))))
print('num of paths: ' + str(len(paths)))
print(paths_lengths[-20:])

print('paths obtained: ' + str( time.time() - t0 ))
t0 = time.time()
# which node is in which path
nodes_paths_mapping[source_node_name] = num_paths - 1
nodes_paths_mapping[sink_node_name] = num_paths - 1
for i in range(0, num_paths):
    for node in paths[i]:
        nodes_paths_mapping[node] = i

# get max potential of paths
groups_parents = {}
paths_max_potential = copy.deepcopy(groups_weights)
levels_work_sums = {}
nodes_weighted_levels = get_nodes_weighted_levels(graph= graph, grouped= True, edges_weights = edges_weights, nodes_groups= nodes_paths_mapping, \
    is_rev= False, _nodes_in_degrees= nodes_in_degrees, src_nodes= [source_node_name])
for node, level in nodes_weighted_levels.items():
    if level not in levels_work_sums:
        levels_work_sums[level] = 0
    levels_work_sums[level] += analysis_graph[node].duration

levels = levels_work_sums.keys() 
work_sums = levels_work_sums.values()

levels, work_sums = (list(t) for t in zip(
        *sorted(zip(levels, work_sums))))

for i in range(1, len(work_sums)):
    work_sums[i] += work_sums[i - 1]

levels_indices_map = {}
current_level = 0
for level in levels:
    if level not in levels_indices_map:
        levels_indices_map[level] = current_level 
        current_level += 1

paths_comms = []
paths_ranges = []
path_ranges_subtract = []
paths_parents = []
for i in range(0, len(paths)):
    path = paths[i]
    if path[0] == source_node_name:
        continue

    path_comm = 0
    path_last_src = 0
    path_first_snk = math.inf
    path_head = path[0]
    path_tail = path[-1]
    path_parents = {}

    first_child_comp = 0
    max_child_comm = 0
    for child in graph[path_tail]:
        child_path = nodes_paths_mapping[child]
        if child_path not in path_parents:
            path_parents[child_path] = 0
        path_parents[child_path] = max(path_parents[child_path], edges_weights[path_tail][child])
        max_child_comm = max(max_child_comm, edges_weights[path_tail][child])
        if nodes_weighted_levels[child] < path_first_snk:
            path_first_snk = nodes_weighted_levels[child]
            path_parents[child_path] = edges_weights[path_tail][child]
            first_child_comp = analysis_graph[child].duration

    path_comm += max_child_comm

    last_parent_comp = 0
    for parent in rev_graph[path_head]:
        path_comm += edges_weights[parent][path_head]
        if nodes_weighted_levels[parent] > path_last_src:
            path_last_src = nodes_weighted_levels[parent]
            parent_path = nodes_paths_mapping[parent]
            last_parent_comp = analysis_graph[parent].duration
            if parent_path not in path_parents:
                path_parents[parent_path] = 0
            path_parents[parent_path] += edges_weights[parent][path_head]

    path_ranges_subtract.append(last_parent_comp + first_child_comp)

    max_comm = 0
    max_comm_indx = 0
    for path_parent, comm in path_parents.items():
        if comm > max_comm:
            max_comm = comm
            max_comm_indx = path_parent

    paths_parents.append(max_comm_indx)

    paths_comms.append(path_comm)
    paths_ranges.append([path_last_src, path_first_snk])

# get the average path length
after_heavy_paths_count = 0
after_heavy_paths_lengths = 0
for path in paths:
    after_heavy_paths_count = after_heavy_paths_count + 1
    after_heavy_paths_lengths = after_heavy_paths_lengths + len(path)

average_path_len = round(after_heavy_paths_lengths / after_heavy_paths_count)

print('averge path len: ' + str(average_path_len))

# getting initial groups
initial_groups = copy.deepcopy(paths)
initial_groups_indices = [1] * num_paths
path_joined_group = {}
paths_become_groups = {}
for i in range(0, len(paths) - 1 ):
    if i in paths_become_groups:
        continue
    path = paths[i]
    path_parent = paths_parents[i]
    path_max_potential = ( work_sums[ levels_indices_map[paths_ranges[i][1]] ] - work_sums[ levels_indices_map[paths_ranges[i][0]] ] ) - ( groups_weights[i]  + path_ranges_subtract[i] )
    if paths_comms[i] >= path_max_potential:
        initial_groups_indices[i] = 0
        groups_weights[path_parent] += groups_weights[i]
        initial_groups[path_parent] += initial_groups[i]

tmp_initial_groups = initial_groups
initial_groups = []
tmp_groups_weights = groups_weights
groups_weights = []
num_initial_groups = 0
for i in range(0, num_paths):
    if initial_groups_indices[i] == 1:
        initial_groups.append(tmp_initial_groups[i])
        groups_weights.append(tmp_groups_weights[i])
        num_initial_groups = num_initial_groups + 1

# parts work distribution over levels
tasks_per_levels = []
max_levels = [0]*len(initial_groups)
min_levels = [1000000]*len(initial_groups)

for i in range(0, len(initial_groups)):
    tasks_per_levels.append(collections.OrderedDict())
    current_group = initial_groups[i]
    for node in current_group:
        node_props = analysis_graph[node]
        node_level = int(node_props.level)
        if node_level in tasks_per_levels[i].keys():
            tasks_per_levels[i][node_level] += node_props.duration
        else:
            tasks_per_levels[i][node_level] = node_props.duration

        if node_level < min_levels[i]:
            min_levels[i] = node_level
        if node_level > max_levels[i]:
            max_levels[i] = node_level

print('Initial merging is done: ' + str( time.time() - t0 ))
t0 = time.time()
# getting main groups-------------------------------------------------

# Returns sum of arr[0..index]. This function assumes 
# that the array is preprocessed and partial sums of 
# array elements are stored in BITree[]. 
def getsum(BITTree,i): 
    s = 0 #initialize result 
  
    # index in BITree[] is 1 more than the index in arr[] 
    i = i+1
  
    # Traverse ancestors of BITree[index] 
    while i > 0: 
  
        # Add current element of BITree to sum 
        s += BITTree[i] 
  
        # Move index to parent node in getSum View 
        i -= i & (-i) 
    return s 
  
# Updates a node in Binary Index Tree (BITree) at given index 
# in BITree. The given value 'val' is added to BITree[i] and 
# all of its ancestors in tree. 
def updatebit(BITTree , n , i ,v): 
  
    # index in BITree[] is 1 more than the index in arr[] 
    i += 1
  
    # Traverse all ancestors and add 'val' 
    while i <= n: 
  
        # Add 'val' to current node of BI Tree 
        BITTree[i] += v 
  
        # Update index to that of parent in update View 
        i += i & (-i) 
  
  
# Constructs and returns a Binary Indexed Tree for given 
# array of size n. 
def construct(arr, n): 
  
    # Create and initialize BITree[] as 0 
    BITTree = [0]*(n+1) 
  
    # Store the actual values in BITree[] using update() 
    for i in range(n): 
        updatebit(BITTree, n, i, arr[i]) 

    return BITTree


final_groups = []
final_groups_weights = []
to_be_merged_groups = []
to_be_merged_groups_weights = []

for i in range(1, no_of_desired_groups + 1):
    final_groups.append(copy.deepcopy(initial_groups[-i]))
    final_groups_weights.append(groups_weights[-i])

final_groups_work_per_levels = []
work_trees = []
final_groups_max_levels = [0]*len(final_groups)
final_groups_min_levels = [math.inf]*len(final_groups)

for indx in range(0, no_of_desired_groups):
    final_groups_work_per_levels.append([])
    work_trees.append([])
    final_groups_work_per_levels[indx] = [0] * no_of_levels

for indx in range(0, no_of_desired_groups):
    current_group = final_groups[indx]
    for node in current_group:
        final_groups_work_per_levels[indx][analysis_graph[node].level] += analysis_graph[node].duration

for indx in range(0, no_of_desired_groups):
    work_trees[indx] = construct(final_groups_work_per_levels[indx],no_of_levels) 

nodes_groups = {}
for node in all_nodes:
    nodes_groups[node] = -1

for i in range(0, len(final_groups)):
    for node in final_groups[i]:
        nodes_groups[node] = i

for i in range(0, len(initial_groups) - no_of_desired_groups):
    #if i not in filling_groups:
    to_be_merged_groups.append(copy.deepcopy(initial_groups[i]))
    to_be_merged_groups_weights.append(groups_weights[i])

# parts work distribution over levels
to_be_merged_groups_earliest_sink_levels = []
to_be_merged_groups_latest_src_levels = []
to_be_merged_groups_tasks_per_levels = []
to_be_merged_groups_len = len(to_be_merged_groups)
to_be_merged_groups_max_levels = [0] * to_be_merged_groups_len
to_be_merged_groups_min_levels = [math.inf] * to_be_merged_groups_len
to_be_merged_groups_densities = [0] * to_be_merged_groups_len
to_be_merged_groups_lengths = [0] * to_be_merged_groups_len
to_be_merged_groups_empty_spots = [0] * to_be_merged_groups_len
to_be_merged_groups_sorting_criteria = [0] * to_be_merged_groups_len
penalize_small_paths = [0] * to_be_merged_groups_len
to_be_merged_groups_indices = []

for i in range(0, to_be_merged_groups_len):
    to_be_merged_groups_indices.append(i)
    to_be_merged_groups_tasks_per_levels.append(collections.OrderedDict())
    current_group = to_be_merged_groups[i]
    min_level = math.inf
    max_level = 0
    for node in current_group:
        node_props = analysis_graph[node]
        node_level = int(node_props.level)
        if node_level in to_be_merged_groups_tasks_per_levels[i].keys():
            to_be_merged_groups_tasks_per_levels[i][node_level] += node_props.duration
        else:
            to_be_merged_groups_tasks_per_levels[i][node_level] = node_props.duration

        if node_level < min_level:
            min_level = node_level
        if node_level > max_level:
            max_level = node_level

    to_be_merged_groups_min_levels[i] = min_level
    to_be_merged_groups_max_levels[i] = max_level
    
    sink_level = math.inf
    for snk_node in graph[current_group[-1]]:
        if int(analysis_graph[snk_node].level) < sink_level:
            sink_level = int(analysis_graph[snk_node].level)

    spanning_over = sink_level - min_level
    to_be_merged_groups_lengths[i] = len(current_group)
    to_be_merged_groups_empty_spots[i] = max(spanning_over - len(current_group) - (sink_level - max_level), 0)
    if len(current_group) < average_path_len:
        penalize_small_paths[i] = 1

    earliest_sink_evel = math.inf
    end_node = current_group[-1]
    for child in graph[end_node]:
        if analysis_graph[child].level < earliest_sink_evel:
            earliest_sink_evel = analysis_graph[child].level

    to_be_merged_groups_earliest_sink_levels.append(earliest_sink_evel)
    to_be_merged_groups_latest_src_levels.append( analysis_graph[current_group[0]].level - 1 )

    if spanning_over <= 0:
        to_be_merged_groups_densities[i] = 0
    else:
        to_be_merged_groups_densities[i] = to_be_merged_groups_weights[i] / spanning_over

if to_be_merged_groups_densities:
    normalized_densities_den = max(to_be_merged_groups_densities) - min(to_be_merged_groups_densities) + 1
    normalized_lengths_den = max(to_be_merged_groups_lengths) - min(to_be_merged_groups_lengths) + 1
    normalized_empty_spots_den = max(to_be_merged_groups_empty_spots) - min(to_be_merged_groups_empty_spots) + 1
    normalized_weights_den = max(to_be_merged_groups_weights) - min(to_be_merged_groups_weights) + 1
    normalized_densities_sub = min(to_be_merged_groups_densities)
    normalized_lengths_sub = min(to_be_merged_groups_lengths)
    normalized_weights_sub = min(to_be_merged_groups_weights)
    normalized_empty_spots_sub = min(to_be_merged_groups_empty_spots)

    for i in range(0, to_be_merged_groups_len):
        to_be_merged_groups_sorting_criteria[i] = (to_be_merged_groups_weights[i] - normalized_weights_sub) / normalized_weights_den + \
        (to_be_merged_groups_densities[i] - normalized_densities_sub) / normalized_densities_den + \
            (to_be_merged_groups_lengths[i] - normalized_lengths_sub) / (normalized_lengths_den) \
            - (to_be_merged_groups_empty_spots[i] - normalized_empty_spots_sub) / normalized_empty_spots_den - penalize_small_paths[i]

    to_be_merged_groups_sorting_criteria, to_be_merged_groups_weights, to_be_merged_groups_min_levels, to_be_merged_groups, to_be_merged_groups_max_levels\
        , to_be_merged_groups_tasks_per_levels, to_be_merged_groups_lengths = \
        (list(t) for t in zip(*sorted(zip(to_be_merged_groups_sorting_criteria, to_be_merged_groups_weights, to_be_merged_groups_min_levels, to_be_merged_groups,\
             to_be_merged_groups_max_levels, to_be_merged_groups_tasks_per_levels, to_be_merged_groups_lengths), reverse=True)))

# merging the groups
for to_be_merged_group_index in range(0, len(to_be_merged_groups)):
    to_be_merged_group = to_be_merged_groups[to_be_merged_group_index]
    branch_main_path_indx = -1
    src_min_level = -1
    branch_src_node = ''
    branch_snk_node = ''
    min_sink_level = math.inf

    to_be_merged_group_comms = [0] * no_of_desired_groups
    to_be_merged_group_total_comm = 0

    for node in to_be_merged_group:
        comm_with_children = [0] * no_of_desired_groups
        comm_with_children_total = 0
        for child_node in graph[node]:
            if child_node not in to_be_merged_group:
                child_group = nodes_groups[child_node]
                if child_group != -1:
                    comm_with_children[child_group] = max(comm_with_children[child_group], edges_weights[node][child_node])
                comm_with_children_total = max(comm_with_children_total, edges_weights[node][child_node])
        to_be_merged_group_total_comm += comm_with_children_total
        to_be_merged_group_comms = [sum(x) for x in zip(to_be_merged_group_comms, comm_with_children)]

        for parent_node in rev_graph[node]:
            if parent_node not in to_be_merged_group:
                if nodes_groups[parent_node] != -1:
                    to_be_merged_group_comms[nodes_groups[parent_node]] += edges_weights[parent_node][node] 
                to_be_merged_group_total_comm += edges_weights[parent_node][node]
    
    src_min_level = int(analysis_graph[to_be_merged_group[0]].level - 1)

    for dst_node in graph[to_be_merged_group[-1]]:
        if int(analysis_graph[dst_node].level) < min_sink_level:
            min_sink_level = int(analysis_graph[dst_node].level)

    min_sum_in_targeted_levels = math.inf
    merge_destination_index = 0
    for i in range(0, no_of_desired_groups):
        sum_in_targeted_levels = getsum(work_trees[i], min_sink_level) - getsum(work_trees[i], src_min_level)  
        
        sum_in_targeted_levels += (to_be_merged_group_total_comm - to_be_merged_group_comms[i])

        if sum_in_targeted_levels < min_sum_in_targeted_levels:
            min_sum_in_targeted_levels = sum_in_targeted_levels
            merge_destination_index = i

    merge_min_level = min(
        to_be_merged_groups_min_levels[to_be_merged_group_index], final_groups_min_levels[merge_destination_index])
    merge_max_level = max(
        to_be_merged_groups_max_levels[to_be_merged_group_index], final_groups_max_levels[merge_destination_index])
    final_groups_weights[merge_destination_index] += to_be_merged_groups_weights[to_be_merged_group_index]
    final_groups[merge_destination_index] += to_be_merged_group
    final_groups_min_levels[merge_destination_index] = merge_min_level
    final_groups_max_levels[merge_destination_index] = merge_max_level
    merge_src_levels_tasks = to_be_merged_groups_tasks_per_levels[to_be_merged_group_index]

    for node in to_be_merged_group:
        nodes_groups[node] = merge_destination_index

    for level, tasks_sum in merge_src_levels_tasks.items():
        final_groups_work_per_levels[merge_destination_index][level] += tasks_sum
        updatebit(work_trees[merge_destination_index], no_of_levels, level, tasks_sum)


# nodes levels calculation
def get_nodes_levels_dsc(_graph, edges_weights, bottom_levels, nodes_clusters = {}):
    # getting the sources of the graph to start the topological traversal from them
    nodes_levels = {}
    graph_keys = {}
    nodes_in_degrees = {}
    if bottom_levels:
        for node in graph:
            nodes_in_degrees[node] = len(graph[node])
    else:
        for node in rev_graph:
            nodes_in_degrees[node] = len(rev_graph[node])

    for graph_key in _graph.keys():
        graph_keys[graph_key] = 0

    for adj_nodes in _graph.values():
        for node in adj_nodes:
            if node in graph_keys:
                graph_keys[node] = 1

    traversal_queueu = queue.Queue()
    for node, source_node in graph_keys.items():
        if source_node == 0:
            if bottom_levels:
                nodes_levels[node] = analysis_graph[node].duration
            else:
                nodes_levels[node] = 0
            traversal_queueu.put(node)

    # start the traversal
    while not traversal_queueu.empty():
        current_node = traversal_queueu.get()
        if current_node in _graph:
            adj_nodes = _graph[current_node]
        else:
            adj_nodes = []
        current_node_weight = analysis_graph[current_node].duration
        current_node_level = nodes_levels[current_node]
        for adj_node in adj_nodes:
            edge_weight = edges_weights[current_node][adj_node]
            if len(nodes_clusters) > 0:
                if nodes_clusters[current_node] == nodes_clusters[adj_node]:
                    edge_weight = 0
            if bottom_levels:
                new_level = current_node_level + \
                    + edge_weight \
                    + analysis_graph[adj_node].duration
            else:
                new_level = current_node_level + \
                    + edge_weight \
                    + current_node_weight
            if adj_node not in nodes_levels or nodes_levels[adj_node] < new_level:
                nodes_levels[adj_node] = new_level
            nodes_in_degrees[adj_node] -= 1
            if nodes_in_degrees[adj_node] == 0:
                traversal_queueu.put(adj_node)

    return nodes_levels

def calc_finish_time(no_of_clusters, nodes_clusters, nodes_levels, graph):
    clusters_ready_times = [0] * no_of_clusters
    nodes_ready_times = {}
    nodes = nodes_levels.keys()
    levels = nodes_levels.values()
    levels, nodes = (list(t) for t in zip(*sorted(zip(levels, nodes))))
    nodes_ready_times['src'] = 0
    for node in nodes:
        node_cluster = nodes_clusters[node]
        ready_time = clusters_ready_times[node_cluster]
        nodes_ready_times[node] = ready_time
        for parent in rev_graph[node]:
            if nodes_clusters[parent] != node_cluster:
                time_from_parent = nodes_ready_times[parent]
                if time_from_parent > nodes_ready_times[node]:
                    nodes_ready_times[node] = time_from_parent

        clusters_ready_times[node_cluster] = max(ready_time, nodes_ready_times[node] + analysis_graph[node].duration) 

    return max(clusters_ready_times)

print('Final merging is done: ' + str( time.time() - t0 ))
t0 = time.time()

nodes_levels = get_nodes_levels_dsc(graph,edges_weights, False, nodes_groups)

makespan = calc_finish_time(len(nodes_groups), nodes_groups, nodes_levels, graph)

print('finish time: ' + str(makespan))