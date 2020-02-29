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
no_of_desired_groups = 2
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

nodes_weighted_levels = get_nodes_weighted_levels(rev_graph, edges_weights)


for node, weighted_level in nodes_weighted_levels.items():
    heapq.heappush(free_nodes, (-weighted_level, node))

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
            del rev_nodes_in_degrees[node]
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
nodes_weighted_levels = get_nodes_weighted_levels(graph= graph, grouped= True, edges_weights = edges_weights, nodes_groups= nodes_paths_mapping, is_rev= False, _nodes_in_degrees= nodes_in_degrees)
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

    to_be_merged_groups_sorting_criteria, to_be_merged_groups_weights, to_be_merged_groups_min_levels, to_be_merged_groups, to_be_merged_groups_max_levels, to_be_merged_groups_tasks_per_levels = \
        (list(t) for t in zip(*sorted(zip(to_be_merged_groups_sorting_criteria, to_be_merged_groups_weights, to_be_merged_groups_min_levels, to_be_merged_groups, to_be_merged_groups_max_levels, to_be_merged_groups_tasks_per_levels), reverse=True)))
    cntt = 0
    heaviest_final_group_index = 0
    max_sum = 0
    first_balancing_wave_group_indices = []
    first_balancing_wave_groups_lengths = []
    first_balancing_wave_groups_candidate_destinations = []
    for i in range(0, no_of_desired_groups):
        current_sum = getsum(work_trees[i], no_of_levels - 1)
        if current_sum > max_sum:
            max_sum = current_sum
            heaviest_final_group_index = i

    indx = 0
    paths_lengths = paths_lengths[:-no_of_desired_groups]
    average_path_len = math.ceil(sum(paths_lengths) / len(paths_lengths))
    std_path_len = statistics.stdev(paths_lengths)
    fully_contaied_map = {}

    for tbg in to_be_merged_groups:
        if len(tbg) > min(round(average_path_len + std_path_len), no_of_levels):
            start_node = tbg[0]
            end_node = tbg[-1]
            end_node_children = graph[end_node]

            fully_contaied = True 
            first_child_group = (nodes_groups[end_node_children[0]] if end_node_children[0] != sink_node_name else -2)
            if first_child_group != -2:
                for child in end_node_children:
                    if child != sink_node_name and nodes_groups[child] != first_child_group:
                        fully_contaied = False
                        break
            if fully_contaied:
                if first_child_group == -2:
                    first_child_group = (nodes_groups[rev_graph[start_node][0]] if nodes_groups[rev_graph[start_node][0]] != source_node_name else -2)
                if first_child_group != -2:
                    for parent in rev_graph[start_node]:
                        if parent != source_node_name and nodes_groups[parent] != first_child_group:
                            fully_contaied = False

            to_be_merged_group_weight = to_be_merged_groups_weights[indx]
            if fully_contaied:# != heaviest_final_group_index:#directly_connected_to_heaviest and (snk_direct_connection or src_direct_connection):
                if first_child_group != -2 and first_child_group != heaviest_final_group_index:
                    sum_in_targeted_levels_i = getsum(\
                        work_trees[first_child_group], to_be_merged_groups_earliest_sink_levels[indx]) - getsum(work_trees[first_child_group], \
                            to_be_merged_groups_latest_src_levels[indx])
                    
                    sum_in_targeted_levels_0 = getsum(\
                        work_trees[heaviest_final_group_index], to_be_merged_groups_earliest_sink_levels[indx]) - getsum(work_trees[heaviest_final_group_index], \
                            to_be_merged_groups_latest_src_levels[indx])

                    if sum_in_targeted_levels_0 - sum_in_targeted_levels_i > to_be_merged_group_weight:
                        first_balancing_wave_group_indices.append(indx)
                        first_balancing_wave_groups_lengths.append(
                            to_be_merged_groups_lengths[indx])
                        first_balancing_wave_groups_candidate_destinations.append(
                            first_child_group)
                else:
                    for i in range(0, no_of_desired_groups):
                        if i == first_child_group:
                            continue
                        sum_in_targeted_levels_i = getsum(
                            work_trees[i], analysis_graph[end_node].level) - getsum(work_trees[i], analysis_graph[start_node].level)
                        sum_in_targeted_levels_0 = getsum(work_trees[heaviest_final_group_index], analysis_graph[end_node].level) - \
                            getsum(work_trees[heaviest_final_group_index],
                                analysis_graph[start_node].level)

                        if sum_in_targeted_levels_0 - sum_in_targeted_levels_i > to_be_merged_group_weight:
                            first_balancing_wave_group_indices.append(indx)
                            first_balancing_wave_groups_lengths.append(
                                to_be_merged_groups_lengths[indx])
                            first_balancing_wave_groups_candidate_destinations.append(
                                i)
                            break
        indx += 1
    
    first_balancing_wave_groups_lengths, first_balancing_wave_group_indices, first_balancing_wave_groups_candidate_destinations = \
        (list(t) for t in zip(*sorted(zip(first_balancing_wave_groups_lengths, first_balancing_wave_group_indices, first_balancing_wave_groups_candidate_destinations), \
            reverse=True)))

    for first_balancing_wave_candidate_i in range(0, len(first_balancing_wave_group_indices)):
        current_candidate_indx = first_balancing_wave_group_indices[
            first_balancing_wave_candidate_i]
        current_destination = first_balancing_wave_groups_candidate_destinations[
            first_balancing_wave_candidate_i]
        current_candidate = to_be_merged_groups[current_candidate_indx]
        terminal_comms = 0
        start_node = current_candidate[0]
        end_node = current_candidate[-1]
        end_node_children = graph[end_node]
        earliest_sink_evel = math.inf
        for child in end_node_children:
            if analysis_graph[child].level < earliest_sink_evel:
                earliest_sink_evel = analysis_graph[child].level
            if nodes_groups[child] == heaviest_final_group_index:
                terminal_comms = max(
                    terminal_comms, edges_weights[end_node][child])

        for parent in rev_graph[start_node]:
            if nodes_groups[parent] == heaviest_final_group_index:
                terminal_comms += edges_weights[parent][start_node]

        merge = False
        while not merge and current_destination < no_of_desired_groups:
            sum_in_targeted_levels_i = getsum(\
                        work_trees[current_destination], to_be_merged_groups_earliest_sink_levels[current_destination]) - getsum(work_trees[current_destination], \
                            to_be_merged_groups_latest_src_levels[current_destination])
                    
            sum_in_targeted_levels_0 = getsum(\
                work_trees[heaviest_final_group_index], to_be_merged_groups_earliest_sink_levels[current_destination]) - getsum(work_trees[heaviest_final_group_index], \
                    to_be_merged_groups_latest_src_levels[current_destination])

            if (sum_in_targeted_levels_0 - sum_in_targeted_levels_i) > (terminal_comms + to_be_merged_groups_weights[current_candidate_indx]):
                merge = True
                break
            current_destination += 1

        if merge:
            print(to_be_merged_groups_lengths[current_candidate_indx])
            merge_min_level = min(
                to_be_merged_groups_min_levels[current_candidate_indx], final_groups_min_levels[current_destination])
            merge_max_level = max(
                to_be_merged_groups_max_levels[current_candidate_indx], final_groups_max_levels[current_destination])
            final_groups_weights[current_destination] += to_be_merged_groups_weights[current_candidate_indx]
            final_groups[current_destination] += current_candidate
            final_groups_min_levels[current_destination] = merge_min_level
            final_groups_max_levels[current_destination] = merge_max_level
            merge_src_levels_tasks = to_be_merged_groups_tasks_per_levels[current_candidate_indx]

            for node in current_candidate:
                nodes_groups[node] = current_destination

            for level, tasks_sum in merge_src_levels_tasks.items():
                final_groups_work_per_levels[current_destination][level] += tasks_sum
                updatebit(work_trees[current_destination],
                          no_of_levels, level, tasks_sum)
    
    indx = 0
    for tbg in to_be_merged_groups:
        start_node = tbg[0]
        end_node = tbg[-1]
        end_node_children = graph[end_node]

        fully_contaied = True 
        first_child_group = (nodes_groups[end_node_children[0]] if end_node_children[0] != sink_node_name else -2)
        if first_child_group != -2:
            for child in end_node_children:
                if child != sink_node_name and nodes_groups[child] != first_child_group:
                    fully_contaied = False
                    break
        if fully_contaied:
            if first_child_group == -2:
                first_child_group = (nodes_groups[rev_graph[start_node][0]] if nodes_groups[rev_graph[start_node][0]] != source_node_name else -2)
            if first_child_group != -2:
                for parent in rev_graph[start_node]:
                    if parent != source_node_name and nodes_groups[parent] != first_child_group:
                        fully_contaied = False

        if fully_contaied and first_child_group >= 0:
            heaviest_final_group_index = 0
            max_sum = 0
            for i in range(0, no_of_desired_groups):
                current_sum = getsum(work_trees[i], no_of_levels - 1)
                if current_sum > max_sum:
                    max_sum = current_sum
                    heaviest_final_group_index = i
            
            if heaviest_final_group_index != first_child_group:
                sum_in_targeted_levels_i = getsum(\
                            work_trees[first_child_group], to_be_merged_groups_earliest_sink_levels[indx]) - getsum(work_trees[first_child_group], \
                                to_be_merged_groups_latest_src_levels[first_child_group])
                        
                sum_in_targeted_levels_0 = getsum(\
                    work_trees[heaviest_final_group_index], to_be_merged_groups_earliest_sink_levels[indx]) - getsum(work_trees[heaviest_final_group_index], \
                        to_be_merged_groups_latest_src_levels[indx])
                        
                if to_be_merged_groups_weights[indx] + sum_in_targeted_levels_i <= sum_in_targeted_levels_0:
                    print('dddddddddddd')

            """ if fully_contaied and first_child_group == 1 and analysis_graph[tbg[0]].level > 1:
                print('---' + str(len(tbg)) ) """
        

        indx += 1

print('Initial merging is done: ' + str(time.time() - t0))
t0 = time.time()

""" indx = 0
for tbg in to_be_merged_groups:
    #if len(tbg) > min(round(average_path_len + std_path_len), max(math.sqrt(no_of_levels), average_path_len)):
    if nodes_groups[tbg[0]] == -1:
        start_node = tbg[0]
        end_node = tbg[-1]
        end_node_children = graph[end_node]

        directly_connected_to_heaviest = True
        src_direct_connection = False
        snk_direct_connection = False
        for child in end_node_children:
            if nodes_groups[child] == 1 or child == sink_node_name:
                src_direct_connection = True
            if nodes_groups[child] != -1 and nodes_groups[child] != 1 and child != sink_node_name:
                directly_connected_to_heaviest = False

        if directly_connected_to_heaviest:
            for parent in rev_graph[start_node]:
                if nodes_groups[parent] == 1 or parent == source_node_name:
                    snk_direct_connection = True
                if nodes_groups[parent] != -1 and nodes_groups[parent] != 1 and parent != source_node_name:
                    directly_connected_to_heaviest = False
        
        if directly_connected_to_heaviest and (snk_direct_connection and src_direct_connection) and to_be_merged_groups_lengths[indx] > 0:
            print(to_be_merged_groups_lengths[indx])
    
    indx += 1 """

""" for to_be_merged_group in to_be_merged_groups:
    if len(graph[to_be_merged_group[-1]]) > 1 or len(rev_graph[to_be_merged_group[0]]) > 1:
        print(len(to_be_merged_group)) """
""" levels_work = {}
for level in range(0, no_of_levels):
    levels_work[level] = [0] * 2

for node in all_nodes.keys():
    if nodes_groups[node] != -1:
        levels_work[analysis_graph[node].level][nodes_groups[node]] += 1

for level, work in levels_work.items():
    print(str(level) + '::' + str(work[0]) + '::' +str(work[1]) )  """

to_be_merged_groups_lengths, to_be_merged_groups_indices = \
        (list(t) for t in zip(*sorted(zip(to_be_merged_groups_lengths, to_be_merged_groups_indices), reverse=True)))
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

print('Final merging is done: ' + str( time.time() - t0 ))
t0 = time.time()      
#post processing paths switching:
# work destribution among levels:
nodes_groups[sink_node_name] = 0
total_swapping_gain = 0
initial_groups_no = len(initial_groups)
initial_groups_indices = []
initial_groups_latest_sorces_levels = []
initial_groups_earliest_sink_levels = []
containing_groups_indices = []
already_swapped = {}
swap_groups_sorting_criteria = []

initial_group_indx = 0
len_of_the_smallest_main_group_candidate = len(initial_groups[-no_of_desired_groups])
for initial_group in initial_groups:
    if len(initial_group) >= len_of_the_smallest_main_group_candidate:
        break
    totally_contained = True
    start_node = initial_group[0]
    end_node = initial_group[-1]
    end_node_children = graph[end_node]
    first_child_group = nodes_groups[end_node_children[0]]
    for child in end_node_children:
        if nodes_groups[child] != first_child_group:
            totally_contained = False
            break

    for parent in rev_graph[start_node]:
        if nodes_groups[parent] != first_child_group:
            totally_contained = False
            break
    
    if first_child_group == nodes_groups[start_node]:
        totally_contained = False

    if totally_contained:
        initial_groups_indices.append(initial_group_indx)
        containing_groups_indices.append(first_child_group)

        min_child_end_level = math.inf
        for child in end_node_children:
            current_child_level = analysis_graph[child].level
            if current_child_level < min_child_end_level:
                min_child_end_level = current_child_level

        initial_groups_earliest_sink_levels.append(min_child_end_level)
        initial_groups_latest_sorces_levels.append(analysis_graph[initial_group[0]].level - 1)

        swap_groups_sorting_criteria.append((initial_groups_earliest_sink_levels[-1] - initial_groups_latest_sorces_levels[-1]) * -1)

    initial_group_indx += 1

if swap_groups_sorting_criteria:
    swap_groups_sorting_criteria, initial_groups_latest_sorces_levels, initial_groups_earliest_sink_levels, initial_groups_indices, containing_groups_indices = (list(t) for t in zip(
            *sorted(zip(swap_groups_sorting_criteria, initial_groups_latest_sorces_levels, initial_groups_earliest_sink_levels, initial_groups_indices, containing_groups_indices))))

no_of_swap_groups = len(initial_groups_indices)
containing_group_levels_work_in_swap_levels = [0] * no_of_swap_groups
swap_groups_final_group_levels_work_in_swap_levels = [0] * no_of_swap_groups
comm_with_containing_groups = [0] * no_of_swap_groups
comm_with_its_groups = [0] * no_of_swap_groups
swap_groups_final_groups = [0] * no_of_swap_groups

for group_indx in range(no_of_swap_groups - 2, -1, -1):
    swap_group = initial_groups[group_indx]
    start_node = swap_group[0]
    swap_group_final_group = nodes_groups[start_node]
    swap_group_containing_group = containing_groups_indices[group_indx]
    swap_groups_final_groups[group_indx] = swap_group_final_group

    swap_group_containing_group_levels_work = getsum(work_trees[swap_group_containing_group], initial_groups_earliest_sink_levels[group_indx] - 1) - \
        getsum(work_trees[swap_group_containing_group], initial_groups_latest_sorces_levels[group_indx] + 1)

    swap_group_final_group_levels_work = getsum(work_trees[swap_group_final_group], initial_groups_earliest_sink_levels[group_indx] - 1) - \
        getsum(work_trees[swap_group_final_group], initial_groups_latest_sorces_levels[group_indx] + 1)

    containing_group_levels_work_in_swap_levels[group_indx] = swap_group_containing_group_levels_work
    swap_groups_final_group_levels_work_in_swap_levels[group_indx] = swap_group_final_group_levels_work

    comm_with_containing_group = 0
    comm_with_its_group = 0

    for node in swap_group:
        comm_with_children_in_containing_group = 0
        comm_with_children_in_its_group = 0
        for child in graph[node]:
            if child not in swap_group:
                if nodes_groups[child] == swap_group_containing_group:
                    comm_with_children_in_containing_group = max(comm_with_children_in_containing_group, edges_weights[node][child])
                elif nodes_groups[child] == swap_group_final_group:
                    comm_with_children_in_its_group = max(comm_with_children_in_its_group, edges_weights[node][child])

        comm_with_containing_group += comm_with_children_in_containing_group
        comm_with_its_group += comm_with_children_in_its_group

        for parent in rev_graph[node]:
            if parent not in swap_group:
                if nodes_groups[parent] == swap_group_containing_group:
                    comm_with_containing_group += edges_weights[parent][node]
                elif nodes_groups[parent] == swap_group_final_group:
                    comm_with_its_group += edges_weights[parent][node]

    comm_with_containing_groups[group_indx] = comm_with_containing_group
    comm_with_its_groups[group_indx] = comm_with_its_group
    group_indx += 1

print('bbbbbbbbbbbbbbbbb')
print(no_of_swap_groups)
print('Refinement_1_1 is done: ' + str( time.time() - t0 ))
t0 = time.time()

for to_be_swapped_group_indx in range(0, no_of_swap_groups - 1):
    to_be_swapped_group_end_level = initial_groups_earliest_sink_levels[to_be_swapped_group_indx]
    to_be_swapped_group_final_group_indx = swap_groups_final_groups[to_be_swapped_group_indx]
    to_be_swapped_group_containing_group_indx = containing_groups_indices[to_be_swapped_group_indx]
    to_be_swapped_group_work = groups_weights[initial_groups_indices[to_be_swapped_group_indx]]
    to_be_swapped_group_containing_group_work = containing_group_levels_work_in_swap_levels[to_be_swapped_group_indx]
    to_be_swapped_group_final_group_work = containing_group_levels_work_in_swap_levels[to_be_swapped_group_final_group_indx]

    max_swapping_gain = 0
    swapping_candidate_indx = -1
    swap_with_group_indx = to_be_swapped_group_indx + 1
    swap_with_group_end_level = initial_groups_earliest_sink_levels[swap_with_group_indx]
    while swap_with_group_end_level <= to_be_swapped_group_end_level and swap_with_group_indx < no_of_swap_groups:
        if swap_with_group_indx not in already_swapped:
            swap_with_group_final_group_indx = swap_groups_final_groups[swap_with_group_indx]
            swap_with_group_containing_group_indx = containing_groups_indices[swap_with_group_indx]
            if swap_with_group_containing_group_indx ==  to_be_swapped_group_final_group_indx and \
                swap_with_group_final_group_indx == to_be_swapped_group_containing_group_indx:

                swap_with_group_work = groups_weights[initial_groups_indices[swap_with_group_indx]]
                swap_with_group_containing_group_work = containing_group_levels_work_in_swap_levels[swap_with_group_indx]

                current_max_time = max(\
                    comm_with_containing_groups[to_be_swapped_group_indx] + swap_with_group_containing_group_work, \
                            comm_with_containing_groups[swap_with_group_indx] + to_be_swapped_group_containing_group_work)
                
                an_alternative_max_time = max(comm_with_its_groups[to_be_swapped_group_indx] + to_be_swapped_group_work + to_be_swapped_group_containing_group_work - swap_with_group_work, \
                    comm_with_its_groups[swap_with_group_indx] + swap_with_group_work + swap_with_group_containing_group_work - to_be_swapped_group_work)
                
                swapping_gain = current_max_time - an_alternative_max_time
                if swapping_gain > max_swapping_gain:
                    max_swapping_gain = swapping_gain
                    swapping_candidate_indx = swap_with_group_indx

        swap_with_group_indx += 1
        if swap_with_group_indx < no_of_swap_groups:
            swap_with_group_end_level = initial_groups_earliest_sink_levels[swap_with_group_indx]

    if swapping_candidate_indx == -1:
        current_time = max(comm_with_containing_groups[to_be_swapped_group_indx] + to_be_swapped_group_final_group_work, \
            to_be_swapped_group_containing_group_work)
        an_alternative_time = max(comm_with_its_groups[to_be_swapped_group_indx] + to_be_swapped_group_work + \
            to_be_swapped_group_containing_group_work,to_be_swapped_group_final_group_work - to_be_swapped_group_work)
        if an_alternative_time < current_time:
            to_be_swapped_group = initial_groups[initial_groups_indices[to_be_swapped_group_indx]]
            for node in to_be_swapped_group:
                nodes_groups[node] = to_be_swapped_group_containing_group_indx
                node_duration = analysis_graph[node].duration
                node_level = analysis_graph[node].level
                final_groups_work_per_levels[to_be_swapped_group_containing_group_indx][node_level] += node_duration
                updatebit(work_trees[to_be_swapped_group_containing_group_indx], no_of_levels, node_level, node_duration) 
                final_groups_work_per_levels[to_be_swapped_group_final_group_indx][node_level] -= node_duration 
                updatebit(work_trees[to_be_swapped_group_final_group_indx], no_of_levels, node_level, -node_duration)
            total_swapping_gain += current_time - an_alternative_time
 
    if swapping_candidate_indx != -1 and swapping_candidate_indx:
        already_swapped[swap_with_group_indx] = 1
        already_swapped[swapping_candidate_indx] = 1
        to_be_swapped_group = initial_groups[initial_groups_indices[to_be_swapped_group_indx]]
        swap_with_group = initial_groups[initial_groups_indices[swapping_candidate_indx]]

        for node in swap_with_group:
            nodes_groups[node] = to_be_swapped_group_final_group_indx
            node_duration = analysis_graph[node].duration
            node_level = analysis_graph[node].level
            final_groups_work_per_levels[to_be_swapped_group_final_group_indx][node_level] +=  node_duration
            updatebit(work_trees[to_be_swapped_group_final_group_indx], no_of_levels, node_level, node_duration)
            final_groups_work_per_levels[to_be_swapped_group_containing_group_indx][node_level] -= node_duration
            updatebit(work_trees[to_be_swapped_group_containing_group_indx], no_of_levels, node_level, -node_duration) 
        for node in to_be_swapped_group:
            nodes_groups[node] = to_be_swapped_group_containing_group_indx
            node_duration = analysis_graph[node].duration
            node_level = analysis_graph[node].level
            final_groups_work_per_levels[to_be_swapped_group_containing_group_indx][node_level] += node_duration
            updatebit(work_trees[to_be_swapped_group_containing_group_indx], no_of_levels, node_level, node_duration)
            final_groups_work_per_levels[to_be_swapped_group_final_group_indx][node_level] -= node_duration
            updatebit(work_trees[to_be_swapped_group_final_group_indx], no_of_levels, node_level, node_duration)

        total_swapping_gain += max_swapping_gain

print('total swapping gain = ' + str(total_swapping_gain))

print('Refinement_1 is done: ' + str( time.time() - t0 ))
t0 = time.time()

#post processing, switching nodes placement modification:
total_switching_gain = 0
switching_nodes_pure_parents = []
switching_nodes_pure_children = []
#map, helpful to find nodes in a level in O(1)
levels_nodes = [None] * no_of_levels
for node, props in analysis_graph.items():
    if node in all_nodes:
        if levels_nodes[props.level] == None:
            levels_nodes[props.level] = []
        levels_nodes[props.level].append(node)
#start from level 2, since level 0 contains src -added by me- and 1 contains nodes that are not children of any node in the original graph
#exclude the last level since it only contains the sink
for i in range(2, no_of_levels - 1):
    for node in levels_nodes[i]:
        all_chidren_in_one_group = True
        all_parents_in_one_group = True
        children = graph[node]
        first_child_group = nodes_groups[children[0]]
        parents = rev_graph[node]
        first_parent_group = nodes_groups[parents[0]]
        for child_node in children:
            if nodes_groups[child_node] != first_child_group:
                all_chidren_in_one_group = False
                break
        for parent_node in parents:
            if nodes_groups[parent_node] != first_parent_group:
                all_parents_in_one_group = False
                break
        #note: if all chidren are in the same group and all parents in the same group, then all of them will be in the same group and
        # there is no switching, this is because at least one path will be passing through the switching point.
        switching_node_group = nodes_groups[node]
        if all_chidren_in_one_group and (not all_parents_in_one_group) and switching_node_group != first_child_group:
            switching_nodes_pure_children.append(node)
        if all_parents_in_one_group and (not all_chidren_in_one_group) and switching_node_group != first_parent_group:
            switching_nodes_pure_parents.append(node)

nodes_initial_groups = {}
for i in range(0, len(initial_groups)):
    for node in initial_groups[i]:
        nodes_initial_groups[node] = i

for switching_node in switching_nodes_pure_children:
    children_final_group = nodes_groups[graph[switching_node][0]]
    switching_node_group = nodes_groups[switching_node]
    comm_from_its_group = 0
    comm_from_children_group = 0
    some_parent_initial_group = -1
    for parent in rev_graph[switching_node]:
        if nodes_groups[parent] == children_final_group:
            comm_from_children_group += edges_weights[parent][switching_node]
            if some_parent_initial_group == -1:
                some_parent_initial_group = nodes_initial_groups[parent]
        elif nodes_groups[parent] == switching_node_group:
            comm_from_its_group += edges_weights[parent][switching_node]
    
    max_comm_to_children = 0
    for child_node in graph[switching_node]:
        if edges_weights[switching_node][child_node] > max_comm_to_children:
            max_comm_to_children = edges_weights[switching_node][child_node]

    comm_from_children_group += max_comm_to_children
    movement_gain = comm_from_children_group - (comm_from_its_group + analysis_graph[switching_node].duration)

    if movement_gain > 0:
        total_switching_gain += movement_gain
        switching_node_weight = analysis_graph[switching_node].duration
        switching_node_level = analysis_graph[switching_node].level
        final_groups_work_per_levels[nodes_groups[switching_node]][switching_node_level] -= switching_node_weight
        updatebit(work_trees[nodes_groups[switching_node]], no_of_levels, switching_node_level, -switching_node_weight)
        final_groups_work_per_levels[children_final_group][switching_node_level] += switching_node_weight
        updatebit(work_trees[children_final_group], no_of_levels, switching_node_level, switching_node_weight)
        nodes_groups[switching_node] = children_final_group
        switching_node_group_indx = nodes_initial_groups[switching_node]
        
        if switching_node_level > analysis_graph[initial_groups[some_parent_initial_group][-1]].level:
            initial_groups[some_parent_initial_group].append(switching_node)
        else:
            tail_node = initial_groups[some_parent_initial_group].pop(-1)
            initial_groups[some_parent_initial_group].append(switching_node)
            initial_groups[some_parent_initial_group].append(tail_node)

        initial_groups[switching_node_group_indx].remove(switching_node)
        nodes_initial_groups[switching_node] = some_parent_initial_group

for switching_node in switching_nodes_pure_parents:
    parents_final_group = nodes_groups[rev_graph[switching_node][0]]
    switching_node_group = nodes_groups[switching_node]
    some_child_initial_group = -1
    for child in graph[switching_node]:
        if nodes_groups[child] == parents_final_group:
            if some_child_initial_group == -1:
                some_child_initial_group = nodes_initial_groups[child]
                break
    
    if analysis_graph[switching_node].duration <= comm_latency:
        switching_node_weight = analysis_graph[switching_node].duration
        switching_node_level = analysis_graph[switching_node].level
        final_groups_work_per_levels[nodes_groups[switching_node]][switching_node_level] -= switching_node_weight
        updatebit(work_trees[nodes_groups[switching_node]], no_of_levels, switching_node_level, -switching_node_weight)
        final_groups_work_per_levels[parents_final_group][switching_node_level] += switching_node_weight
        updatebit(work_trees[parents_final_group], no_of_levels, switching_node_level, switching_node_weight)
        nodes_groups[switching_node] = parents_final_group
        switching_node_group_indx = nodes_initial_groups[switching_node]
        initial_groups[some_child_initial_group].insert(0, switching_node)
        initial_groups[switching_node_group_indx].remove(switching_node)
        nodes_initial_groups[switching_node] = some_child_initial_group

tmp_initial_groups = initial_groups
initial_groups = []
for group in tmp_initial_groups:
    if len(group) != 0:
        initial_groups.append(group)

print('total_switching_gain = ' + str(total_switching_gain))

print('Refinement is done: ' + str( time.time() - t0 ))
t0 = time.time()
#handle collocation groups:
for collocation_group in collocations:
    final_groups_indices = [i for i in range(0, no_of_desired_groups)]
    group_outside_comms = [0] * no_of_desired_groups
    final_groups_criteria = [0] * no_of_desired_groups

    for collocation_node in collocation_group:
        final_groups_criteria[nodes_groups[collocation_node]] += analysis_graph[collocation_node].duration
        max_comms = [0] * no_of_desired_groups
        for adj_node in graph[collocation_node]:
            if adj_node not in collocation_group:
                adj_node_group = nodes_groups[adj_node]
                if edges_weights[collocation_node][adj_node] > max_comms[adj_node_group]:
                    max_comms[adj_node_group] = edges_weights[collocation_node][adj_node]

        group_outside_comms = [sum(x) for x in zip(group_outside_comms, max_comms)]

        for rev_adj in rev_graph[collocation_node]:
            if rev_adj not in collocation_group:
                group_outside_comms[nodes_groups[rev_adj]] += edges_weights[rev_adj][collocation_node]

    final_groups_criteria = [sum(x) for x in zip(final_groups_criteria, group_outside_comms)]

    target_group_indx = final_groups_criteria.index(max(final_groups_criteria))

    for node in collocation_group:
        nodes_groups[node] = target_group_indx

#memory----------------------------------------------------------------------------------------
nodes_memory = {}
additional_memory = {}
nodes_res_memory = {}
resident_nodes = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_memory[node_name] = int(splitted[1])
        #if '^' + node_name in all_nodes:
        #    nodes_memory['^' + node_name] = int(splitted[1])

with open(in6_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_res_memory[node_name] = int(splitted[1])
        resident_nodes[node_name] = 1

for node in all_nodes:
    if node not in nodes_memory:
        nodes_memory[node] = 0
    if node not in nodes_res_memory:
        nodes_res_memory[node] = 0

def prepare_for_memory_balancing_round(round_no):
    memory_limit_is_exceeded = False
    for node in all_nodes.keys():
        nodes_levels_scheduled[node] = 0

    traversal_queue = []
    heapq.heappush(traversal_queue, (nodes_levels_scheduled[source_node_name], source_node_name))
    groups_times_till_now = [0] * no_of_desired_groups

    while traversal_queue:
        [current_node_start_time, current_node] = heapq.heappop(traversal_queue)
        current_node_end_time = current_node_start_time + analysis_graph[current_node].duration
        groups_times_till_now[nodes_groups[current_node]] += analysis_graph[current_node].duration
        current_node_group = nodes_groups[current_node]

        for adj_node in graph[current_node]:
            adj_node_group = nodes_groups[adj_node]
            nodes_levels_scheduled[adj_node] = \
                max([current_node_end_time + (int(edges_weights[current_node][adj_node]) if current_node_group != nodes_groups[adj_node] else 1), \
                    groups_times_till_now[adj_node_group], nodes_levels_scheduled[adj_node]])
            tmp_nodes_in_degrees[adj_node] -= 1
            if tmp_nodes_in_degrees[adj_node] == 0:
                heapq.heappush(traversal_queue, (nodes_levels_scheduled[adj_node],adj_node))
    
    """ for node in nodes_levels_scheduled.keys():
        nodes_levels_scheduled[node] = analysis_graph[node].level """
    for node in all_nodes.keys():
        parents_last_active_levels[node] = [nodes_levels_scheduled[node]] * no_of_desired_groups
        parents_first_active_levels[node] = [0] * no_of_desired_groups
        parents_last_active_no_ops_levels[node] = nodes_levels_scheduled[node]
        nodes_earliest_parents_levels[node] = nodes_levels_scheduled[node]
        nodes_comms[node] = [0] * no_of_desired_groups
        nodes_parents_levels_to_memory[node] = {}
        nodes_parents_levels_to_nodes_names[node] = {}
        parents_all_active_levels[node] = []
        for group_no in range(0, no_of_desired_groups):
            parents_all_active_levels[node].append([nodes_levels_scheduled[node]])

        for child in graph[node]:
            child_level = nodes_levels_scheduled[child]
            child_group = nodes_groups[child]

            if edges_weights[node][child] >nodes_comms[node][child_group]:
                nodes_comms[node][child_group] = edges_weights[node][child]

            if child not in no_op_nodes and child not in sudo_nodes:
                parents_all_active_levels[node][child_group].append(child_level)
                if child_level > parents_last_active_levels[node][child_group]:
                    parents_last_active_levels[node][child_group] = child_level
        
        for i in range(0, no_of_desired_groups):
            parents_first_active_levels[node][i] = min(parents_all_active_levels[node][i])

        for parent in rev_graph[node]:
            parent_level = nodes_levels_scheduled[parent]
            parent_group = nodes_groups[parent]
            parent_memory = nodes_memory[parent]
            nodes_comms[node][parent_group] += edges_weights[parent][node]
            if parent_memory > 0:
                if parent_level not in nodes_parents_levels_to_memory[node]:
                    nodes_parents_levels_to_memory[node][parent_level] = 0
                    nodes_parents_levels_to_nodes_names[node][parent_level] = []

                nodes_parents_levels_to_memory[node][parent_level] += parent_memory
                nodes_parents_levels_to_nodes_names[node][parent_level].append(parent)
                if parent_level < nodes_earliest_parents_levels[node]:
                    nodes_earliest_parents_levels[node] = parent_level           

    for node, parents in rev_graph.items():
        node_additional_memory = 0
        if node != sink_node_name:
            for parent in parents:
                if nodes_levels_scheduled[node] >= parents_last_active_levels[parent][nodes_groups[node]]:
                    node_additional_memory += nodes_memory[parent]
        
        additional_memory [node] = node_additional_memory
        if node_additional_memory > memory_limit_per_group:
            print(node)
            print(node_additional_memory)
            print('one node additional memory is exceeding the limit')

    groups_non_empty_levels = []
    for i in range(0, no_of_desired_groups):
        groups_non_empty_levels.append({})

    nodes_list = nodes_levels_scheduled.keys()
    scheduled_levels_list = nodes_levels_scheduled.values()

    for node, scheduled_level in nodes_levels_scheduled.items():
        groups_non_empty_levels[nodes_groups[node]][scheduled_level] = 1

    scheduled_levels_list, nodes_list = (list(t) for t in zip(
        *sorted(zip(scheduled_levels_list, nodes_list))))

    commulative_memory_from_parents_to_children = [0] * no_of_desired_groups
    subtract_commulative_memory_at = {}
    visited_levels = {}

    indx = 0
    for level in scheduled_levels_list:
        subtract_commulative_memory_at[level] = [0] * no_of_desired_groups
        if not level in levels_indices_map: 
            levels_indices_map[level] = indx
            indx += 1
    
    levels_ends = {}
    ends_levels = {}
    indx = 0
    for level in scheduled_levels_list:
        if level not in levels_ends:
            levels_ends[level] = [0] * no_of_desired_groups
        levels_ends[level][nodes_groups[nodes_list[indx]]] = indx
        indx += 1
    
    for level, ends in levels_ends.items():
        for end in ends:
            ends_levels[end] = level
        
    for collocation_group in collocations:
        collocation_final_group = nodes_groups[collocation_group[0]]
        if collocation_final_group == round_no:
            group_inside_comms = 0
            group_outside_comms = 0
            collocation_group_memory = 0
            a_var_node = ''
            for collocation_node in collocation_group:
                if collocation_node in var_nodes:
                    collocation_group_memory += nodes_memory[collocation_node]
                    a_var_node = collocation_node
                
                tmp_inside = 0
                tmp_outside = 0
                for adj_node in graph[collocation_node]:
                    if nodes_groups[adj_node] != collocation_final_group:
                        tmp_outside = max(tmp_outside, edges_weights[collocation_node][adj_node]) 
                    elif adj_node not in collocation_group:
                        tmp_inside = max(tmp_inside, edges_weights[collocation_node][adj_node])

                group_inside_comms += tmp_inside
                group_outside_comms += tmp_outside

                for rev_adj in rev_graph[collocation_node]:
                    if nodes_groups[rev_adj] != collocation_final_group:
                        group_outside_comms += edges_weights[rev_adj][collocation_node] 
                    if rev_adj not in collocation_group:
                        group_inside_comms += edges_weights[rev_adj][collocation_node]

            nodes_mem_potentials[a_var_node] = collocation_group_memory
            nodes_comms[a_var_node] = [group_outside_comms] * no_of_desired_groups
            nodes_comms[a_var_node][collocation_final_group] = group_inside_comms
            heapq.heappush(collocated_nodes_heap, ( (group_inside_comms + 1) / (collocation_group_memory + 0.1) , a_var_node ) )
    
    for node in resident_nodes:
        if nodes_groups[node] == round_no and node not in var_nodes and node not in ref_nodes:
            heapq.heappush(resident_nodes_heap, ((nodes_comms[node][nodes_groups[node]] + 1) / (nodes_memory[node] + 0.1), node) )
            nodes_mem_potentials[node] = nodes_memory[node]


    final_groups_memory_consumptions = np.zeros((no_of_desired_groups, len(levels_indices_map)), dtype= np.int64)

    residual_memories = [0] * no_of_desired_groups
    for node in var_nodes.keys():
        residual_memories[nodes_groups[node]] += nodes_memory[node]

    """ print('-----------------')
    for group_num in range(0, no_of_desired_groups):
        print(residual_memories[group_num] / (1024 * 1024 * 1024))
    print('-----------------') """

    for node, mem in nodes_res_memory.items():
        if node not in ref_nodes and node not in var_nodes:
            residual_memories[nodes_groups[node]] += mem

    for group_num in range(0, no_of_desired_groups):
        print(residual_memories[group_num] / (1024 * 1024 * 1024))
    print('-----------------')

    for group_num in range(0, no_of_desired_groups):
        final_groups_memory_consumptions[group_num][:] += residual_memories[group_num]
        

    """ for i in range(0, len(scheduled_levels_list)):
        node = nodes_list[i]
        if node not in var_nodes and node not in ref_nodes:
            res_mem = nodes_res_memory[node]
            node_group = nodes_groups[node]
            node_level = scheduled_levels_list[i]
            residual_memories[node_group] += res_mem
            for group_num in range(0, no_of_desired_groups):
                final_groups_memory_consumptions[group_num][levels_indices_map[node_level]] = residual_memories[group_num] """

    """ max_mem = 0
    for level in scheduled_levels_list:
        _str = '' + str(level) + '::'
        sum_in_level = 0
        prntt = False
        for grpp in range(0, no_of_desired_groups):
            sum_in_level += final_groups_memory_consumptions[grpp][levels_indices_map[level]]
            if final_groups_memory_consumptions[grpp][levels_indices_map[level]] / (1024 * 1024 * 1024) > 20:
                prntt = True
            _str += str(final_groups_memory_consumptions[grpp][levels_indices_map[level]] / (1024 * 1024 * 1024)) + ' '
        if sum_in_level > max_mem:
            max_mem = sum_in_level
        if prntt:
            print(_str)  """   

    for node_indx in range(0, len(nodes_list)):
        node = nodes_list[node_indx]
        node_scheduled_level = scheduled_levels_list[node_indx]
        nodes_indices_map[node] = node_indx

        node_group = nodes_groups[node]
        node_memory = nodes_memory[node]

        final_groups_memory_consumptions[node_group][levels_indices_map[node_scheduled_level]] += (node_memory if (node not in resident_nodes and node not in ref_nodes) else 0)
        if node_scheduled_level not in visited_levels or visited_levels[node_scheduled_level][node_group] == 0:
            final_groups_memory_consumptions[node_group][levels_indices_map[node_scheduled_level]] += commulative_memory_from_parents_to_children[node_group] 

        if final_groups_memory_consumptions[node_group][levels_indices_map[node_scheduled_level]] > memory_limit_per_group:
            memory_limit_is_exceeded = True
            """ print(final_groups_memory_consumptions[node_group][levels_indices_map[node_scheduled_level]])
            print(node_scheduled_level)
            print(node_group)
            print('----------------------') """
        
        if node_indx in ends_levels and node_scheduled_level == ends_levels[node_indx]:
            commulative_memory_from_parents_to_children[node_group] -= subtract_commulative_memory_at[node_scheduled_level][node_group]

        for group_num in range(0, no_of_desired_groups):
            if node_scheduled_level not in groups_non_empty_levels[group_num] and node_scheduled_level not in visited_levels:

                final_groups_memory_consumptions[group_num][levels_indices_map[node_scheduled_level]] += commulative_memory_from_parents_to_children[group_num] 
            
            level = parents_last_active_levels[node][group_num]
            if level > node_scheduled_level and (group_num != node_group or (node not in ref_nodes and node not in resident_nodes) ):
                if node != sink_node_name and graph[node][0] != sink_node_name:
                    commulative_memory_from_parents_to_children[group_num] += node_memory
                    subtract_commulative_memory_at[level][group_num] += node_memory
                else:
                    commulative_memory_from_parents_to_children[node_group] += node_memory

        if node_scheduled_level not in visited_levels:
            visited_levels[node_scheduled_level] = [0] * no_of_desired_groups
        visited_levels[node_scheduled_level][node_group] = 1
    print([final_groups_memory_consumptions[0][0]])
    print([final_groups_memory_consumptions[0][1]])
    return [final_groups_memory_consumptions, nodes_list, scheduled_levels_list, memory_limit_is_exceeded]



merged = False
non_mergable_nodes = []
for i in range(0, no_of_desired_groups):
    non_mergable_nodes.append([])

for group_no in range(0, no_of_desired_groups):
    parents_last_active_levels = {}
    parents_first_active_levels = {}
    parents_last_active_no_ops_levels = {}
    parents_all_active_levels = {}
    nodes_parents_levels_to_memory = {}
    nodes_parents_levels_to_nodes_names = {}
    nodes_earliest_parents_levels = {}
    nodes_comms = {}
    nodes_levels_scheduled = {}
    tmp_nodes_in_degrees = copy.deepcopy(nodes_in_degrees)
    final_groups_memory_consumptions = None
    nodes_indices_map = {}
    levels_indices_map = {}
    nodes_mem_potentials = {}
    collocated_nodes_heap = []
    resident_nodes_heap = []
    ineffective_resident_nodes = {}
    ineffective_resident_nodes_heap = []
    [final_groups_memory_consumptions, nodes_list, scheduled_levels_list, memory_limit_is_exceeded] = prepare_for_memory_balancing_round(group_no)
    
    bad_levels_for_print = []

    if memory_limit_is_exceeded:
        print('limit is exceeded')
        """ max_mem = 0
        for level in levels_indices_map.keys():
            _str = '' + str(level) + '::'
            sum_in_level = 0
            prntt = False
            for grpp in range(0, no_of_desired_groups):
                sum_in_level += final_groups_memory_consumptions[grpp][levels_indices_map[level]]
                if final_groups_memory_consumptions[grpp][levels_indices_map[level]] > 80  * (1024 * 1024 * 1024):
                    bad_levels_for_print.append(level)
                    prntt = True
                _str += str(final_groups_memory_consumptions[grpp][levels_indices_map[level]] / (1024 * 1024 * 1024)) + ' '
            if sum_in_level > max_mem:
                max_mem = sum_in_level
            if prntt:
                print(_str) """
    else:
        """ max_mem = 0
        for level in scheduled_levels_list:
            _str = '' + str(level) + '::'
            sum_in_level = 0
            prntt = False
            for grpp in range(0, no_of_desired_groups):
                sum_in_level += final_groups_memory_consumptions[grpp][levels_indices_map[level]]
                if final_groups_memory_consumptions[grpp][levels_indices_map[level]] / (1024 * 1024 * 1024) > 25:
                    prntt = True
                _str += str(final_groups_memory_consumptions[grpp][levels_indices_map[level]] / (1024 * 1024 * 1024)) + ' '
            if sum_in_level > max_mem:
                max_mem = sum_in_level
            if prntt:
                print(_str) """
        break
    
    node_index = len(nodes_list) - 1
    nodes_heap = []
    no_op_nodes_heap = []
    criteria_heap= []
    big_nodes = [] # nodes with memory potential more than the overflow
    removed_nodes = {}
    visited_nodes = {}
    merged_nodes = {}
    replicated_nodes = {}
    nodes_active_parents = {}

    while node_index > 0:
        node = nodes_list[node_index]
        node_group = nodes_groups[node]
        scheduled_level = scheduled_levels_list[node_index]
        
        if node_group == group_no:
            if not (node in sudo_nodes or node in no_op_nodes or node in ref_nodes or node in var_nodes):
                heapq.heappush(nodes_heap, (-nodes_earliest_parents_levels[node], node))
                visited_nodes[node] = 1

                candidate_node_mem_potential = 0
                for level, mem in nodes_parents_levels_to_memory[node].items():
                    if level <= scheduled_level:
                        for parent in nodes_parents_levels_to_nodes_names[node][level]:
                            if (nodes_groups[parent] == node_group or level < scheduled_level) or not(nodes_groups[parent] == group_no and parent in resident_nodes):
                                if node not in nodes_active_parents:
                                    nodes_active_parents[node] = []
                                if scheduled_level == parents_last_active_levels[parent][node_group]:
                                    candidate_node_mem_potential += nodes_memory[parent]
                                    nodes_active_parents[node].append(parent)

                if node_name in resident_nodes:
                    candidate_node_mem_potential += nodes_res_memory[node_name]

                nodes_mem_potentials[node] = candidate_node_mem_potential
                heapq.heappush(criteria_heap, ( (nodes_comms[node][node_group] + 1) / (candidate_node_mem_potential + 0.1), node) )

            while nodes_heap:
                heap_top = heapq.heappop(nodes_heap)
                if abs(heap_top[0]) <= scheduled_level:
                    heapq.heappush(nodes_heap, heap_top)
                    break
                removed_nodes[heap_top[1]] = 1

            overflow = final_groups_memory_consumptions[node_group][levels_indices_map[scheduled_level]] - memory_limit_per_group
            if overflow > 0:
                """ print('crh:' + str(len(criteria_heap)))

                _str = '' + str(scheduled_level) + '::'
                for grpp in range(0, no_of_desired_groups):
                    _str += str(final_groups_memory_consumptions[scheduled_level][grpp] / (1024 * 1024 * 1024)) + ' '
                print(_str)

                print(overflow) """
                while overflow > 0 and (criteria_heap or big_nodes or resident_nodes_heap or collocated_nodes_heap):
                    from_big_nodes = False
                    heaps_are_empty = False
                    
                    while ineffective_resident_nodes_heap:
                        if -ineffective_resident_nodes_heap[0][0] > scheduled_level:
                            res_node = ineffective_resident_nodes_heap[0][1]
                            heapq.heappush(resident_nodes_heap, (nodes_res_memory[res_node], res_node) )
                            heapq.heappop(ineffective_resident_nodes_heap)
                            del ineffective_resident_nodes[res_node]
                        else:
                            break

                    if criteria_heap or collocated_nodes_heap or resident_nodes_heap:
                        criteria_val = math.inf
                        if criteria_heap:
                            candidate_node = criteria_heap[0]
                            criteria_val = candidate_node[0]
                        if collocated_nodes_heap:
                            alt_collocated = collocated_nodes_heap[0]
                            if alt_collocated[0] < criteria_val:
                                candidate_node = alt_collocated
                                criteria_val = candidate_node[0]
                        alt_resident = None
                        while resident_nodes_heap:
                            alt_resident = resident_nodes_heap[0]
                            alt_resident_name = alt_resident[1]
                            alt_resident_level = nodes_levels_scheduled[alt_resident_name]
                            if parents_last_active_levels[alt_resident_name][group_no] >= scheduled_level and alt_resident_level < scheduled_level:
                                ineffective_resident_nodes[alt_resident_name] = parents_all_active_levels[alt_resident_name][group_no]
                                heapq.heappush(ineffective_resident_nodes_heap, (-alt_resident_level, alt_resident_name) )
                            else:
                                if alt_resident[0] < criteria_val:
                                    candidate_node = alt_resident
                                break
                            alt_resident = None
                            heapq.heappop(resident_nodes_heap)
                        
                        if collocated_nodes_heap and  candidate_node == alt_collocated:
                            heapq.heappop(collocated_nodes_heap)
                        elif resident_nodes_heap and  candidate_node == alt_resident:
                            heapq.heappop(resident_nodes_heap)
                        elif criteria_heap and candidate_node == criteria_heap[0]:
                            heapq.heappop(criteria_heap)

                    else:
                        candidate_node = heapq.heappop(big_nodes)
                        from_big_nodes = True
                        heaps_are_empty = True

                    node_name = candidate_node[1]
                    if nodes_groups[node_name] != group_no:
                        continue
                    if node_name in replicated_nodes:
                        if replicated_nodes[node_name] == 0:
                            replicated_nodes[node_name] = 1
                        else:
                            continue
                        
                    if nodes_mem_potentials[node_name] > overflow and not heaps_are_empty:
                        heapq.heappush(big_nodes, (nodes_comms[node_name][group_no], node_name))
                    else:
                        if big_nodes and not heaps_are_empty:
                            alternative_candidate = heapq.heappop(big_nodes)
                            if alternative_candidate[0] <= nodes_comms[node_name][group_no]:
                                node_name = alternative_candidate[1]
                                from_big_nodes = True
                                heapq.heappush(criteria_heap, candidate_node)
                            else:
                                heapq.heappush(big_nodes, alternative_candidate)

                        candidate_node_level = nodes_levels_scheduled[node_name] 

                        if node_name in removed_nodes or node_name in merged_nodes or node in non_mergable_nodes:
                            continue

                        node_updated = False
                        parents_to_remove = []
                        if node_name in nodes_active_parents:
                            for parent in nodes_active_parents[node_name]: 
                                if nodes_levels_scheduled[parent] > scheduled_level or nodes_groups[parent] != node_group and not (nodes_groups[parent] == group_no and parent in resident_nodes):
                                    parents_to_remove.append(parent)
                                    nodes_mem_potentials[node_name] -= nodes_memory[parent]
                                    node_updated = True
                            
                            for parent in parents_to_remove:
                                nodes_active_parents[node_name].remove(parent)
                            
                        if node_updated:
                            if from_big_nodes:
                                heapq.heappush(big_nodes, (nodes_comms[node_name][group_no], node_name) )
                            else:
                                heapq.heappush(criteria_heap, ( (nodes_comms[node_name][node_group] + 1)/ (nodes_mem_potentials[node_name] + 0.1), node_name) )
                            continue

                        final_groups_indices = []
                        final_groups_memory_consumptions_in_current_level_inverted = [] # inverted due to reverse sort
                        for i in range(0, no_of_desired_groups):
                            final_groups_indices.append(i)
                            final_groups_memory_consumptions_in_current_level_inverted.append(-final_groups_memory_consumptions[i][levels_indices_map[candidate_node_level]])
                        node_comms = nodes_comms[node_name]
                        #giving priority to the group which this node is communicated with the most in addition to the one having the least memory in the targeted level
                        node_comms, final_groups_memory_consumptions_in_current_level_inverted, final_groups_indices = \
                            (list(t) for t in zip(*sorted(zip(node_comms, final_groups_memory_consumptions_in_current_level_inverted, final_groups_indices), reverse=True)))

                        for final_group_indx in final_groups_indices:
                            merged = False
                            if final_group_indx != node_group:
                                merged = True
                                affected_levels_additional_mems = {}
                                affected_levels_to_subtract_mems = {}
                                ranges_additions = []
                                ranges_subtractions = []
                                current_level_indx = nodes_indices_map[node_name]
                                stop_at_level = nodes_earliest_parents_levels[node_name]
                                value_to_add = 0
                                value_to_subtract = 0
                                levels_to_subtract_at_from_subtract_value = {}
                                levels_to_subtract_at_from_add_value = {}

                                if node_name in nodes_active_parents or node_name in resident_nodes:
                                    for parent in rev_graph[node_name]:
                                        if not(nodes_groups[parent] == group_no and parent in resident_nodes):
                                            parent_memory = nodes_memory[parent]
                                            if parent_memory > 0:
                                                value_to_add += parent_memory
                                                level_to_subtract_at = min(parents_last_active_levels[parent][final_group_indx], candidate_node_level)
                                                if level_to_subtract_at not in levels_to_subtract_at_from_add_value:
                                                    levels_to_subtract_at_from_add_value[level_to_subtract_at] = 0
                                                levels_to_subtract_at_from_add_value[level_to_subtract_at] += parent_memory

                                                value_to_subtract += parent_memory
                                                parents_all_active_levels_assuming_removal = copy.deepcopy(parents_all_active_levels[parent][group_no])
                                                parents_all_active_levels_assuming_removal.remove(candidate_node_level)
                                                level_to_subtract_at =min(max(parents_all_active_levels_assuming_removal), candidate_node_level)
                                                if level_to_subtract_at not in levels_to_subtract_at_from_subtract_value:
                                                    levels_to_subtract_at_from_subtract_value[level_to_subtract_at] = 0
                                                levels_to_subtract_at_from_subtract_value[level_to_subtract_at] += parent_memory

                                    if candidate_node_level not in levels_to_subtract_at_from_add_value:
                                        levels_to_subtract_at_from_add_value[candidate_node_level] = 0

                                    levels_to_subtract_at = levels_to_subtract_at_from_add_value.keys()
                                    values_to_subtract = levels_to_subtract_at_from_add_value.values()
                                    levels_to_subtract_at, values_to_subtract = (list(t) for t in zip(*sorted(zip(levels_to_subtract_at, values_to_subtract), reverse=True)))

                                    for i in range(0, len(levels_to_subtract_at)):
                                        ranges_additions.append([ levels_indices_map[levels_to_subtract_at[i]], value_to_add - values_to_subtract[i]])
                                        value_to_add -= values_to_subtract[i]
                                    #----------------------------------------------
                                    if candidate_node_level not in levels_to_subtract_at_from_subtract_value:
                                        levels_to_subtract_at_from_subtract_value[candidate_node_level] = 0

                                    levels_to_subtract_at = levels_to_subtract_at_from_subtract_value.keys()
                                    values_to_subtract = levels_to_subtract_at_from_subtract_value.values()
                                    levels_to_subtract_at, values_to_subtract = (list(t) for t in zip(*sorted(zip(levels_to_subtract_at, values_to_subtract), reverse=True)))

                                    for i in range(0, len(levels_to_subtract_at)):
                                        ranges_subtractions.append([ levels_indices_map[levels_to_subtract_at[i]], value_to_subtract - values_to_subtract[i]])
                                        value_to_subtract -= values_to_subtract[i]
                                    
                                else:
                                    merged = False

                                if merged:
                                    if node_name in var_nodes:
                                        if np.amax( final_groups_memory_consumptions[final_group_indx][:]) + nodes_mem_potentials[node_name] > memory_limit_per_group:
                                            merged = False
                                        else:
                                            final_groups_memory_consumptions[final_group_indx][:] += nodes_mem_potentials[node_name] 
                                    else:
                                        if node_name in resident_nodes:
                                            resident_memory_first_affected_level = levels_indices_map[ parents_last_active_levels[node_name][final_group_indx] ] + 1
                                            if resident_memory_first_affected_level < len(levels_indices_map):
                                                final_groups_memory_consumptions[final_group_indx][ resident_memory_first_affected_level : len(levels_indices_map) ] += \
                                                    nodes_res_memory[node_name]
                                                if np.amax( final_groups_memory_consumptions[final_group_indx][ : \
                                                    len(scheduled_levels_list)] ) > memory_limit_per_group:
                                                    merged = False
                                                    final_groups_memory_consumptions[final_group_indx][ resident_memory_first_affected_level : len(levels_indices_map) ] \
                                                        -= nodes_res_memory[node_name]

                                            resident_memory_first_affected_level = levels_indices_map[parents_first_active_levels[node_name][final_group_indx]]
                                            if resident_memory_first_affected_level > 0:
                                                final_groups_memory_consumptions[final_group_indx][ 0 : resident_memory_first_affected_level ] += \
                                                    nodes_res_memory[node_name]
                                                
                                                if np.amax( final_groups_memory_consumptions[final_group_indx][ 0 :  ] \
                                                    ) > memory_limit_per_group:
                                                    merged = False
                                                    final_groups_memory_consumptions[final_group_indx][ 0 : resident_memory_first_affected_level ] \
                                                        -= nodes_res_memory[node_name]
    
                                        if final_groups_memory_consumptions[final_group_indx][ levels_indices_map[candidate_node_level]] + nodes_memory[node_name] >\
                                             memory_limit_per_group:
                                            merged = False 
                                        else:
                                            final_groups_memory_consumptions[final_group_indx][ levels_indices_map[candidate_node_level]] += nodes_memory[node_name]
                                    
                                    if merged and node_name not in var_nodes:
                                        roll_back_indx = 0
                                        for i in range(0, len(ranges_additions) - 1):
                                            final_groups_memory_consumptions[final_group_indx][ ranges_additions[i+1][0] + 1 : ranges_additions[i][0] + 1 ] += \
                                                ranges_additions[i][1]
                                            roll_back_indx += 1
                                            #print(str(ranges_additions[i][0]) + ':' + str(ranges_additions[i+1][0]))
                                            if np.amax( final_groups_memory_consumptions[final_group_indx] [ranges_additions[i+1][0] + 1 : ranges_additions[i][0] + 1 ] ) > memory_limit_per_group:
                                                merged = False
                                                for i in range(0, roll_back_indx):
                                                    final_groups_memory_consumptions[final_group_indx][ ranges_additions[i+1][0] + 1 : ranges_additions[i][0] + 1 ] -= \
                                                        ranges_additions[i][1]
                                                break

                                if merged:
                                    merged_nodes[node_name] = 1
                                    nodes_groups[node_name] = final_group_indx

                                    if node_name in var_nodes:
                                        final_groups_memory_consumptions[node_group][:] -= nodes_mem_potentials[node_name]
                                    else:
                                        if node_name in resident_nodes:
                                            resident_memory_first_affected_level = levels_indices_map[parents_last_active_levels[node_name][group_no]] + 1
                                            if resident_memory_first_affected_level < len(levels_indices_map):
                                                final_groups_memory_consumptions[node_group][ resident_memory_first_affected_level : len(levels_indices_map) ] -= \
                                                    nodes_res_memory[node_name]
                                            
                                            resident_memory_first_affected_level = levels_indices_map[parents_first_active_levels[node_name][group_no]]
                                            if resident_memory_first_affected_level > 0:
                                                final_groups_memory_consumptions[node_group][ 0 : resident_memory_first_affected_level ] -= \
                                                    nodes_res_memory[node_name]
                                    
                                        final_groups_memory_consumptions[node_group][ levels_indices_map[candidate_node_level]] -= nodes_memory[node_name]
                                    
                                        for i in range(0, len(ranges_subtractions) - 1):
                                            final_groups_memory_consumptions[node_group][ ranges_subtractions[i+1][0] + 1 : ranges_subtractions[i][0] + 1 ] -= \
                                                ranges_subtractions[i][1]
                                    
                                    if node_name in var_nodes:
                                        collocation_group_indx = nodes_collocation_groups[node_name]
                                        for coll_node in collocations[collocation_group_indx]:
                                            nodes_groups[coll_node] = final_group_indx
                                    
                                    if node_name not in var_nodes:
                                        for child in graph[node_name]:
                                            nodes_comms[child][group_no] -= edges_weights[node_name][child]
                                            nodes_comms[child][final_group_indx] += edges_weights[node_name][child]

                                        for parent in rev_graph[node_name]:
                                            if candidate_node_level >= parents_last_active_levels[parent][final_group_indx]:
                                                parents_last_active_levels[parent][final_group_indx] = candidate_node_level
                                            
                                            parents_all_active_levels[parent][final_group_indx].append(candidate_node_level)
                                            parents_all_active_levels[parent][node_group].remove(candidate_node_level)
                                            
                                            if candidate_node_level == parents_last_active_levels[parent][node_group]:
                                                parents_last_active_levels[parent][node_group] = max(parents_all_active_levels[parent][node_group])
                                                if parent in ineffective_resident_nodes and parents_last_active_levels[parent][node_group] < candidate_node_level:
                                                    heapq.heappush(resident_nodes_heap, ((nodes_comms[parent][node_group] + 1) / (nodes_memory[parent] + 0.1), parent ) )
                                                    del ineffective_resident_nodes[parent]

                                                for child in graph[parent]:
                                                    if nodes_groups[child] == node_group or nodes_groups[child] == final_group_indx:
                                                        nodes_comms[parent][nodes_groups[child]] = max(nodes_comms[parent][nodes_groups[child]], \
                                                            edges_weights[parent][child])
                                                    if nodes_groups[child] == node_group and child in visited_nodes and \
                                                        child not in removed_nodes and \
                                                        nodes_levels_scheduled[child] == parents_last_active_levels[parent][node_group]\
                                                            and not(nodes_groups[parent] == group_no and parent in resident_nodes)\
                                                                and nodes_levels_scheduled[child] != candidate_node_level:

                                                        nodes_active_parents[child].append(parent)
                                                        nodes_mem_potentials[child] += nodes_memory[parent]
                                                        
                                                        heapq.heappush(criteria_heap, ( (nodes_comms[child][node_group] + 1) / (nodes_mem_potentials[child] + 0.1), child) )
                                                        replicated_nodes[child] = 0

                                            for group_num in range(0, no_of_desired_groups):
                                                parents_first_active_levels[parent][group_num] = min(parents_all_active_levels[parent][group_num])
                                    
                                    overflow = final_groups_memory_consumptions[node_group][levels_indices_map[scheduled_level]] - memory_limit_per_group

                                    break
                                else:
                                    non_mergable_nodes[group_no].append(node)
                            
            if overflow > 0:
                print('cannot be addressed')
                #print(analysis_graph[node].level)
                print(overflow/(1024 * 1024 * 1024))
                """ parents_last_active_levels = {}
                parents_first_active_levels = {}
                parents_last_active_no_ops_levels = {}
                parents_all_active_levels = {}
                nodes_parents_levels_to_memory = {}
                nodes_parents_levels_to_nodes_names = {}
                nodes_earliest_parents_levels = {}
                nodes_comms = {}
                nodes_levels_scheduled = {}
                tmp_nodes_in_degrees = copy.deepcopy(nodes_in_degrees)
                final_groups_memory_consumptions = None
                nodes_indices_map = {}
                levels_indices_map = {}
                nodes_mem_potentials = {}
                collocated_nodes_heap = []
                resident_nodes_heap = []
                ineffective_resident_nodes = {}
                ineffective_resident_nodes_heap = []
                [final_groups_memory_consumptions, nodes_list, scheduled_levels_list, memory_limit_is_exceeded] = prepare_for_memory_balancing_round(1)
                max_mem = 0
                for level in levels_indices_map.keys():
                    _str = '' + str(level) + '::'
                    sum_in_level = 0
                    prntt = False
                    for grpp in range(0, no_of_desired_groups):
                        sum_in_level += final_groups_memory_consumptions[grpp][levels_indices_map[level]]
                        if final_groups_memory_consumptions[grpp][levels_indices_map[level]] > 28 * (1024 * 1024 * 1024):
                            prntt = True
                        _str += str(final_groups_memory_consumptions[grpp][levels_indices_map[level]] / (1024 * 1024 * 1024)) + ' '
                    if sum_in_level > max_mem:
                        max_mem = sum_in_level
                    if prntt:
                        print(_str)
 """
                exit()

        node_index -= 1

#[final_groups_memory_consumptions, nodes_list, scheduled_levels_list, memory_limit_is_exceeded] = prepare_for_memory_balancing_round(0)

with open(out1, 'w') as f:
    smm = [0] * (no_of_desired_groups + 1 )
    light_levels_sum = [0] * (no_of_desired_groups + 1)
    cntt = [0] * (no_of_desired_groups + 1)
    count = [0] * (no_of_desired_groups + 1)
    for node, group in nodes_groups.items():
        if node in vanilla_placement and vanilla_placement[node] == '-1':
            f.write(node + ' -1\n')
        else:
            if not node.startswith("^"):
                f.write(node + ' ' + str(group) + '\n')
                smm[group] += analysis_graph[node].duration
                if analysis_graph[node].level >= 510 and analysis_graph[node].level <= 650:
                    light_levels_sum[group] += analysis_graph[node].duration
                    count[group] += 1
                cntt[group] += 1

    for i in range(0, no_of_desired_groups):
        print(str(i) + ': ' + str(cntt[i]) +
              ', ' + str(smm[i]) + ', ' + str(count[i]) + ', ' + str(light_levels_sum[i]))