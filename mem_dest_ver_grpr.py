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

# output file
out1 = io_folder_path + 'ver_grouper_placement_e_nc.place'

# grouper parameters
no_of_desired_groups = 2
memory_limit_per_group = 31 * 1024 * 1024 * 1024

comm_latency = 45
average_tensor_size_if_not_provided = 1
comm_transfer_rate = 1000000 / (140 * 1024 * 1024 * 1024)

reverse_levels = {}
# get nodes levels
with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            reverse_levels[node_and_level[0]] = node_and_level[1]

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
        edges_weights[tensor_name] = float(tensor_size) * comm_transfer_rate + comm_latency

# getting time (weight) info for nodes
analysis_graph = utils.read_profiling_file(in2, True)

# get_node_average_weiht
total_nodes_weight = 0
for node, node_props in analysis_graph.items():
    total_nodes_weight = total_nodes_weight + node_props.duration

average_node_weight = total_nodes_weight/len(analysis_graph)

# will contain the graph as an adgacency list
graph = {}
all_nodes = {}
sink_node_name = 'snk'
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

            all_nodes[splits[1]].parents.append(splits[0])
            all_nodes[splits[0]].children.append(splits[1])

            if splits[0] in graph.keys():
                graph[splits[0]].append(splits[1])
            else:
                graph[splits[0]] = [splits[1]]


for node, node_props in all_nodes.items():
    if node in analysis_graph:
        analysis_graph[node].parents = node_props.parents
        analysis_graph[node].children = node_props.children
    else:
        analysis_graph[node] = node_props


for node in all_nodes:
    if not node in tensors_sizes:
        tensors_sizes[node] = 0
        edges_weights[node] = float(comm_latency)

# constructing the graph and initializing the nodes levels from the dot file
rev_graph = {}
with open(in4_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            if nodes[0] in rev_graph:
                rev_graph[nodes[0]].append(nodes[1])
            else:
                rev_graph[nodes[0]] = [nodes[1]]

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

def get_nodes_weighted_levels(graph, edges_weights, src_nodes=None):
    # getting the sources of the graph to start the topological traversal from them
    graph_keys = {}
    nodes_weighted_levels={}
    tmp_nodes_in_degrees = copy.deepcopy(rev_nodes_in_degrees)

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
        nodes_weighted_levels[node] = 0  # analysis_graph[node].duration
        traversal_queueu.put(node)

    # start the traversal
    while not traversal_queueu.empty():
        current_node = traversal_queueu.get()
        if current_node in graph:
            adj_nodes = graph[current_node]
        else:
            adj_nodes = []

        current_node_level = nodes_weighted_levels[current_node]
        for adj_node in adj_nodes:
            if adj_node not in src_nodes:
                new_level = current_node_level + edges_weights[adj_node] + analysis_graph[adj_node].duration
                tmp_nodes_in_degrees[adj_node] -= 1
                if adj_node not in nodes_weighted_levels or nodes_weighted_levels[adj_node] < new_level:
                    nodes_weighted_levels[adj_node] = new_level
                if tmp_nodes_in_degrees[adj_node] == 0:
                    traversal_queueu.put(adj_node)
    
    return nodes_weighted_levels

nodes_weighted_levels = get_nodes_weighted_levels(rev_graph, edges_weights)
graph[sink_node_name] = []

print(nodes_weighted_levels['gradients/rnnlm_1/rnnlm/multi_rnn_cell/cell_5/basic_lstm_cell/mul_21_grad/mul_1'])
print(nodes_weighted_levels['rnnlm_1/rnnlm/multi_rnn_cell/cell_5/basic_lstm_cell/sigmoid_21'])
# extracting all vertical paths in the graph
source_node_name = 'src'
free_nodes = []
heapq.heappush(free_nodes, (0, source_node_name))
paths = []
current_path = []
visited = {}
groups_weights = []
paths_lengths = []
current_path_weight = 0
current_path_weight_with_comm = 0
num_paths = 0
nodes_paths_mapping = {}
nodes_to_visit = list(all_nodes.keys())

while free_nodes:
    current_node = heapq.heappop(free_nodes)[1]

    while current_node !='' and current_node not in visited:
        current_path.append(current_node)
        current_path_weight = current_path_weight + \
            analysis_graph[current_node].duration
        current_path_weight_with_comm = current_path_weight_with_comm + \
            analysis_graph[current_node].duration + edges_weights[current_node]
        visited[current_node] = 1
        max_priority = -1
        next_node = ''
        for adj_node in graph[current_node]:
            if adj_node not in visited and nodes_weighted_levels[adj_node] > max_priority:
                max_priority = nodes_weighted_levels[adj_node]
                next_node = adj_node
        current_node = next_node

    if len(current_path) > 0:
        paths.append(current_path)
        groups_weights.append(current_path_weight)
        paths_lengths.append(len(current_path))
        if current_path_weight_with_comm >= groups_weights[0] / 20 or len(paths) <= no_of_desired_groups:
            nodes_weighted_levels = get_nodes_weighted_levels(rev_graph, edges_weights, visited)

        for node in current_path:
            for adj_node in graph[node]:
                if adj_node not in visited:
                    heapq.heappush(
                        free_nodes, (-nodes_weighted_levels[adj_node], adj_node))

        current_path = []
        current_path_weight = 0
        current_path_weight_with_comm = 0
        num_paths = num_paths + 1

# sort paths from shortest to longest
paths_lengths, groups_weights, paths = (list(t) for t in zip(
    *sorted(zip(paths_lengths, groups_weights, paths))))
print('num of paths: ' + str(len(paths)))
print(paths_lengths[-20:])

# which node is in which path
nodes_paths_mapping[source_node_name] = num_paths - 1
nodes_paths_mapping[sink_node_name] = num_paths - 1
for i in range(0, num_paths):
    for node in paths[i]:
        nodes_paths_mapping[node] = i
    
    if 'gradients/rnnlm_1/rnnlm/multi_rnn_cell/cell_5/basic_lstm_cell/sigmoid_21_grad/sigmoidgrad' in paths[i]:
        print(paths[i])

# get max potential of paths
groups_parents = {}
paths_max_potential = copy.deepcopy(groups_weights)

for i in range(0, len(paths)):
    current_path = paths[i]
    current_path_len = len(current_path) - 1
    parent_path_indx = -1
    found = False
    heaviest_parent_child_tensor = 0
    heaviest_parent_or_child_path = -1
    if current_path[0] != source_node_name and current_path[current_path_len] != sink_node_name:
        for src_node in analysis_graph[current_path[0]].parents:
            if tensors_sizes[src_node] > heaviest_parent_child_tensor:
                heaviest_parent_child_tensor = tensors_sizes[src_node]
                heaviest_parent_or_child_path = nodes_paths_mapping[src_node]
            for dst_node in analysis_graph[current_path[current_path_len]].children:
                if tensors_sizes[dst_node] > heaviest_parent_child_tensor:
                    heaviest_parent_child_tensor = tensors_sizes[dst_node]
                    heaviest_parent_or_child_path = nodes_paths_mapping[dst_node]
                if nodes_paths_mapping[src_node] == nodes_paths_mapping[dst_node]:
                    parent_path_indx = nodes_paths_mapping[src_node]
                    paths_max_potential[parent_path_indx] = paths_max_potential[parent_path_indx] + \
                        paths_max_potential[i]
                    found = True
                    break
                if found:
                    break
    if parent_path_indx == -1:
        parent_path_indx = heaviest_parent_or_child_path
    groups_parents[i] = parent_path_indx

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

#map, helpful to find nodes in a level in O(1)
levels_nodes = [None] * no_of_levels
for node, props in analysis_graph.items():
    if node in all_nodes:
        if levels_nodes[props.level] == None:
            levels_nodes[props.level] = []
        levels_nodes[props.level].append(node)

# get the average path length
after_heavy_paths_count = 0
after_heavy_paths_lengths = 0
for path in paths:
    if analysis_graph[path[0]].level > 8:
        after_heavy_paths_count = after_heavy_paths_count + 1
        after_heavy_paths_lengths = after_heavy_paths_lengths + len(path)

average_path_len = math.ceil(
    after_heavy_paths_lengths / after_heavy_paths_count)

# getting initial groups
initial_groups = copy.deepcopy(paths)
initial_groups_indices = [1] * num_paths
path_joined_group = {}

for i in range(0, num_paths - 1):
    current_group = initial_groups[i]
    current_group_weight = groups_weights[i]
    group_comm_time = 0
    total_branching_potential = 0
    branch_start = ''
    branch_end = ''
    branching_main_path = groups_parents[i]
    sibling_from_branching_main_path = ''
    current_group_siblings_potentials = 0
    sibling_from_branching_main_path_weight = 0

    if (current_group_weight >= average_node_weight or len(current_group) >= average_path_len) and current_group_weight > 0:
        if current_group[0] != source_node_name and current_group[len(current_group) - 1] != sink_node_name:
            for src_node in analysis_graph[current_group[0]].parents:
                if nodes_paths_mapping[src_node] == branching_main_path:
                    branch_start = src_node
            for dst_node in analysis_graph[current_group[len(current_group) - 1]].children:
                if nodes_paths_mapping[dst_node] == branching_main_path:
                    branch_end = dst_node

            if branch_start != '' and branch_end != '':
                current_group_siblings_heads = analysis_graph[branch_start].children

                for node in current_group_siblings_heads:
                    if node == current_group[0]:
                        continue
                    if nodes_paths_mapping[node] != branching_main_path:
                        current_group_siblings_potentials = current_group_siblings_potentials + \
                            paths_max_potential[nodes_paths_mapping[node]]
                    else:
                        sibling_from_branching_main_path = node
                        traversal_queue = [sibling_from_branching_main_path]
                        visited_nodes = {}
                        while len(traversal_queue) > 0:
                            current_node = traversal_queue.pop(0)
                            if current_node != branch_end and current_node not in visited_nodes and analysis_graph[current_node].level < analysis_graph[branch_end].level:
                                sibling_from_branching_main_path_weight = sibling_from_branching_main_path_weight + \
                                    analysis_graph[current_node].duration
                                traversal_queue = traversal_queue + \
                                    graph[current_node]
                            visited_nodes[current_node] = 1

                total_branching_potential = (
                    current_group_siblings_potentials) + sibling_from_branching_main_path_weight
                in_tensor_size = 0
                out_tensor_size = 0
                if branch_start in tensors_sizes:
                    in_tensor_size = tensors_sizes[branch_start]
                else:
                    in_tensor_size = average_tensor_size_if_not_provided
                if current_group[len(current_group) - 1] in tensors_sizes:
                    out_tensor_size = tensors_sizes[current_group[len(current_group) - 1]]
                else:
                    out_tensor_size = average_tensor_size_if_not_provided

                group_comm_time = comm_latency * 2 + \
                    (in_tensor_size + out_tensor_size) * comm_transfer_rate

                if group_comm_time >= total_branching_potential + current_group_weight:
                    while initial_groups_indices[branching_main_path] == 0:
                        # union find like stuff
                        branching_main_path = path_joined_group[branching_main_path]
                    path_joined_group[i] = branching_main_path
                    initial_groups_indices[i] = 0
                    groups_weights[branching_main_path] = groups_weights[branching_main_path] + \
                        groups_weights[i]
                    if len(initial_groups[branching_main_path]) > 1:
                        main_path_tail = initial_groups[branching_main_path].pop(
                            len(initial_groups[branching_main_path]) - 1)
                        initial_groups[branching_main_path] = initial_groups[branching_main_path] + \
                            initial_groups[i]
                        initial_groups[branching_main_path].append(
                            main_path_tail)
                    else:
                        initial_groups[branching_main_path] = initial_groups[branching_main_path] + \
                            initial_groups[i]
    else:
        if branching_main_path == -1:
            branching_main_path = nodes_paths_mapping[analysis_graph[current_group[0]].parents[0]]
        while initial_groups_indices[branching_main_path] == 0:
            branching_main_path = path_joined_group[branching_main_path]
        path_joined_group[i] = branching_main_path
        initial_groups_indices[i] = 0
        groups_weights[branching_main_path] = groups_weights[branching_main_path] + \
            groups_weights[i]
        if len(initial_groups[branching_main_path]) > 1:
            main_path_tail = initial_groups[branching_main_path].pop(
                len(initial_groups[branching_main_path]) - 1)
            initial_groups[branching_main_path] = initial_groups[branching_main_path] + \
                initial_groups[i]
            initial_groups[branching_main_path].append(main_path_tail)
        else:
            initial_groups[branching_main_path] = initial_groups[branching_main_path] + \
                initial_groups[i]

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
min_levels = [20000]*len(initial_groups)

for i in range(0, len(initial_groups)):
    tasks_per_levels.append(collections.OrderedDict())
    current_group = initial_groups[i]
    for node in current_group:
        node_props = analysis_graph[node]
        node_level = int(node_props.level)
        if node_level in tasks_per_levels[i].keys():
            tasks_per_levels[i][node_level] = tasks_per_levels[i][node_level] + \
                node_props.duration
        else:
            tasks_per_levels[i][node_level] = node_props.duration

        if node_level < min_levels[i]:
            min_levels[i] = node_level
        if node_level > max_levels[i]:
            max_levels[i] = node_level

# getting main groups
main_groups_indices = [-1] * no_of_desired_groups
for i in range(0, no_of_desired_groups):
    main_groups_indices[i] = num_initial_groups - i - 1
total_gain = 0

nodes_groups = {}
nodes_initial_groups = {}
for node in analysis_graph.keys():
    nodes_groups[node] = -1

for i in range(0, len(initial_groups)):
    for node in initial_groups[i]:
        nodes_initial_groups[node] = i

for i in range(0, len(main_groups_indices)):
    for node in initial_groups[main_groups_indices[i]]:
        nodes_groups[node] = i

cntt = 0
final_groups = copy.deepcopy(initial_groups)
final_groups_weights = copy.deepcopy(groups_weights)

# merging the groups
while len(final_groups) > no_of_desired_groups:
    to_be_merged_group_index = len(final_groups) - no_of_desired_groups - 1
    to_be_merged_group = final_groups[to_be_merged_group_index]
    branch_main_path_indx = -1

    src_min_level = -1
    branch_src_node = ''
    branch_snk_node = ''
    src_max_level = 20000
    for src_node in analysis_graph[to_be_merged_group[0]].parents:
        if int(analysis_graph[src_node].level) > src_min_level:
            src_min_level = int(analysis_graph[src_node].level)
            branch_src_node = src_node
    for dst_node in analysis_graph[to_be_merged_group[len(to_be_merged_group) - 1]].children:
        if int(analysis_graph[dst_node].level) < src_max_level:
            src_max_level = int(analysis_graph[dst_node].level)
            branch_snk_node = dst_node

    for i in range(0, len(final_groups)):
        if branch_src_node in final_groups[i] and branch_snk_node in final_groups[i]:
            branch_main_path_indx = i

    # Note: the tensor size sent to the neighbors is the same, so pick the one with the lowest level.
    max_parent_or_child_tensor = ''
    max_parent_or_child_tensor_size = 0
    if branch_main_path_indx == -1:
        for parent_node in analysis_graph[to_be_merged_group[0]].parents:
            if parent_node in tensors_sizes and tensors_sizes[parent_node] > max_parent_or_child_tensor_size and nodes_groups[parent_node] != -1:
                max_parent_or_child_tensor_size = tensors_sizes[parent_node]
                max_parent_or_child_tensor = parent_node
        if max_parent_or_child_tensor != '':
            branch_main_path_indx = nodes_groups[max_parent_or_child_tensor]

    in_tensor_size = 0
    out_tensor_size = 0
    if branch_src_node in tensors_sizes:
        in_tensor_size = tensors_sizes[branch_src_node]
    else:
        in_tensor_size = average_tensor_size_if_not_provided

    if to_be_merged_group[len(to_be_merged_group) - 1] in tensors_sizes:
        out_tensor_size = tensors_sizes[to_be_merged_group[len(to_be_merged_group) - 1]]
    else:
        out_tensor_size = average_tensor_size_if_not_provided

    group_comm_time = comm_latency * 2 + \
        (in_tensor_size + out_tensor_size) * comm_transfer_rate

    min_sum_in_targeted_levels = 1000000000000
    main_branch_sum_in_targeted_levels = 0
    merge_destination_index = 0

    for i in main_groups_indices:
        sum_in_targeted_levels = 0
        # current_min_level = max(src_min_level, min_levels[i])
        # current_max_level = min(src_max_level, max_levels[i])

        for level in range(src_min_level + 1, src_max_level - 1):
            # and level in tasks_per_levels[to_be_merged_group_index].keys():
            if level in tasks_per_levels[i].keys():
                sum_in_targeted_levels = sum_in_targeted_levels + \
                    tasks_per_levels[i][level]

        if sum_in_targeted_levels < min_sum_in_targeted_levels:
            min_sum_in_targeted_levels = sum_in_targeted_levels
            merge_destination_index = i
        if i == branch_main_path_indx:
            main_branch_sum_in_targeted_levels = sum_in_targeted_levels

    if merge_destination_index != branch_main_path_indx and branch_main_path_indx in main_groups_indices:
        current_branch_max_time = min_sum_in_targeted_levels + \
            group_comm_time + final_groups_weights[to_be_merged_group_index]
        an_alternative_max_time = main_branch_sum_in_targeted_levels + \
            final_groups_weights[to_be_merged_group_index]

        tmp_group = copy.deepcopy(to_be_merged_group)
        poped_group = []
        tmp_group_weight = final_groups_weights[to_be_merged_group_index]
        poped_group_weight = 0

    merge_min_level = min(
        min_levels[to_be_merged_group_index], min_levels[merge_destination_index])
    merge_max_level = max(
        max_levels[to_be_merged_group_index], max_levels[merge_destination_index])
    final_groups_weights[merge_destination_index] = final_groups_weights[merge_destination_index] + \
        final_groups_weights[to_be_merged_group_index]
    final_groups_weights.pop(to_be_merged_group_index)
    final_groups[merge_destination_index] = final_groups[merge_destination_index] + \
        to_be_merged_group
    final_groups.pop(to_be_merged_group_index)
    min_levels[merge_destination_index] = merge_min_level
    max_levels[merge_destination_index] = merge_max_level
    min_levels.pop(to_be_merged_group_index)
    max_levels.pop(to_be_merged_group_index)
    merge_src_levels_tasks = tasks_per_levels[to_be_merged_group_index]

    for node in to_be_merged_group:
        nodes_groups[node] = main_groups_indices.index(merge_destination_index)

    for j in range(0, no_of_desired_groups):
        main_groups_indices[j] = main_groups_indices[j] - 1

    for level, tasks_sum in merge_src_levels_tasks.items():
        if level in tasks_per_levels[merge_destination_index].keys():
            tasks_per_levels[merge_destination_index][level] = tasks_per_levels[merge_destination_index][level] + tasks_sum
        else:
            tasks_per_levels[merge_destination_index][level] = tasks_sum
    tasks_per_levels.pop(to_be_merged_group_index)

nodes_groups[sink_node_name] = 0

#post processing, switching nodes placement modification:
switching_nodes_pure_parents = []
switching_nodes_pure_children = []
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

for switching_node in switching_nodes_pure_children:
    children_final_group = nodes_groups[graph[switching_node][0]]
    switching_node_group = nodes_groups[switching_node]
    comm_from_its_group = 0
    comm_from_children_group = 0
    some_parent_initial_group = -1
    for parent in rev_graph[switching_node]:
        if nodes_groups[parent] == children_final_group:
            comm_from_children_group += edges_weights[parent]
            if some_parent_initial_group == -1:
                some_parent_initial_group = nodes_initial_groups[parent]
        elif nodes_groups[parent] == switching_node_group:
            comm_from_its_group += edges_weights[parent]
    
    comm_from_children_group += edges_weights[switching_node]
    movement_gain = comm_from_children_group - (comm_from_its_group + analysis_graph[switching_node].duration)
    if movement_gain > 0:
        nodes_groups[switching_node] = children_final_group
        initial_groups[some_parent_initial_group].append(switching_node)
        initial_groups[nodes_initial_groups[switching_node]].remove(switching_node)
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
        nodes_groups[switching_node] = parents_final_group
        initial_groups[some_child_initial_group].insert(0, switching_node)
        initial_groups[nodes_initial_groups[switching_node]].remove(switching_node)
        nodes_initial_groups[switching_node] = some_child_initial_group


nodes_memory = {}
additional_memory = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        nodes_memory[splitted[0]] = int(splitted[1])
        additional_memory[splitted[0]] = int(splitted[2])

for node in all_nodes.keys():
    if not node in additional_memory:
        additional_memory[node] = 0
        nodes_memory[node] = 0

levels_additional_memory = []
levels_additional_memory_by_node = []
for i in range(0, no_of_desired_groups):
    levels_additional_memory.append([])
    levels_additional_memory_by_node.append([])

for i in range(0, no_of_desired_groups):
    for j in range(0, no_of_levels):
        levels_additional_memory_by_node[i].append([])

current_level = 0
for current_level_nodes in levels_nodes:
    max_addition = [0] * no_of_desired_groups
    for node in current_level_nodes:
        current_node_group = nodes_groups[node]
        max_addition[current_node_group] = max(
            additional_memory[node], max_addition[current_node_group])
        levels_additional_memory_by_node[current_node_group][current_level].append(additional_memory[node])
    current_level += 1

    for i in range(0, no_of_desired_groups):
        levels_additional_memory[i].append(max_addition[i])

def get_levels_memory_consumption(graph, src_nodes=None):
    # getting the sources of the graph to start the topological traversal from them
    graph_keys = {}
    nodes_ranks = {}
    levels_memory_consumption = []

    for i in range(0, no_of_desired_groups):
        levels_memory_consumption.append({-1: 0})

    tmp_nodes_in_degrees = copy.deepcopy(nodes_in_degrees)

    for graph_key in graph.keys():
        graph_keys[graph_key] = 0
        nodes_ranks[graph_key] = len(graph[graph_key])

    for adj_nodes in graph.values():
        for node in adj_nodes:
            if node in graph_keys:
                graph_keys[node] = 1
            else:
                nodes_ranks[node] = 0

    traversal_queueu = queue.Queue()

    if src_nodes is None:
        src_nodes = []
        for node, source_node in graph_keys.items():
            if source_node == 0:
                src_nodes.append(node)

    for node in src_nodes:
        traversal_queueu.put(node)

    # start the traversal
    previously_visited = []
    previously_visited_levels = []
    while not traversal_queueu.empty():
        current_node = traversal_queueu.get()
        current_level = analysis_graph[current_node].level
        current_group = nodes_groups[current_node]

        if current_node in graph:
            adj_nodes = graph[current_node]
        else:
            adj_nodes = []

        #remove footprint of nodes which has no children in their groups group
        if previously_visited:
            removed_indices = {}
            for i in range(0, len(previously_visited)):
                if previously_visited_levels[i] < current_level:
                    previously_visited_node = previously_visited[i]
                    levels_memory_consumption[nodes_groups[previously_visited_node]
                                            ][analysis_graph[previously_visited_node].level] -= nodes_memory[previously_visited_node]
                    removed_indices[i] = i
            tmp_previously_visited_levels = copy.deepcopy(previously_visited_levels)
            tmp_previously_visited = copy.deepcopy(previously_visited)
            previously_visited_levels = []
            previously_visited = []
            for i in range(0, len(tmp_previously_visited)):
                if i not in removed_indices:
                    previously_visited.append(tmp_previously_visited[i])
                    previously_visited_levels.append(tmp_previously_visited_levels[i])

        if current_level not in levels_memory_consumption[current_group]:
            #fill levels which has no nodes
            for level in range(len(levels_memory_consumption[current_group]) - 1, current_level):
                levels_memory_consumption[current_group][level] = levels_memory_consumption[current_group][level - 1] 

            levels_memory_consumption[current_group][current_level] = nodes_memory[current_node] + \
                levels_memory_consumption[current_group][current_level - 1] + \
                levels_additional_memory[current_group][current_level] - \
                levels_additional_memory[current_group][current_level - 1]
        else:
            levels_memory_consumption[current_group][current_level] += nodes_memory[current_node]

        for parent_node in analysis_graph[current_node].parents:
            nodes_ranks[parent_node] -= 1
            if nodes_ranks[parent_node] == 0:
                parent_node_group = nodes_groups[parent_node]
                if current_level not in levels_memory_consumption[parent_node_group]:
                    for level in range(len(levels_memory_consumption[parent_node_group]) - 1, current_level + 1):
                        levels_memory_consumption[parent_node_group][level] = levels_memory_consumption[parent_node_group][level - 1] + \
                        levels_additional_memory[parent_node_group][level] - \
                        levels_additional_memory[parent_node_group][level - 1]

                levels_memory_consumption[parent_node_group][current_level] -= nodes_memory[parent_node]

        groups_visited = {}
        number_of_adjacents_in_the_same_group = 0

        for i in range(0, no_of_desired_groups):
            groups_visited[i] = False
        for adj_node in adj_nodes:
            adj_node_group = nodes_groups[adj_node]
            if adj_node_group != current_group:
                if current_level + 1 not in levels_memory_consumption[adj_node_group]:
                    for level in range(len(levels_memory_consumption[adj_node_group]) - 1, current_level + 2):
                        levels_memory_consumption[adj_node_group][level] = levels_memory_consumption[adj_node_group][level - 1] + \
                        levels_additional_memory[adj_node_group][level] - \
                        levels_additional_memory[adj_node_group][level - 1]
                if not groups_visited[adj_node_group]:
                    levels_memory_consumption[adj_node_group][current_level + 1] += nodes_memory[current_node]
                    groups_visited[adj_node_group] = True
            else:
                number_of_adjacents_in_the_same_group += 1

            tmp_nodes_in_degrees[adj_node] -= 1
            if tmp_nodes_in_degrees[adj_node] == 0:
                traversal_queueu.put(adj_node)
        # A node that has no adjacents in its group: it is stored so that its memory consumption is subtracted when moving to the next level(TF way of handling, adding sudo send nodes)
        if number_of_adjacents_in_the_same_group == 0:
            previously_visited.append(current_node)
            previously_visited_levels.append(current_level)

    return levels_memory_consumption


final_groups_memory_consumptions = get_levels_memory_consumption(graph)

memory_limit_is_exceeded = False
for i in range(0, no_of_desired_groups):
    for level in range(no_of_levels - 1, 0, -1):
        if final_groups_memory_consumptions[i][level] > memory_limit_per_group:
            memory_limit_is_exceeded = True
            break

if memory_limit_is_exceeded:
    #add dependencies:
    nodes_last_child_level_with_deps = {}
    nodes_last_child_level = {}
    for node in all_nodes.keys():
        max_child_level = 0
        nodes_last_child_level[node] = 0
        for child in graph[node]:
            child_level = analysis_graph[child].level
            if  child_level > max_child_level:
                max_child_level = child_level
            if child_level > nodes_last_child_level[node]:
                nodes_last_child_level[node] = child_level
            visited = {}
            splitted_child = child.split('/')
            if child.startswith('^') or 'group_deps' in splitted_child[-1] or 'control_dependency' in splitted_child[-1]:
                nodes_to_visit = [child]
                while nodes_to_visit:
                    current_visit = nodes_to_visit.pop()
                    for child_of_child in graph[current_visit]:
                        if child_of_child not in visited:
                            visited[child_of_child] = 1
                            splitted_child_of_child = child_of_child.split('/')
                            if analysis_graph[child_of_child].level > max_child_level:
                                max_child_level = analysis_graph[child_of_child].level
                            if child_of_child.startswith('^') or 'group_deps' in splitted_child_of_child[-1] or 'control_dependency' in splitted_child_of_child[-1]:
                                nodes_to_visit.append(child_of_child)
        
        nodes_last_child_level_with_deps[node] = max_child_level

    for node in all_nodes.keys():
        for level in range(nodes_last_child_level, nodes_last_child_level_with_deps):
            final_groups_memory_consumptions[nodes_groups[node]][level] += nodes_memory[node]

    # work destribution among levels:
    final_groups_levels_work = []
    for group in range(0, no_of_desired_groups):
        final_groups_levels_work.append([])
        for level in range(0, no_of_levels):
                final_groups_levels_work[group].append(0)

    for level in range(0, no_of_levels):
        for node in levels_nodes[level]:
            final_groups_levels_work[nodes_groups[node]][level] += analysis_graph[node].duration


    # memory consumption of the initial groups
    initial_groups_mem_cons = []
    current_initial_group = 0
    for group in initial_groups:
        current_group_mem_cons = 0
        for node in group:
            current_group_mem_cons += nodes_memory[node]
        initial_groups_mem_cons.append(current_group_mem_cons)

    #communication and computation of the initial groups
    current_group_indx = 0
    initial_groups_comps_comms = []
    initial_groups_comms = []
    initial_groups_comm_comp_to_mem_ratio = []
    initial_groups_start_levels = []
    initial_groups_earliest_sink_levels = []
    com_comp_min = math.inf
    com_comp_max = 0
    ratio_min = math.inf
    ratio_max = 0
    for initial_group in initial_groups:
        starting_node = initial_group[0]
        end_node = initial_group[len(initial_group) - 1]
        initial_groups_start_levels.append(analysis_graph[starting_node].level)
        comm_cost = 0
        for parent in rev_graph[starting_node]:
            comm_cost += tensors_sizes[parent]

        comm_cost += tensors_sizes[end_node]
        initial_groups_comms.append(comm_cost)
        total_cost = comm_cost + groups_weights[current_group_indx]
        initial_groups_comps_comms.append(total_cost)
        ratio = total_cost / (initial_groups_mem_cons[current_group_indx] + 1) #negligable addition, just to avoid devision by zero without using if statements
        initial_groups_comm_comp_to_mem_ratio[current_group_indx] = ratio

        min_child_end_level = math.inf
        for child in graph[end_node]:
            current_child_level = analysis_graph[child].level
            if current_child_level < min_child_end_level:
                min_child_end_level = current_child_level

        initial_groups_earliest_sink_levels.append(min_child_end_level)

        if total_cost > com_comp_max:
            com_comp_max = total_cost
        elif total_cost < com_comp_min:
            com_comp_min = total_cost

        if ratio > ratio_max:
            ratio_max = ratio
        elif ratio < ratio_min:
            ratio_min = ratio

        current_group_indx += 1

    normalized_ratio_den = ratio_max - ratio_min
    normalized_comm_comp_den = com_comp_max - com_comp_min
    initial_groups_sorting_criteria = []
    for i in range(0, len(initial_group)):
        initial_groups_sorting_criteria[i] = initial_groups_comps_comms[i] / normalized_comm_comp_den + initial_groups_comm_comp_to_mem_ratio[i] / normalized_ratio_den

    initial_groups_mem_cons, initial_groups_sorting_criteria, initial_groups, initial_groups_start_levels, initial_groups_earliest_sink_levels, initial_groups_comms = (list(t) for t in zip(
        *sorted(zip(initial_groups_mem_cons, initial_groups_sorting_criteria, initial_groups, initial_groups_start_levels, initial_groups_earliest_sink_levels, initial_groups_comms))))

    initial_final_group_mapping = {}
    current_initial_group_indx = 0
    for group in initial_groups:
        current_final_group = nodes_groups[group[0]]
        initial_final_group_mapping[current_initial_group_indx] = current_final_group
        current_initial_group_indx += 1

    current_level = 0
    for group_no in range(0, no_of_desired_groups):
        while current_level < no_of_levels:
            if final_groups_memory_consumptions[level] > memory_limit_per_group:
                overflow = final_groups_memory_consumptions[level] - \
                    memory_limit_per_group
                candidate_groups_indices = []
                candidate_groups_mem_cons = []
                candidate_groups_sotring_weights = []
                candidate_group_mem_cons = 0
                sum_of_candidates = 0
                current_indx = 0
                while candidate_group_mem_cons <= overflow and sum_of_candidates <= no_of_desired_groups * overflow:
                    if initial_final_group_mapping[current_indx] == group_no and initial_groups_start_levels[current_indx] <= current_level:
                        candidate_group_mem_cons = initial_groups_mem_cons[current_indx]
                        candidate_groups_indices.append(current_indx)
                        candidate_groups_mem_cons.append(candidate_group_mem_cons)
                        candidate_groups_sotring_weights.append(initial_groups_sorting_criteria[current_indx])

                    current_indx += 1

                candidate_groups_sotring_weights, candidate_groups_mem_cons, candidate_groups_indices = (list(t) for t in zip(
                    *sorted(zip(candidate_groups_sotring_weights, candidate_groups_mem_cons, candidate_groups_indices))))         

                for sub_group_indx in candidate_groups_indices:
                    current_sub_group = candidate_groups_indices[sub_group_indx]
                    merge_with_index = -1
                    sub_group_start_level = analysis_graph[sub_group_indx[0]].level
                    comm_comp_with_final_groups = [initial_groups_comms[sub_group_indx]] * no_of_desired_groups
                    #how much communication with the other final groups will this movement incure
                    for parent_node in rev_graph[current_sub_group[0]]:
                        comm_comp_with_final_groups[nodes_groups[parent_node]] -= tensors_sizes[parent_node]
                    comm_comp_with_final_groups[nodes_groups[current_sub_group[-1]]] -= tensors_sizes[current_sub_group[-1]]

                    final_groups_indices = []
                    for i in range(0, no_of_desired_groups):
                        final_groups_indices.append(i)

                    for i in range(0, len(final_groups)):
                        for level in range(sub_group_start_level, initial_groups_earliest_sink_levels[sub_group_indx] - 1):
                            comm_comp_with_final_groups[i] += final_groups_levels_work[i][level]

                    comm_comp_with_final_groups, final_groups_indices = (list(t) for t in zip( *sorted(zip(comm_comp_with_final_groups, final_groups_indices))))

                    tmp_groups_mem_consumption = copy.deepcopy(final_groups_memory_consumptions)
                    tmp_levels_additional_memory = copy.deepcopy(levels_additional_memory)
                    tmp_levels_additional_memory_by_node = copy.deepcopy(levels_additional_memory_by_node)
                    merged = False
                    for final_group_indx in final_groups_indices:
                        merged = True
                        for node in current_sub_group:
                            node_level = analysis_graph[node].level
                            node_additional_memory = additional_memory[node]
                            node_mem_cons = nodes_memory[node]
                            tmp_levels_additional_memory_by_node[group_no][node_level].remove(node_additional_memory)
                            if tmp_levels_additional_memory[group_no][node_level] <= node_additional_memory:
                                tmp_levels_additional_memory[group_no][node_level] = max(tmp_levels_additional_memory_by_node[group_no][node_level])
                                tmp_groups_mem_consumption[group_no][node_level] += tmp_levels_additional_memory[group_no][node_level]
                                tmp_groups_mem_consumption[group_no][node_level] -= node_additional_memory
                            tmp_levels_additional_memory_by_node[final_group_indx][node_level].append(node_additional_memory)
                            if tmp_levels_additional_memory[final_group_indx][node_level] < node_additional_memory:
                                tmp_levels_additional_memory[final_group_indx][node_level] = node_additional_memory
                                tmp_groups_mem_consumption[final_group_indx][node_level] += node_additional_memory
                                tmp_groups_mem_consumption[final_group_indx][node_level] -= levels_additional_memory[final_group_indx][node_level]
                                if tmp_groups_mem_consumption[final_group_indx][node_level] > memory_limit_per_group:
                                    merged = False
                                    break

                            for level in range(node_level, nodes_last_child_level_with_deps[node]):
                                tmp_groups_mem_consumption[group_no][level] -= node_mem_cons
                                tmp_groups_mem_consumption[final_group_indx][level] += node_mem_cons
                                if tmp_groups_mem_consumption > memory_limit_per_group:
                                    merged = False
                                    break
                        
                        if merged:
                            final_groups_memory_consumptions = tmp_groups_mem_consumption
                            levels_additional_memory = tmp_levels_additional_memory
                            levels_additional_memory_by_node = tmp_levels_additional_memory_by_node

                            initial_final_group_mapping[sub_group_indx] = final_group_indx
                            for node in  current_sub_group:
                                nodes_groups[node] = final_group_indx
                            
                            for node in current_sub_group:
                                node_comp = analysis_graph[node].duration
                                node_level = analysis_graph[node].level
                                final_groups_levels_work[group_no][node_level] -= node_comp 
                                final_groups_levels_work[final_group_indx][node_level] += node_comp 

                            break

            current_level += 1


with open(out1, 'w') as f:
    smm = [0] * no_of_desired_groups
    light_levels_sum = [0] * no_of_desired_groups
    cntt = [0] * no_of_desired_groups
    count = [0] * no_of_desired_groups
    for node, group in nodes_groups.items():
        if not node.startswith("^"):
            f.write(node + ' ' + str(group) + '\n')
            smm[group] += analysis_graph[node].duration
            if analysis_graph[node].level >= 0 and analysis_graph[node].level < 10:
                light_levels_sum[group] += analysis_graph[node].duration
                count[group] += 1
            cntt[group] += 1

    for i in range(0, no_of_desired_groups):
        print(str(i) + ': ' + str(cntt[i]) +
              ', ' + str(smm[i]) + ', ' + str(count[i]) + ', ' + str(light_levels_sum[i]))
