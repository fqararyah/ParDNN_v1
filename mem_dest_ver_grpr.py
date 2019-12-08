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

""" # folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

in1 = io_folder_path + 'resnet_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step0_10_low.json'
# 'part_8_1799_src_sink_nodes_levels.txt'
in3 = io_folder_path + 'resnet_src_sink_nodes_levels_low.txt'
# 'rev_part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_resnet_src_sink_nodes_levels_low.txt'
in5 = io_folder_path + 'tensors_sz_32_low.txt' """

""" # folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/nmt/'

in1 = io_folder_path + 'nmt_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step1554.json'
# 'part_8_1799_src_sink_nodes_levels.txt'
in3 = io_folder_path + 'nmt_src_sink_nodes_levels_low.txt'
# 'rev_part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_nmt_src_sink_nodes_levels_low.txt'
in5 = io_folder_path + 'tensors_sz_32_low.txt' """

""" # folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/vgg/'

in1 = io_folder_path + 'vgg_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step110.json'
# 'part_8_1799_src_sink_nodes_levels.txt'
in3 = io_folder_path + 'vgg_src_sink_nodes_levels_low.txt'
# 'rev_part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_vgg_src_sink_nodes_levels_low.txt'
in5 = io_folder_path + 'tensors_sz_32_low.txt' """


""" in1 = io_folder_path + 'part_1_39_src_sink.dot'#'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'timeline_step303.json'
# 'part_8_1799_src_sink_nodes_levels.txt'
in3 = io_folder_path + 'part_1_39_src_sink_nodes_levels.txt'
# 'rev_part_8_1799_src_sink_nodes_levels.txt'
in4 = io_folder_path + 'rev_part_1_39_src_sink_nodes_levels.txt'
in5 = io_folder_path + 'tensors_sz.txt' """

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
total_tensor_sz = 0
# get tensors sizes
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensors_sizes[splitted[0]] = splitted[1]
        total_tensor_sz = total_tensor_sz + \
            int(splitted[1]) * comm_transfer_rate + comm_latency

avg_tensor_size = total_tensor_sz / len(tensors_sizes)

# getting time (weight) info for nodes
analysis_graph = utils.read_profiling_file(in2, True)

# get_node_average_weiht
total_nodes_weight = 0
for node, node_props in analysis_graph.items():
    total_nodes_weight = total_nodes_weight + node_props.duration

average_node_weight = total_nodes_weight/len(analysis_graph)

ccr = total_tensor_sz / total_nodes_weight

print('CCR= ' + str(ccr))

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
        tensors_sizes[node] = average_tensor_size_if_not_provided

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
for adjs in rev_graph.values():
    for adj in adjs:
        if adj in nodes_in_degrees:
            nodes_in_degrees[adj] += 1
        else:
            nodes_in_degrees[adj] = 1


def get_nodes_weighted_levels(graph, edges_weights, src_nodes=None, nodes_weighted_levels={}):
    # getting the sources of the graph to start the topological traversal from them
    graph_keys = {}
    tmp_nodes_in_degrees = copy.deepcopy(nodes_in_degrees)
    for graph_key in graph.keys():
        graph_keys[graph_key] = 0

    for adj_nodes in graph.values():
        for node in adj_nodes:
            if node in graph_keys:
                graph_keys[node] = 1

    traversal_queueu = queue.Queue()

    if src_nodes is None:
        src_nodes = []
        for node, source_node in graph_keys.items():
            if source_node == 0:
                src_nodes.append(node)

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
            new_level = current_node_level + \
                (int(edges_weights[adj_node]) * comm_transfer_rate + comm_latency) + \
                analysis_graph[adj_node].duration
            tmp_nodes_in_degrees[adj_node] -= 1
            if adj_node not in nodes_weighted_levels or nodes_weighted_levels[adj_node] < new_level:
                nodes_weighted_levels[adj_node] = new_level
            if tmp_nodes_in_degrees[adj_node] == 0:
                traversal_queueu.put(adj_node)

    return nodes_weighted_levels


def prioratize_adj_lists(graph, priorities):
    tmp_graph = {}
    for node, adjs in graph.items():
        tmp_graph[node] = []
        for adj in adjs:
            heapq.heappush(tmp_graph[node], (-priorities[adj], adj))

    graph = tmp_graph
    graph[sink_node_name] = []

    # change adjacents priority queues to lists
    for node, adjacents in graph.items():
        ordered_adjacents_list = []
        while adjacents:
            ordered_adjacents_list.append(heapq.heappop(adjacents)[1])
        graph[node] = ordered_adjacents_list

    return graph


nodes_weighted_levels = get_nodes_weighted_levels(rev_graph, tensors_sizes)
graph = prioratize_adj_lists(graph, nodes_weighted_levels)

graph[sink_node_name] = []


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
    adj_nodes = graph[current_node]

    while current_node not in visited and current_node != sink_node_name:
        current_path.append(current_node)
        current_path_weight = current_path_weight + \
            analysis_graph[current_node].duration
        current_path_weight_with_comm = current_path_weight_with_comm + \
            analysis_graph[current_node].duration + \
            int(tensors_sizes[current_node]) * \
            comm_transfer_rate + comm_latency
        visited[current_node] = 1
        nodes_to_visit.remove(current_node)
        for adj_node in graph[current_node]:
            if adj_node not in visited:
                current_node = adj_node
                break
    if len(current_path) > 0:
        paths.append(copy.deepcopy(current_path))
        groups_weights.append(current_path_weight)
        paths_lengths.append(len(current_path))

        if current_path_weight_with_comm >= groups_weights[0] or len(paths) <= no_of_desired_groups:
            nodes_weighted_levels = get_nodes_weighted_levels(
                rev_graph, tensors_sizes, list(visited.keys()), nodes_weighted_levels)
            prioratize_adj_lists(graph, nodes_weighted_levels)

        for node in current_path:
            for adj_node in graph[node]:
                if adj_node not in visited:
                    heapq.heappush(
                        free_nodes, (-nodes_weighted_levels[adj_node], adj_node))

        current_path = []
        current_path_weight = 0
        current_path_weight_with_comm = 0
        num_paths = num_paths + 1

    if len(free_nodes) == 0 and len(nodes_to_visit) > 0:
        heapq.heappush(
            free_nodes, (nodes_weighted_levels[nodes_to_visit[0]], nodes_to_visit[0]))
        nodes_to_visit.pop(0)

# sort paths from shortest to longest
paths_lengths, groups_weights, paths = (list(t) for t in zip(
    *sorted(zip(paths_lengths, groups_weights, paths))))
print('num of paths: ' + str(len(paths)))
# which node is in which path
nodes_paths_mapping[source_node_name] = num_paths - 1
nodes_paths_mapping[sink_node_name] = num_paths - 1
for i in range(0, num_paths):
    for node in paths[i]:
        nodes_paths_mapping[node] = i


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
            if int(tensors_sizes[src_node]) > heaviest_parent_child_tensor:
                heaviest_parent_child_tensor = int(tensors_sizes[src_node])
                heaviest_parent_or_child_path = nodes_paths_mapping[src_node]
            for dst_node in analysis_graph[current_path[current_path_len]].children:
                if int(tensors_sizes[dst_node]) > heaviest_parent_child_tensor:
                    heaviest_parent_child_tensor = int(tensors_sizes[dst_node])
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

commulative_levels_weights = {-1: 0}
for level in range(0, no_of_levels):
    level_weight = levels_weights[level]
    commulative_levels_weights[level] = commulative_levels_weights[level - 1] + level_weight

levels_nodes = [None] * no_of_levels
for node, props in analysis_graph.items():
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
                    in_tensor_size = int(tensors_sizes[branch_start])
                else:
                    in_tensor_size = average_tensor_size_if_not_provided
                if current_group[len(current_group) - 1] in tensors_sizes:
                    out_tensor_size = int(
                        tensors_sizes[current_group[len(current_group) - 1]])
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
            if parent_node in tensors_sizes and int(tensors_sizes[parent_node]) > max_parent_or_child_tensor_size and nodes_groups[parent_node] != -1:
                max_parent_or_child_tensor_size = int(
                    tensors_sizes[parent_node])
                max_parent_or_child_tensor = parent_node
        if max_parent_or_child_tensor != '':
            branch_main_path_indx = nodes_groups[max_parent_or_child_tensor]

    in_tensor_size = 0
    out_tensor_size = 0
    if branch_src_node in tensors_sizes:
        in_tensor_size = int(tensors_sizes[branch_src_node])
    else:
        in_tensor_size = average_tensor_size_if_not_provided

    if to_be_merged_group[len(to_be_merged_group) - 1] in tensors_sizes:
        out_tensor_size = int(
            tensors_sizes[to_be_merged_group[len(to_be_merged_group) - 1]])
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
            group_comm_time + groups_weights[to_be_merged_group_index]
        an_alternative_max_time = main_branch_sum_in_targeted_levels + \
            groups_weights[to_be_merged_group_index]

        tmp_group = copy.deepcopy(to_be_merged_group)
        poped_group = []
        tmp_group_weight = groups_weights[to_be_merged_group_index]
        poped_group_weight = 0

    merge_min_level = min(
        min_levels[to_be_merged_group_index], min_levels[merge_destination_index])
    merge_max_level = max(
        max_levels[to_be_merged_group_index], max_levels[merge_destination_index])
    groups_weights[merge_destination_index] = groups_weights[merge_destination_index] + \
        groups_weights[to_be_merged_group_index]
    groups_weights.pop(to_be_merged_group_index)
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


nodes_memory = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        nodes_memory[splitted[0]] = splitted[1]


def get_levels_memory_consumption(graph, src_nodes=None):
    # getting the sources of the graph to start the topological traversal from them
    graph_keys = {}
    nodes_ranks = {}
    levels_memory_consumption = []

    for i in range(0, no_of_desired_groups):
        levels_memory_consumption[i] = {}

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
    previously_visited_level = 0
    while not traversal_queueu.empty():
        current_node = traversal_queueu.get()
        current_level = analysis_graph[current_node].level
        current_group = nodes_groups[current_node]

        if current_node in graph:
            adj_nodes = graph[current_node]
        else:
            adj_nodes = []

        if not current_level in levels_memory_consumption[current_group]:
            levels_memory_consumption[current_group][current_level] = nodes_memory[current_node]
        else:
            levels_memory_consumption[current_group][current_level] += nodes_memory[current_node]

        if previously_visited and previously_visited_level < current_level:
            previously_visited_level = current_level
            previously_visited = []
            for i in range(0, len(previously_visited)):
                previously_visited_node = previously_visited[i]
                levels_memory_consumption[nodes_groups[previously_visited_node]
                                          ][analysis_graph[previously_visited_node].level] -= nodes_memory[previously_visited_node]

        for parent_node in analysis_graph[current_node].parents:
            nodes_ranks[parent_node] -= 1
            if nodes_ranks[parent_node] == 0:
                levels_memory_consumption[nodes_groups[parent_node]
                                          ][current_level] -= nodes_memory[parent_node]

        groups_visited = {}
        number_of_adjacents_in_the_same_group = 0

        for i in range(0, no_of_desired_groups):
            groups_visited[i] = False
        for adj_node in adj_nodes:
            adj_node_group = nodes_groups[adj_node]
            if adj_node_group != current_group:
                if not groups_visited[adj_node_group]:
                    levels_memory_consumption[adj_node_group][current_level +
                                                              1] += nodes_memory[current_node]
                    groups_visited[adj_node_group] = True
            else:
                number_of_adjacents_in_the_same_group += 1

            tmp_nodes_in_degrees[adj_node] -= 1
            if tmp_nodes_in_degrees[adj_node] == 0:
                traversal_queueu.put(adj_node)

        if number_of_adjacents_in_the_same_group == 0:
            previously_visited.append(current_node)
            previously_visited_level = current_level

    return levels_memory_consumption


final_groups_memory_consumptions = get_levels_memory_consumption(graph)

final_groups_comulative_work = []
for i in range(0, no_of_desired_groups):
    final_groups_comulative_work.append({-1: 0})

for level in range(0, no_of_levels):
    for group in range(0, no_of_desired_groups):
        final_groups_comulative_work[group][level] = 0

for level in range(0, no_of_levels):
    for node in levels_nodes[level]:
        final_groups_comulative_work[nodes_groups[node]
                                     ][level] = final_groups_comulative_work[nodes_groups[node]][level - 1] + analysis_graph[node].duration

initial_groups_mem_cons = []
for i in range(0, no_of_desired_groups):
    initial_groups_mem_cons.append([])

for group in initial_groups:
    current_final_group = nodes_groups[group[0]]
    current_group_mem_cons = 0
    for node in group:
        current_group_mem_cons += nodes_memory[node]
    heapq.heappush(
        initial_groups_mem_cons[current_final_group], (current_group_mem_cons, current_group))

current_level = 0
for group_no in range(0, no_of_desired_groups):
    while current_level < no_of_levels:
        if final_groups_memory_consumptions[level] > memory_limit_per_group:
            overflow = final_groups_memory_consumptions[level] - \
                memory_limit_per_group
            candidate_groups = []
            candidate_groups_weights = []
            candidate_group_weight = 0
            sum_of_candidates = 0
            while candidate_group_weight <= overflow and initial_groups_mem_cons[group_no] and sum_of_candidates <= no_of_desired_groups * overflow:
                [candidate_group_weight, candidate_group] = heapq.heappop(
                    initial_groups_mem_cons[group_no])
                candidate_groups.append(candidate_group)
                candidate_groups_weights.append(candidate_group_weight)
                sum_of_candidates += candidate_group_weight

            for sub_group in candidate_groups:
                for i in range(0, len(final_groups)):
                    can_be_merged = True
                    for node in sub_group:
                        node_level = analysis_graph[node].level
                        if final_groups_memory_consumptions[i][node_level + 1] > memory_limit_per_group or \
                            ( len(graph[node]) > 0 and graph[node][0] != sink_node_name and final_groups_memory_consumptions[i][node_level + 2] > memory_limit_per_group):
                            can_be_merged = False
                            break

                        max_neighbor_level = node_level
                        for neighbor_node in graph[node]:
                            if neighbor_node != sink_node_name and nodes_groups[neighbor_node] == nodes_groups[node]:
                                max_neighbor_level = analysis_graph[neighbor_node].level
                            for level in range(node_level + 3, max_neighbor_level):
                                if final_groups_memory_consumptions[i][level] > memory_limit_per_group:
                                    can_be_merged = False
                                    break
                    
                    if can_be_merged:

        level += 1


with open(out1, 'w') as f:
    for i in range(0, len(final_groups)):
        smm = 0
        light_levels_sum = 0
        cntt = 0
        count = 0
        for node in final_groups[i]:
            if not node.startswith("^"):
                f.write(node + ' ' + str(no_of_desired_groups - i - 1) + '\n')
                smm = smm + analysis_graph[node].duration
                if analysis_graph[node].level >= 0 and analysis_graph[node].level < 10:
                    light_levels_sum = light_levels_sum + \
                        analysis_graph[node].duration
                    count = count + 1
                cntt = cntt + 1

        print(str(no_of_desired_groups - i - 1) + ': ' + str(cntt) +
              ', ' + str(smm) + ', ' + str(count) + ', ' + str(light_levels_sum))
    for node in all_nodes:
        if node.startswith('^'):
            f.write(node + ' ' + str(nodes_groups[node]) + '\n')
