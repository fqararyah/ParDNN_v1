import utils
import to_lower
# folder containing the work files
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/time_steps_32_b_4800/'

io_folder_path = utils.io_folder_path

# input files
network_app = utils.network_app
in1 = io_folder_path + 'vanilla_cleaned_low.place'
in2 = io_folder_path + 'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e.place'
in3 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'#'resnet_src_sink_nodes_levels_low.txt'
in5 = io_folder_path + network_app + '_src_sink_low.dot'#'resnet_src_sink_low.dot'
in5_b = io_folder_path + 'rev_' + network_app + '_src_sink_low.dot'#'resnet_src_sink_low.dot'
in6 = io_folder_path + 'colocation_32_low.txt'
in7 = io_folder_path + 'timeline_step17_low.json'


""" # input files
in1 = io_folder_path + 'vanilla_cleaned_low.place'
in2 = io_folder_path + 'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e.place'
in3 = io_folder_path + 'nmt_src_sink_nodes_levels_low.txt'#'resnet_src_sink_nodes_levels_low.txt'
in4 = io_folder_path + 'blacklist_low.txt'
in5 = io_folder_path + 'nmt_src_sink_low.dot'#'resnet_src_sink_low.dot'
in6 = io_folder_path + 'colocation_32_low.txt' """
""" 
# input files
in1 = io_folder_path + 'vanilla_cleaned_low.place'
in2 = io_folder_path + 'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e.place'
in3 = io_folder_path + 'resnet_src_sink_nodes_levels_low.txt'#'resnet_src_sink_nodes_levels_low.txt'
in4 = io_folder_path + 'blacklist_low.txt'
in5 = io_folder_path + 'resnet_src_sink_low.dot'#'resnet_src_sink_low.dot'
in5_b = io_folder_path + 'rev_resnet_src_sink_low.dot'#'resnet_src_sink_low.dot'
in6 = io_folder_path + 'colocation_32_low.txt' """

out1 = io_folder_path + 'placement.place'

nodes_levels = {}


analysis_graph = utils.read_profiling_file(in7, True)

with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        nodes_levels[splits[0]] = int(splits[1])


vanilla_placement = {}
placer_placement = {}

with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        vanilla_placement[splits[0]] = splits[1]

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        placer_placement[splits[0]] = splits[1]


graph = {}
rev_graph = {}
#constructing the graph and initializing the nodes levels from the dot file
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes)  > 1:
            if nodes[0] in graph:
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]

with open(in5_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        nodes = line.split("->")
        if len(nodes)  > 1:
            if nodes[0] in rev_graph:
                rev_graph[nodes[0]].append(nodes[1])
            else:
                rev_graph[nodes[0]] = [nodes[1]]

print("fff")
level = 7
collocated = {}
rev_collocated = {}
with open(in6) as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' vs ')
        if splits[0] not in collocated:
            collocated[splits[0]] = [splits[1]]
        else:
            collocated[splits[0]].append(splits[1])
        if splits[1] not in rev_collocated:
            rev_collocated[splits[1]] = [splits[0]]
        else:
            rev_collocated[splits[1]].append(splits[0])

for collocation_src in collocated.keys():
    #if collocation_src in graph:
    for adj_node in collocated[collocation_src]:#graph[collocation_src]:
        placer_placement[adj_node] = placer_placement[collocation_src]
        for another_src in rev_collocated[adj_node]:
            placer_placement[another_src] = placer_placement[collocation_src]

#print(placer_placement['unit_2_2/conv_2/kernel'])
#print(placer_placement['save/Assign_185'])

#backward adjustment:
for node, adjs in rev_graph.items():
    if node.endswith("/read"):
        placer_placement[node] = placer_placement[node[:-5]]
        for adj_node in adjs:
            placer_placement[adj_node] = placer_placement[node]
            for adj_adj_node in graph[adj_node]:
                if adj_adj_node.endswith("/assign") and nodes_levels[adj_adj_node] <= nodes_levels[node] + 1:
                    placer_placement[adj_adj_node] = placer_placement[node]
                    nodes_to_visit = [adj_adj_node]
                    visited = {}
                    while nodes_to_visit:
                        node_to_visit = nodes_to_visit.pop()
                        if node_to_visit != 'src' and node_to_visit != 'snk' and node_to_visit not in visited:
                            visited[node_to_visit] = 1
                            placer_placement[node_to_visit] = placer_placement[node]
                            nodes_to_visit = nodes_to_visit + rev_graph[node_to_visit]
                if adj_adj_node.split('/')[-1].startswith('apply'):
                    if adj_adj_node =='adam/update_rnnlm/multi_rnn_cell/cell_0/basic_lstm_cell/bias/applyadam':
                        print(placer_placement[node])
                    placer_placement[adj_adj_node] = placer_placement[node]

parts_weights = {}
with open(out1, 'w') as f:
    for node, part in vanilla_placement.items():
        if not node.startswith('^') and node in placer_placement:
            if node in nodes_levels and node in placer_placement:
                if part == '4' or nodes_levels[node] < 0:
                    f.write(node + ' ' + part + '\n')
                else:
                    f.write(node + ' ' + placer_placement[node] + '\n')
                    if placer_placement[node] not in parts_weights:
                        parts_weights[placer_placement[node]] = analysis_graph[node].duration if node in analysis_graph else 1
                    else:
                        parts_weights[placer_placement[node]] += analysis_graph[node].duration if node in analysis_graph else 1
            else:
                f.write(node + ' ' + part + '\n')

print(parts_weights)
