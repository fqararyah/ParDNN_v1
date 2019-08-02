import utils

# folder containing the work files
io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/'

# input files
in1 = io_folder_path + 'vanilla_cleaned_low.place'
in2 = io_folder_path + 'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e_nc.place'#'ver_grouper_placement_e.place'
in3 = io_folder_path + 'resnet_src_sink_nodes_levels_low.txt'
in4 = io_folder_path + 'blacklist_low.txt'
in5 = io_folder_path + 'resnet_src_sink_low.dot'
in6 = io_folder_path + 'colocation_32_low.txt'

out1 = io_folder_path + 'mixed_placement_v_part_nc.place'

nodes_levels = {}
blacklist = []

with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        nodes_levels[splits[0]] = int(splits[1])

with open(in4, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        blacklist.append(line)

vanilla_placement = {}
h_zoltan_2_placement = {}

with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        vanilla_placement[splits[0]] = splits[1]

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        h_zoltan_2_placement[splits[0]] = splits[1]


graph = {}

#constructing the graph and initializing the nodes levels from the dot file
with open(in5, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        nodes = line.split(" -> ")
        if len(nodes)  > 1:
            if nodes[0] in graph:
                graph[nodes[0]].append(nodes[1])
            else:
                graph[nodes[0]] = [nodes[1]]

level = 7


""" while level >= 0:
    for node, adjs in graph.items():
        changed = False
        if int(nodes_levels[node]) == level:
            for adj in adjs:
                if (adj in h_zoltan_2_placement and h_zoltan_2_placement[adj] == '1')  and  node in vanilla_placement and vanilla_placement[node] == '0':
                    #vanilla_placement[adj] = '0'
                    #h_zoltan_2_placement[adj] = '0'
                    vanilla_placement[node] = '1'
                    changed = True
                    break
        if changed:
            for adj in adjs:
                if (adj in vanilla_placement and vanilla_placement[adj] == '0'):
                    vanilla_placement[adj] = '1'
    level = level - 1 """


collocated = {}
rev_collocated = {}
with open(in6) as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' vs ')
        collocated[splits[0]] = splits[1]
        if splits[1] not in rev_collocated:
            rev_collocated[splits[1]] = [splits[0]]
        else:
            rev_collocated[splits[1]].append(splits[0])

for collocation_src in collocated.keys():
    for adj_node in graph[collocation_src]:
        if vanilla_placement[adj_node] != '4':
            if collocation_src in h_zoltan_2_placement and adj_node in h_zoltan_2_placement:
                if h_zoltan_2_placement[adj_node] != h_zoltan_2_placement[collocation_src]:
                    h_zoltan_2_placement[adj_node] = h_zoltan_2_placement[collocation_src]
                    if adj_node in rev_collocated:
                        for tra_adj_node in rev_collocated[adj_node]:
                            h_zoltan_2_placement[tra_adj_node] = h_zoltan_2_placement[collocation_src] 
            elif adj_node in h_zoltan_2_placement:
                if h_zoltan_2_placement[adj_node] != vanilla_placement[collocation_src]:
                    h_zoltan_2_placement[adj_node] = vanilla_placement[collocation_src]
                    if adj_node in rev_collocated:
                        for tra_adj_node in rev_collocated[adj_node]:
                            h_zoltan_2_placement[tra_adj_node] = vanilla_placement[collocation_src] 
            else:
                if vanilla_placement[adj_node] != vanilla_placement[collocation_src]:
                    vanilla_placement[adj_node] = vanilla_placement[collocation_src]
                    if adj_node in rev_collocated:
                        for tra_adj_node in rev_collocated[adj_node]:                        
                            vanilla_placement[tra_adj_node] = vanilla_placement[collocation_src] 

cntt = 0
with open(out1, 'w') as f:
    for node, part in vanilla_placement.items():
        if node in nodes_levels and node in h_zoltan_2_placement:
            if part == '4' or nodes_levels[node] < 8:
                f.write(node + ' ' + part + '\n')
            else:
                f.write(node + ' ' + h_zoltan_2_placement[node] + '\n')
        else:
            f.write(node + ' ' + part + '\n')

