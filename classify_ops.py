import utils

# folder containing the work files
io_folder_path = utils.io_folder_path
network_app = utils.network_app
in1 = io_folder_path + network_app + \
    '_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'operations_attributes.txt'
in3 = io_folder_path + 'vanilla_cleaned.place'

out1 = io_folder_path + 'var_nodes.txt'
out2 = io_folder_path + 'ref_nodes.txt'
out3 = io_folder_path + 'no_ops.txt'
out4 = io_folder_path + 'collocations.txt'

graph = {}
rev_graph = {}
# initializing the nodes and adjacencies from the dot file
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            if splits[0] in graph.keys():
                graph[splits[0]].append(splits[1])
            else:
                graph[splits[0]] = [splits[1]]
            
            if splits[1] in rev_graph.keys():
                rev_graph[splits[1]].append(splits[0])
            else:
                rev_graph[splits[1]] = [splits[0]]

no_ops = {}
ref_ops = {}
var_ops = {}
ops_types = {}
with open(in2, 'r') as f:
    for line in f:
        splits = utils.clean_line(line).lower().split('::')
        ops_types[splits[0]] = splits[1]
        if splits[1] == 'noop':
            no_ops[splits[0]] = 1
        elif len(splits) > 2 and splits[2] == 'true':
            ref_ops[splits[0]] = 1
            if splits[1] in ['variablev2', 'variable']:
                var_ops[splits[0]] = 1

vanilla_placement = {}
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splits = line.split(' ')
        vanilla_placement[splits[0]] = splits[1]

""" for node in graph['rnnlm/softmax_w']:
    if vanilla_placement[node] != '-1':
        print(node)
        for rev_adj in rev_graph[node]:
            print(rev_adj)
print('--------------------')
for node in graph['batch_time']:
    if vanilla_placement[node] != '-1':
        print(node)
        for rev_adj in rev_graph[node]:
            print(rev_adj) """

collocations = {}
collocated = {}
for node in var_ops.keys():
    list_to_add_to = []
    if vanilla_placement[node] != '-1':
        if node not in collocated:
            collocations[node] = []
            list_to_add_to = collocations[node]
            collocated[node] = ''
        else:
            list_to_add_to = collocations[collocated[node]]
        for adj in graph[node]:
            if adj in ref_ops and adj not in collocated and vanilla_placement[adj] != '-1':
                list_to_add_to.append(adj)
                collocated[adj] = node
                for rev_adj in rev_graph[adj]:
                    if rev_adj in ref_ops and rev_adj not in collocated and vanilla_placement[rev_adj] != '-1':
                        list_to_add_to.append(rev_adj)
                        collocated[rev_adj] = node

with open(out1, 'w') as f:
    for var_node in var_ops.keys():
        f.write(var_node + '\n')

with open(out2, 'w') as f:
    for ref_node in ref_ops.keys():
        f.write(ref_node + '\n')

with open(out3, 'w') as f:
    for no_op_node in no_ops.keys():
        f.write(no_op_node + '\n')

print(len(collocations))
with open(out4, 'w') as f:
    for coll_key, collocated in collocations.items():
        _str = coll_key
        for collocated_node in collocated:
            _str += '::' + collocated_node
        _str += '\n'
        f.write(_str)