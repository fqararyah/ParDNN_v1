import utils

io_folder_path = utils.io_folder_path
in1 = io_folder_path + 'mem.txt'
in2 = io_folder_path + utils.network_app + '_src_sink_low.dot'
in4 = io_folder_path + 'no_ops.txt'
in5 = io_folder_path + 'operations_attributes.txt'
in6 = io_folder_path + 'tensors_sz_32_low.txt'
out1 = io_folder_path + 'memory.txt'
out1_1 = io_folder_path + 'nf_memory.txt'
out2 = io_folder_path + 'res_memory.txt'
out2_1 = io_folder_path + 'nf_res_memory.txt'

in10 = io_folder_path + 'ref_nodes.txt'

ref_nodes = {}
with open(in10, 'r') as f:
    for line in f:
        ref_nodes[utils.clean_line(line)] = 1

in11 = io_folder_path + 'var_nodes.txt'
var_nodes = {}
with open(in11, 'r') as f:
    for line in f:
        var_nodes[utils.clean_line(line)] = 1

in12 = io_folder_path + 'collocations.txt'
collocations = {}
with open(in12, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("::")
        for node in splits:
            collocations[node] = 1

all_nodes = {}

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            all_nodes[splits[0]] = 1
            all_nodes[splits[1]] = 1

tensors_sizes = {}
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensor_size = int(splitted[1])
        tensor_name = splitted[0]
        tensors_sizes[tensor_name] = tensor_size

no_op_nodes = {}
with open(in4, 'r') as f:
    for line in f:
        no_op_nodes[utils.clean_line(line)] = 1

do_not_check_ops = {}
with open(in5, 'r') as f:
    for line in f:
        splits = utils.clean_line(line).lower().split('::')
        if splits[1] == 'switch' or splits[1] == 'identity' or (len(splits) > 2 and splits[2] == 'true' ):
            do_not_check_ops[splits[0]] = 1

sum_inits = 0

def text_to_bytes(mem_cons):
    node_mem_cons = 0 
    if mem_cons.endswith('GB'):
        node_mem_cons = float(mem_cons[:-2]) * 1024 * 1024 * 1024
    elif mem_cons.endswith('MB'):
        node_mem_cons = float(mem_cons[:-2]) * 1024 * 1024
    elif mem_cons.endswith('KB'):
        node_mem_cons = float(mem_cons[:-2]) * 1024
    elif mem_cons.endswith('B'):
        node_mem_cons = float(mem_cons[:-1])

    return node_mem_cons

nodes_memory = {}
nf_nodes_memory = {}
additional_memory = {}
res_memory = {}
nf_res_memory = {}
leaking = True
with open(in1, 'r') as f:
    for line in f:
        if not '_TFProfRoot' in line:
            line = utils.clean_line_keep_spaces(line)
            splits = line.split('::(')
            if len(splits) < 2:
                splits = line.split(' (')

            if len(splits) > 1:
                node_name = splits[0].lower()
                node_name = utils.clean_line(node_name)

                mem_cons = utils.clean_line(splits[1]).split(',')

                if len(mem_cons) > 2:
                    res_cons = text_to_bytes(mem_cons[2].split('/')[0])
                    if res_cons > 0 and node_name not in var_nodes:#(not leaking or node_name in var_nodes):
                        if node_name in all_nodes:
                            res_memory[node_name] = res_cons
                        else:
                            nf_res_memory[node_name] = res_cons

                mem_cons = mem_cons[-1]
                mem_cons = mem_cons.split('/')[0]

                node_mem_cons = text_to_bytes(mem_cons)
                
                #if node_name in tensors_sizes:
                #    nodes_memory[node_name] = abs(max(tensors_sizes[node_name], node_mem_cons))
                #else:
                if node_name in all_nodes:
                    nodes_memory[node_name] = node_mem_cons
                elif node_mem_cons > 0:
                    nf_nodes_memory[node_name] = node_mem_cons
                    #if node_name in all_nodes and node_mem_cons > 0:
                    #   print(node_mem_cons)
                
                    #print(node_name + ' ' + str(node_mem_cons))
                #if node_mem_cons > 0 and node_name in all_nodes:
                # print(node_name)

                if node_name in all_nodes:
                    all_nodes[node_name] = 0

                if node_name not in all_nodes:
                    if node_name in nodes_memory and nodes_memory[node_name] > 0:
                        print(node_name)
                        sum_inits += nodes_memory[node_name]
            
print(sum_inits/(1024*1024*1024))

""" for node, val in all_nodes.items():
    #found = False
    if (val == 1 or nodes_memory[node] == 0) and not node.startswith('^') and node not in no_op_nodes and node not in do_not_check_ops:
        for node_name in nf_nodes_memory.keys():
            if node in node_name:
                found = True
                nodes_memory[node] = nf_nodes_memory[node_name]
                break 

    if (val == 1 or node not in res_memory or res_memory[node] == 0) and not node.startswith('^') and node not in no_op_nodes and node not in do_not_check_ops:
        for node_name in nf_res_memory.keys():
            if node in node_name:
                res_memory[node] = nf_res_memory[node_name]
                break  """

for node, val in all_nodes.items():
    if val == 1:        
        nodes_memory[node] = 0
smm = 0

"""for node, size in nodes_memory.items():
  if node in tensors_sizes and size > tensors_sizes[node] and node not in ref_nodes:
    print(node + '::' + str(size - (tensors_sizes[node] if node in tensors_sizes else size))) """

""" for node, size in tensors_sizes.items():
    if (node not in nodes_memory or size > nodes_memory[node]) and node not in ref_nodes and not 'control_dependency':# in node and not node.endswith('read'): 
        smm += size - (nodes_memory[node] if node in nodes_memory else 0)
        #print(node + '::' + str(size - (nodes_memory[node] if node in nodes_memory else 0)))
        nodes_memory[node] = size

print(smm/(1024*1024*1024)) """

summ = 0
with open(out1, 'w') as f:
    for key, val in nodes_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')
        summ += val
print(summ/1000000000)

summ = 0
print(len(res_memory))
res_memory= {}
with open(out2, 'w') as f:
    for key, val in res_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')
        summ += val

print(summ/1000000000)


""" with open(out1_1, 'w') as f:
    for key, val in nf_nodes_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')

with open(out2_1, 'w') as f:
    for key, val in nf_res_memory.items():
        f.write(key + '::' + str(int(val)) + '\n') """




"""not node_name.endswith(('transpose', 'transpose_1', 'transpose_2', 'transpose_3', 'transpose_4','transpose_5', \
                                                                'slice', 'slice_1', 'slice_2', 'slice_3', 'slice_4', 'slice_5',\
                                                                'cast',\
                                                                'split','split_1', 'split_2', 'split_3', 'split_4', 'split_5', \
                                                                'concat', 'concat_1', 'concat_2', 'concat_3', 'concat_4',
                                                                'stridedslicegrad', 
                                                                'randomuniform',
                                                                'matmul', 'matmul_1', 'matmul_2', 'matmul_3', 'matmul_4', 
                                                                'pad'))"""