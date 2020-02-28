import utils

io_folder_path = utils.io_folder_path
in1 = io_folder_path + 'mem.txt'
in2 = io_folder_path + utils.network_app + '_src_sink_low.dot'
in3 = io_folder_path + 'memory_tensors.txt'
in4 = io_folder_path + 'no_ops.txt'
in5 = io_folder_path + 'operations_attributes.txt'
out1 = io_folder_path + 'memory.txt'
out1_1 = io_folder_path + 'nf_memory.txt'
out2 = io_folder_path + 'res_memory.txt'
out2_1 = io_folder_path + 'nf_res_memory.txt'

all_nodes = {}

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            all_nodes[splits[0]] = 1
            all_nodes[splits[1]] = 1

tensors_sizes = {}
edges_weights = {}
# get tensors sizes
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        tensor_size = int(splitted[1])
        tensor_name = splitted[0].lower()
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
                    if res_cons > 0:
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

for node, val in all_nodes.items():
    found = False
    if (val == 1 or nodes_memory[node] == 0) and not node.startswith('^') and node not in no_op_nodes and node not in do_not_check_ops:
        for node_name in nf_nodes_memory.keys():
            if node in node_name:
                found = True
                nodes_memory[node] = nf_nodes_memory[node_name]
                print(nf_nodes_memory[node_name])
                break

    if (val == 1 or node not in res_memory or res_memory[node] == 0) and not node.startswith('^') and node not in no_op_nodes and node not in do_not_check_ops:
        for node_name in nf_res_memory.keys():
            if node in node_name:
                res_memory[node] = nf_res_memory[node_name]
                print(nf_res_memory[node_name])
                break

    if not found and val == 1:        
        nodes_memory[node] = 0
        
with open(out1, 'w') as f:
    for key, val in nodes_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')

with open(out2, 'w') as f:
    for key, val in res_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')

with open(out1_1, 'w') as f:
    for key, val in nf_nodes_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')

with open(out2_1, 'w') as f:
    for key, val in nf_res_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')