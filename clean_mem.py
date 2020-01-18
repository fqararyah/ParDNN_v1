import utils

io_folder_path = utils.io_folder_path
in1 = io_folder_path + 'mem.txt'
in2 = io_folder_path + utils.network_app + '_src_sink_low.dot'
in3 = io_folder_path + 'memory_tensors.txt'
out1 = io_folder_path + 'memory.txt'

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
        tensor_name = splitted[0]
        tensors_sizes[tensor_name] = tensor_size

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
additional_memory = {}
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
                total_cons = mem_cons[0]
                total_cons = total_cons.split('/')[0]
                mem_cons = mem_cons[len(mem_cons) - 1]
                mem_cons = mem_cons.split('/')[0]

                node_mem_cons = text_to_bytes(mem_cons)
                total_cons = text_to_bytes(total_cons)
                additional_mem_cons = max(total_cons - node_mem_cons, 0)
                
                if node_name in tensors_sizes:
                    if node_mem_cons > 0:
                        nodes_memory[node_name] = tensors_sizes[node_name]
                    else:
                        nodes_memory[node_name] = 0
                else:
                    nodes_memory[node_name] = node_mem_cons
                    #if node_name in all_nodes and node_mem_cons > 0:
                     #   print(node_mem_cons)
                additional_memory[node_name] = additional_mem_cons
                
                if node_name in all_nodes:
                    sum_inits += node_mem_cons
                    #print(node_name + ' ' + str(node_mem_cons))
                #if node_mem_cons > 0 and node_name in all_nodes:
                # print(node_name)

                if node_name in all_nodes:
                    all_nodes[node_name] = 0
            
print(sum_inits/1000000000)

for node, val in all_nodes.items():
    if val == 1:
        if node in tensors_sizes:
            nodes_memory[node] = tensors_sizes[node]
            print('fffff')
        else:
            nodes_memory[node] = 0
        additional_memory[node] = 0
        
with open(out1, 'w') as f:
    for key, val in nodes_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')