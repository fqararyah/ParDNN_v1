import utils

io_folder_path = utils.io_folder_path
in1 = io_folder_path + 'mem_6.txt'
in2 = io_folder_path + utils.network_app + '_src_sink_low.dot'
out1 = io_folder_path + 'memory_6.txt'

all_nodes = {}

with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            all_nodes[splits[0]] = 1
            all_nodes[splits[1]] = 1

sum_inits = 0

nodes_memory = {}
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        node_name = splits[0].lower()
        if node_name == "gradients/unit_1_0/bn_2/batchnorm/add_1_grad/Sum_1":
            print(1)

        mem_cons = splits = line.split('/')
        mem_cons = mem_cons[len(mem_cons) - 1]
        mem_cons = mem_cons[:-1]
        node_mem_cons = 0
        if mem_cons.endswith('GB'):
            node_mem_cons = float(mem_cons[:-2]) * 1024 * 1024 * 1024
        elif mem_cons.endswith('MB'):
            node_mem_cons = float(mem_cons[:-2]) * 1024 * 1024
        elif mem_cons.endswith('KB'):
            node_mem_cons = float(mem_cons[:-2]) * 1024
        elif mem_cons.endswith('B'):
            node_mem_cons = float(mem_cons[:-1])
        
        nodes_memory[node_name] = node_mem_cons

        if node_name in all_nodes:
            all_nodes[node_name] = 0
        else:
            sum_inits += node_mem_cons
            
print(sum_inits)

for node, val in all_nodes.items():
    if val == 1:
        nodes_memory[node] = 0
        
with open(out1, 'w') as f:
    for key, val in nodes_memory.items():
        f.write(key + '::' + str(int(val)) + '\n')