import utils

network_app = utils.network_app
io_folder_path = utils.io_folder_path
# input files
in1 = io_folder_path + utils.network_app + '_src_sink_low.dot'
in2 = io_folder_path + 'op_mem_for.txt'
in3 = io_folder_path + 'res_nodes.txt'
in4 = io_folder_path + 'placement.place'
in5 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
in6 = io_folder_path + 'memory.txt'

out1 = io_folder_path + 'res_nodes_cleaned.txt'
out2 = io_folder_path + 'forwarding_paths.txt'

#forwarding_parent_indices = {}
forwarding_parent = {}

all_nodes = {}
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split("->")
        if len(splits) > 1:
            all_nodes[splits[0]] = 1
            all_nodes[splits[1]] = 1

with open(in2, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split('::')
    exists =True
    if len(splits) > 1:
      parent = ''
      child = ''
      if splits[0] in all_nodes:
        parent = splits[0]
      elif splits[0].split('-')[0] in all_nodes:
        parent = splits[0].split('-')[0]
      else:
        exists = False

      if splits[1] in all_nodes:
        child = splits[1]
      elif splits[1].split('-')[0] in all_nodes:
        child = splits[1].split('-')[0]
      else:
        exists = False
    
      if exists:
        if parent != child:
          forwarding_parent[parent] = child

res_nodes = {}
smm = 0
overall = 0
with open(in3, 'r') as f:
  for line in f:
    line = utils.clean_line(line).lower()
    splits = line.split('::')
    node = ''
    if len(splits) > 2:
      exists = True
      if splits[1] in all_nodes:
        node = splits[1]
      #elif splits[1].split('-')[0] in all_nodes:
      #  node = splits[1].split('-')[0]
      else:
        exists = False
        smm += int(splits[2])

      if (node not in res_nodes or res_nodes[node] < int(splits[2])):
        overall += int(splits[2])
        
      if exists and (node not in res_nodes or res_nodes[node] < int(splits[2])):
        res_nodes[node] = int(splits[2])
      

print('wasted mem::' + str(smm/(1024*1024*1024)))
print('overall mem::' + str(overall/(1024*1024*1024)))

mappings = {}
forwarding_paths = {}
for node in res_nodes:
  src = node
  forwarding_paths[src] = []
  while src in forwarding_parent:
    if forwarding_parent[src] not in res_nodes or res_nodes[forwarding_parent[src]] <= res_nodes[node]:
      forwarding_paths[node].append(forwarding_parent[src])
      src = forwarding_parent[src]
    else:
      break
  
  if len(forwarding_paths[node]) == 0:
    del forwarding_paths[node]

with open(out1, 'w') as f:
  for key, val in res_nodes.items():
    f.write(key + '::' + str(val) + '\n')

placement = {}
with open(in4, 'r') as f:
  for line in f:
    line = utils.clean_line_keep_spaces(line)
    splits = line.split(' ')
    if len(splits) > 1:
      placement[splits[0]] = splits[1]

nodes_levels = {}
with open(in5, 'r') as f:
  for line in f:
    line = utils.clean_line(line)
    splits = line.split('::')
    if len(splits) > 1:
      nodes_levels[splits[0]] = int(splits[1])


nodes_memory = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_memory[node_name] = int(splitted[1])

with open(out2, 'w') as f:
  for key, val in forwarding_paths.items():
    f.write(key + '::')
    iii = 0
    for val_ in val:
      iii += 1
      f.write(val_)
      if iii < len(val):
        f.write('::') 

    f.write('\n')

smm = 0

for node, path in forwarding_paths.items():
  #print(path)
  node = node.lower()
  #if len(path) > 0 and node in nodes_levels and path[-1] in nodes_levels and nodes_levels[path[-1]] - nodes_levels[node] > 1 and \
    #node in placement and placement[node] == '0' and nodes_levels[node] > 3500 and nodes_levels[node] < 5000:
  #if nodes_levels[path[-1]] - nodes_levels[node] > 14 and node in placement and placement[node] == '0' and \
  #if nodes_levels[path[-1]] - nodes_levels[node] > 200:
  if nodes_levels[node] > 3500 and nodes_levels[node] < 5050 and \
    nodes_levels[path[-1]] - nodes_levels[node] > 1500 and node in placement and placement[node] == '0':  
    smm += res_nodes[node]

print(smm/(1000*1000*1000))