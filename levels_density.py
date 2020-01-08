import utils

io_folder_path= 'C:/Users/fareed/PycharmProjects/tf_project/inc/rnn/'

in1 = io_folder_path + 'rnn_src_sink_nodes_levels_low.txt'
in2 = io_folder_path + 'memory.txt'

out1 = io_folder_path + 'levels_densities_6.txt'

mem_sum = 0
nodes_memory = {}
# get memory consumption
with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        nodes_memory[splitted[0]] = splitted[1]

levels_density = {}
levels_density_memory = {}
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        if len(splits) > 1:
            if splits[1] in levels_density:
                levels_density[splits[1]] += 1
                levels_density_memory[splits[1]] += int(nodes_memory[splits[0]]) if splits[0] in nodes_memory else 0
            else:
                levels_density[splits[1]] = 1
                levels_density_memory[splits[1]] = int(nodes_memory[splits[0]]) if splits[0] in nodes_memory else 0
        #mem_sum += int(nodes_memory[splits[0]])

print("total memory consumption of the model is: " + str(mem_sum))

levels = list(levels_density.keys())
densities = list(levels_density.values())
densities_memory = list(levels_density_memory.values())

densities, densities_memory, levels = (list(t) for t in zip(
                *sorted(zip( densities, densities_memory, levels), reverse=True)))

with open(out1, 'w') as f:
    for i in range(0, len(levels)):
        f.write('-' + levels[i]+'::' + str(densities[i]) + '::' + str(densities_memory[i]) + '\n')

