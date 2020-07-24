import utils

io_folder_path= 'C:/Users/fareed/PycharmProjects/tf_project/inc/e3d/'

in1 = io_folder_path + 'e3d_src_sink_nodes_levels_low.txt'
in2 = io_folder_path + 'memory.txt'

out1 = io_folder_path + 'levels_densities_6.txt'

mem_sum = 0
nodes_memory = {}
mem_hist = {}
# get memory consumption
with open(in2, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        nodes_memory[splitted[0]] = int(splitted[1])
        if float(splitted[1]) / (1024 * 1024) not in mem_hist:
            mem_hist[float(splitted[1]) / (1024 * 1024)] = 0
        mem_hist[float(splitted[1]) / (1024 * 1024)] += 1

lst1 = mem_hist.keys()
lst2 = mem_hist.values()
lst1, lst2 = (list(t) for t in zip(
                *sorted(zip( lst1, lst2), reverse=True)))
""" for i in range(0, len(lst1)):
    print(str(lst1[i]) + ' : ' + str(lst2[i])) """

levels_density = {}
levels_density_memory = {}
with open(in1, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splits = line.split('::')
        if len(splits) > 1:
            if int(splits[1]) in levels_density:
                levels_density[int(splits[1])] += 1
                levels_density_memory[int(splits[1])] += int(nodes_memory[splits[0]]) if splits[0] in nodes_memory else 0
            else:
                levels_density[int(splits[1])] = 1
                levels_density_memory[int(splits[1])] = int(nodes_memory[splits[0]]) if splits[0] in nodes_memory else 0
        #mem_sum += int(nodes_memory[splits[0]])

print("total memory consumption of the model is: " + str(mem_sum))

no_levels = len(levels_density)
densities_hist = [0] * 20
for level, density in levels_density.items():
    for i in range(min(19, density), 0, -1):
        densities_hist[i] += 1

for i in range(1, 5):
    print(str(pow(2, i)) + ' : ' + str(densities_hist[pow(2, i)] / no_levels))

levels = []
densities = []
densities_memory = []

for level in levels_density.keys():
    levels.append(level)
    densities.append(levels_density[level])
    densities_memory.append(levels_density_memory[level])

densities, densities_memory, levels = (list(t) for t in zip(
                *sorted(zip( densities, densities_memory, levels), reverse=True)))

with open(out1, 'w') as f:
    for i in range(0, len(levels_density)):
        f.write('-' + str(levels[i]) +'::' + str(densities[i]) + '::' + str(densities_memory[i]/1000000000 if i in levels_density_memory else 0) + '\n')


densities_memory, levels = (list(t) for t in zip(
                *sorted(zip( densities_memory, levels), reverse=True)))

for i in range(10):
  print(str(densities_memory[i]/1000000000) + '::' + str(levels[i]))