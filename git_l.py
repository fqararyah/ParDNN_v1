import utils
import nodeProps
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import collections
import heapq
import random
import queue

# folder containing the work files
io_folder_path = utils.io_folder_path
network_app = utils.network_app
in1 = io_folder_path + network_app + \
    '_src_sink_low.dot'  # 'part_8_1799_src_sink.dot'
in2 = io_folder_path + 'operations_attributes.txt'
in3 = io_folder_path + network_app + '_src_sink_nodes_levels_low.txt'
in6 = io_folder_path + 'memory.txt'
in6_b = io_folder_path + 'res_memory.txt'
in7 = io_folder_path + 'placement.place'
in8 = io_folder_path + 'nf_memory.txt'
in8_b = io_folder_path + 'nf_res_memory.txt'

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
ops_types = {}
var_ops = {}
with open(in2, 'r') as f:
    for line in f:
        splits = utils.clean_line(line).lower().split('::')
        ops_types[splits[0]] = splits[1]
        if splits[1] == 'noop':
            no_ops[splits[0]] = 1
        elif len(splits) > 2 and splits[2] == 'true':
            ref_ops[splits[0]] = 1 
            if splits[1].startswith('variable'):
                var_ops[splits[0]] = 1

nodes_levels = {}
# get nodes levels
with open(in3, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        node_and_level = line.split("::")
        if len(node_and_level) > 1:
            int_node_level = int(node_and_level[1])
            nodes_levels[node_and_level[0]] = int_node_level

nodes_memory = {}
# get memory consumption
with open(in6, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_memory[node_name] = int(splitted[1])

nodes_res_memory = {}
# get memory consumption
with open(in6_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nodes_res_memory[node_name] = int(splitted[1])

nf_nodes_memory = {}
# get memory consumption
with open(in8, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nf_nodes_memory[node_name] = int(splitted[1])

nf_nodes_res_memory = {}
# get memory consumption
with open(in8_b, 'r') as f:
    for line in f:
        line = utils.clean_line(line)
        splitted = line.split('::')
        node_name = splitted[0].lower()
        nf_nodes_res_memory[node_name] = int(splitted[1])

smm = 0
for node, mem in nf_nodes_memory.items():
    smm += mem

print('not found mem cons:: ' + str(smm / (1024 * 1024 * 1024) ))
smm = 0
for node, mem in nf_nodes_res_memory.items():
    smm += mem

print('not found residual mem cons:: ' + str(smm / (1024 * 1024 * 1024) ))

no_of_groups = 0
nodes_groups = {}
with open(in7, 'r') as f:
    for line in f:
        line = utils.clean_line_keep_spaces(line)
        splitted = line.split(' ')
        node_name = splitted[0].lower()
        nodes_groups[node_name] = int(splitted[1])
        if int(splitted[1]) > no_of_groups:
            no_of_groups = int(splitted[1])
        if int(splitted[1]) == -1:
            nodes_groups[node_name] = 0

no_of_groups += 1

collocations = {}
for node in ref_ops.keys():
    for rev_adj in rev_graph[node]:
        if rev_adj in ref_ops:
            if node not in collocations.keys():
                collocations[node] = []
            if not ops_types[rev_adj].endswith(('variable','variablev2')):
                collocations[node].append(str(rev_adj) + ':' + str(ops_types[rev_adj]))

for node, adjs in collocations.items():
    if adjs:
        print(node + '::' + str(adjs))

""" ref_smm = 0
non_ref_smm = 0
ref_count = 0
non_ref_count = 0
collocations = {}
used = {}
for node in no_ops.keys():
    if nodes_levels[node] > 15:
        for rev_adj in rev_graph[node]:
            if rev_adj not in used and nodes_groups[rev_adj] == 0:
                if node not in collocations.keys():
                    collocations[node] = []
                collocations[node].append(str(rev_adj) + ':' + str(rev_adj in ref_ops))
                if rev_adj in ref_ops:
                    ref_smm += nodes_memory[rev_adj]
                    ref_count += 1
                else:
                    non_ref_smm += nodes_memory[rev_adj]
                    non_ref_count += 1

            used[rev_adj] = 1

print(ref_count)
print(non_ref_count)

print(ref_smm / (1024 * 1024 * 1024))
print(non_ref_smm / (1024 * 1024 * 1024))

var_ops_sums = [0] * 4
for node in graph.keys():
    add = False
    if node not in ref_ops:
        for adj in graph[node]:
            if nodes_groups[adj] == nodes_groups[node]:
                add = True
                print(str(nodes_levels[node]) + '::' + str(nodes_levels[adj]) + '::' + str(nodes_memory[node]))
        if add:
            var_ops_sums[nodes_groups[node]] += nodes_memory[node]

print('--------------------')
for var_ops_sum in var_ops_sums:
    print(var_ops_sum / (1024 * 1024 * 1024))

print(len(ref_ops))
smm = 0
for op, op_type in ops_types.items():
    if op_type.endswith(('variablev2', 'variable')) and nodes_groups[op] == 0: 
        smm+= nodes_memory[op]"""

""" count = 0
non_ref_nodes_with_no_op_child = {}
for node in nodes_res_memory.keys():
    if node not in ref_ops:
        print( node + '::' + str(nodes_res_memory[node]) + '::' + str(nodes_levels[node]) + '::' + str(ops_types[node]) )

print(count)
for node, count in non_ref_nodes_with_no_op_child.items():
    print(node) """

var_count = 0
ref_count = 0
res_count = 0
norm_count = 0

var_sum = 0
ref_sum = 0
res_sum = 0
norm_sum = 0

for node, mem in nodes_memory.items():
    if mem > 0 and nodes_groups[node] == 6:
        if node in var_ops:
            var_count += 1
            var_sum += mem
        elif node in ref_ops:
            ref_count += 1
            ref_sum += mem
        elif node in nodes_res_memory:
            res_count += 1
            res_sum += mem
        else:
             norm_count += 1
             norm_sum += mem

print('-----------------------')
print('var_count: ' + str(var_count)) 
print('ref_count: ' + str(ref_count)) 
print('res_count: ' + str(res_count)) 
print('norm_count: ' + str(norm_count)) 

print('var_sum: ' + str( var_sum / (1024 * 1024 * 1024) )) 
print('ref_sum: ' + str(ref_sum / (1024 * 1024 * 1024) )) 
print('res_sum: ' + str(res_sum / (1024 * 1024 * 1024) )) 
print('norm_sum: ' + str(norm_sum / (1024 * 1024 * 1024) )) 
print('-----------------------')

""" for node in rev_graph.keys():
    if node in ref_ops:
        for rev_adjs in rev_graph[node]:
            if rev_adj not in ref_ops and not rev_adj.startswith('^') and rev_adj not in no_ops:
                print(rev_adj)  """

levels = {}
for node in graph.keys():
    if node not in ref_ops and node not in var_ops and node not in nodes_res_memory and nodes_memory[node] > 0:
        for adj in graph[node]:
            if adj in no_ops or adj.startswith('^'):
                #print(node + '::' + adj + '::' + str(nodes_memory[node]) + ' :: ' + str(nodes_levels[node]) )
                if nodes_levels[node] not in levels:
                    levels[nodes_levels[node]] = 0
                levels[nodes_levels[node]] += nodes_memory[node]

""" for level, count in levels.items():
    if count > 1:
        print(str(level) + '::' + str(count))  """

smm = 0
for node in rev_graph['snk']:
    if (node in nodes_memory and nodes_memory[node] > 0):
        smm += nodes_memory[node]
    if (node in nodes_res_memory and nodes_res_memory[node] > 0):
        smm += nodes_res_memory[node]
    if node.startswith('^'):
        print(1)

print(smm)
#print(smm / (1024*1024*1024))