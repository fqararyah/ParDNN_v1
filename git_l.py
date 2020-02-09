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
with open(in2, 'r') as f:
    for line in f:
        splits = utils.clean_line(line).lower().split('::')
        if splits[1] == 'noop':
            no_ops[splits[0]] = 1
        elif len(splits) > 2 and splits[2] == 'true':
            ref_ops[splits[0]] = 1 

collocations = {}
for node in ref_ops.keys():
    for rev_adj in rev_graph[node]:
        if rev_adj in ref_ops:
            if node not in collocations.keys():
                collocations[node] = []
            collocations[node].append(rev_adj)

for node, adjs in collocations.items():
    print(node + '::' + str(adjs))