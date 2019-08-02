import tensorflow as tf
import graphProfiler as gprof
import json
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
    x = tf.Variable(tf.random_normal([1000, 3000], seed=1234), name='x')
    x2 = tf.Variable(tf.random_normal([1000, 3000], seed=1234), name='x')
    y = tf.Variable(tf.random_normal([3000, 1000], seed=1234), name='y')
    z = tf.Variable(tf.random_normal([3000, 1000], seed=1234), name='z')
    w = tf.Variable(tf.random_normal([3000, 1000], seed=1234), name='w')

with tf.device('/device:GPU:1'):
	x_y_mul = tf.matmul(x, y)
with tf.device('/device:GPU:0'):
	with tf.name_scope("result"):
		res = x_y_mul + tf.matmul(x, z) + tf.matmul(x, w) + tf.matmul(x2, y) + tf.matmul(x2, z)

init_op = tf.global_variables_initializer()


def parents(op):
    return set(input.op for input in op.inputs)


def children(op):
    return set(op for out in op.outputs for op in out.consumers())


def get_graph():
    """Creates dictionary {node: {child1, child2, ..},..} for current
    TensorFlow graph. Result is compatible with networkx/toposort"""

    ops = tf.get_default_graph().get_operations()
    return {op: children(op) for op in ops}


def print_tf_graph(graph):
    """Prints tensorflow graph in dictionary form."""
    for node in graph:
        for child in graph[node]:
            print("%s -> %s" % (node.name, child.name))
    print("**********************************")


def plot_graph(G):
    '''Plot a DAG using NetworkX'''

    def mapping(node):
        return node.name

    G = nx.DiGraph(G)
    nx.relabel_nodes(G, mapping, copy=False)
    nx.draw(G, cmap=plt.get_cmap('jet'), with_labels=True)
    plt.show()


def tf_to_dot(graph):
    dot = Digraph()

    for n in graph.as_graph_def().node:
        dot.node(n.name, label=n.name)

        for i in n.input:
            dot.edge(i, n.name)

    return dot

operations_tensors = {}
operations_names = tf.get_default_graph().get_operations()
count1 = 0
count2 = 0
#print([n.type for n in tf.get_default_graph().as_graph_def().node])
for operation in operations_names:
    operation_name = operation.name
    operations_info = tf.get_default_graph().get_operation_by_name(operation_name).values()
    if len(operations_info) > 0:
        if not (operations_info[0].shape.ndims is None):
            operation_shape = operations_info[0].shape.as_list()
            operation_dtype_size = operations_info[0].dtype.size
            if not (operation_dtype_size is None):
                operation_no_of_elements = 1
                for dim in operation_shape:
                    if not(dim is None):
                        operation_no_of_elements = operation_no_of_elements * dim
                total_size = operation_no_of_elements * operation_dtype_size
                operations_tensors[operation_name] = total_size
            else:
                count1 = count1 + 1
        else:
            count1 = count1 + 1
            operations_tensors[operation_name] = -1


    else:
        count2 = count2 + 1
        operations_tensors[operation_name] = -1

#print(count1)
#print(count2)

with open('tensors_szzz.json', 'w') as f:
    json.dump(operations_tensors, f)


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)

    tf.train.write_graph(sess.graph_def, 'savedGraphs', 'helloGraph.pbtxt')
    run_metadata = tf.RunMetadata()
    gg = tf_to_dot(tf.get_default_graph())
    with open('profs/h_dot.dot', 'w') as f:
        f.write(str(gg))

    sess.run(init_op, run_metadata=run_metadata, options=options)
    gprof.profile(run_metadata)

    writer = tf.summary.FileWriter('logs', sess.graph)
    for i in range(1, 100):
        values = sess.run(res, run_metadata=run_metadata, options=options)
        if i % 10 == 3:
            writer.add_run_metadata(run_metadata, str(i))
            gprof.profile(run_metadata, i)
            option = tf.profiler.ProfileOptionBuilder.float_operation()
            option["min_bytes"] = 0
            option["min_micros"] = 0
            option["output"] = 'file:outfile=ooo_ooo.txt'
            option["select"] = ("bytes", "peak_bytes", "output_bytes",
                                "residual_bytes")
            mem = tf.profiler.profile(tf.Graph(), run_meta=run_metadata, cmd="scope", options=option)

    options = tf.profiler.ProfileOptionBuilder.time_and_memory()
    options["min_bytes"] = 0
    options["min_micros"] = 0
    options["output"] = 'file:outfile=ooo.txt'
    options["select"] = ("bytes", "peak_bytes", "output_bytes",
                         "residual_bytes")
    mem = tf.profiler.profile(tf.Graph(), run_meta=run_metadata, cmd="scope", options=options)



   
