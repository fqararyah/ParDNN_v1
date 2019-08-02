def clean_line(node_string):
    return (node_string.strip(';\n')).replace('"', '').replace('\t', '')

black_list = {}
with open('C:/Users/fareed/PycharmProjects/tensorboardAmnist/resnet/resnet/blacklist_and_dot.txt', 'r') as f:
    for line in f:
        line = clean_line(line)
        if line.split(": ")[0] == 'blacklist':
            black_list[line.split(": ")[1].split(" ")[0]] = "1"


with open("C:/Users/fareed/PycharmProjects/tensorboardAmnist/resnet/resnet/blackList.txt", "w") as f:
    for node, smth in black_list.items():
        f.write(node + "\n")


non_black_list = {}
with open('C:/Users/fareed/PycharmProjects/tensorboardAmnist/resnet/resnet/blacklist_and_dot.txt', 'r') as f:
    for line in f:
        line = clean_line(line)
        if line.split(": ")[0] == 'non-blacklist':
            non_black_list[line.split(": ")[1].split(" ")[0]] = "1"


with open("C:/Users/fareed/PycharmProjects/tensorboardAmnist/resnet/resnet/non-blackList.txt", "w") as f:
    for node, smth in non_black_list.items():
        f.write(node + "\n")


actual_dot = {}
with open('C:/Users/fareed/PycharmProjects/tensorboardAmnist/resnet/resnet/blacklist_and_dot.txt', 'r') as f:
    for line in f:
        line = clean_line(line)
        nodes = line.split("->")
        if len(nodes) > 1:
            src = nodes[0].strip()
            dst = nodes[1].strip()
            if src != "_SOURCE" and dst != "_SINK":
                if src in actual_dot:
                     if not(dst in actual_dot[src]):
                        actual_dot[src].append(dst)
                else:
                    actual_dot[src] = [dst]

.dot
with open("C:/Users/fareed/PycharmProjects/tensorboardAmnist/resnet/resnet/widedeep_dot.dot", "w") as f:
    for src, dst in actual_dot.items():
        for node in dst:
            f.write('"' + src + '" -> "' + node + '"\n')
