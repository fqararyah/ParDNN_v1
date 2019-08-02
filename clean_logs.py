io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/inc/'

in1 = io_folder_path + 'doga_place.txt'

out = io_folder_path + 'doga_place.place'

def clean_line(node_string):
    return (node_string.strip(';\n')).replace('"', '').replace('\t', '')

actual_placement = {}
with open(in1, 'r') as f:
    for line in f:
        line = clean_line(line)
        splits = line.split(" ")
        if len(splits) > 2:
            device = splits[2]
            if device.endswith('CPU:0'):
                device = '4'
            elif device.endswith('GPU:0'):
                device = '0'
            elif device.endswith('GPU:1'):
                device = '1'
            elif device.endswith('GPU:2'):
                device = '2'
            elif device.endswith('GPU:3'):
                device = '3'
            actual_placement[splits[0][:-1]] = device

with open(out, 'w') as f:
    for key, val in actual_placement.items():
        f.write(key + ' ' + val + '\n')

