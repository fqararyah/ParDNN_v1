import utils
#io_folder_path = 'C:/Users/fareed/PycharmProjects/tf_project/resnet/winter_34_my_timing/'

io_folder_path = utils.io_folder_path

in1 = io_folder_path + 'act_place.place'

out = io_folder_path + 'vanilla_cleaned.place'

def clean_line(node_string):
    return (node_string.strip(';\n')).replace('"', '').replace('\t', '')

actual_placement = {}
with open(in1, 'r', encoding="utf8") as f:
    for line in f:
        line = clean_line(line)
        splits = line.split(" ")
        device_n = ''
        if len(splits) > 2:
            device = splits[2]
            if device.endswith('CPU:0'):
                device_n = '4'
            elif device.endswith('GPU:0'):
                device_n = '0'
            elif device.endswith('GPU:1'):
                device_n = '1'
            elif device.endswith('GPU:2'):
                device_n = '2'
            elif device.endswith('GPU:3'):
                device_n = '3'
            
            if device_n != '' and (splits[0][:-1] not in actual_placement or device_n == '4'):
                actual_placement[splits[0][:-1]] = device_n

with open(out, 'w') as f:
    for key, val in actual_placement.items():
        f.write(key + ' ' + val + '\n')

