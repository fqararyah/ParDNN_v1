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
        line = line.lower()
        splits = line.split(" ")
        device_n = ''
        if len(splits) > 2:
            device = splits[2]
            if device.endswith('cpu:0'):
                device_n = '-1'
            elif device.endswith('gpu:0'):
                device_n = '0'
            elif device.endswith('gpu:1'):
                device_n = '1'
            elif device.endswith('gpu:2'):
                device_n = '2'
            elif device.endswith('gpu:3'):
                device_n = '3'
            elif device.endswith('gpu:4'):
                device_n = '4'
            elif device.endswith('gpu:5'):
                device_n = '5'
            elif device.endswith('gpu:6'):
                device_n = '6'
            elif device.endswith('gpu:7'):
                device_n = '7'
            elif device.endswith('gpu:8'):
                device_n = '8'
            elif device.endswith('gpu:9'):
                device_n = '9'
            elif device.endswith('gpu:10'):
                device_n = '10'
            elif device.endswith('gpu:11'):
                device_n = '11'
            elif device.endswith('gpu:12'):
                device_n = '12'
            elif device.endswith('gpu:13'):
                device_n = '13'
            elif device.endswith('gpu:14'):
                device_n = '14'
            elif device.endswith('gpu:15'):
                device_n = '15'
            
            if device_n != '' and (splits[0][:-1] not in actual_placement or device_n == '-1'):
                actual_placement[splits[0][:-1]] = device_n

with open(out, 'w') as f:
    for key, val in actual_placement.items():
        f.write(key + ' ' + val + '\n')

