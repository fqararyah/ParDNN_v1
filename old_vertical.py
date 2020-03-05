data = [1,1,1,1,1,12,12,5,7,6,6,6,4,4,3]
output = []

#version 1:
indx = 0
while indx < len(data):
    #store the current value and its counter
    current_value = data[indx]
    current_value_count = 1
    indx += 1
    #keep going if the next element is equal to the stored one.
    while indx < len(data) and data[indx] == current_value:
        current_value_count += 1
        indx += 1

    output.append(str(current_value_count) + '@' + str(current_value))

print(output)

#version 2:
indx = 0
output = []
while indx < len(data):
    #store the current value and its counter
    current_value = data[indx]
    current_value_count = 1
    indx += 1
    #keep going if the next element is equal to the stored one.
    while indx < len(data) and data[indx] == current_value:
        current_value_count += 1
        indx += 1

    if current_value_count < 3:
        for i in range(0, current_value_count):
            output.append(current_value)
    else:
        output.append(str(current_value_count) + '@' + str(current_value))

print(output)