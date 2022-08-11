import numpy as np
import math
import random

def transformer_rotation():
    x = random.randrange(0, 360)
    y = random.randrange(0, 360)
    z = random.randrange(0, 360)

    x = math.pi * (x / 180)
    y = math.pi * (y / 180)
    z = math.pi * (z / 180)

    transformer_x = np.array([[1, 0,           0,                0],
                              [0, math.cos(x), -1 * math.sin(x), 0],
                              [0, math.sin(x), math.cos(x),      0],
                              [0, 0,           0,                1]])
    transformer_y = np.array([[math.cos(y),      0, math.sin(y), 0],
                              [0,                1, 0,           0],
                              [-1 * math.sin(y), 0, math.cos(y), 0],
                              [0,                0, 0,           1]])
    transformer_z = np.array([[math.cos(z), -1 * math.sin(z), 0, 0],
                              [math.sin(z), math.cos(y),      0, 0],
                              [0,           0,                1, 0],
                              [0,           0,                0, 1]])
    output = transformer_z @ transformer_y @ transformer_x @ np.array([1, 1, 1, 1])
    x, y, z = output[0], output[1], output[2]
    alpha = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= alpha
    y /= alpha
    z /= alpha
    return np.array([x, y, z])

imsi_data = [[], [], [], [], [], []]

folderName = "Data/"
fileName = ["LeftPosition_", "RightPosition_", "RootPosition_",  "LeftRotation_", "RightRotation_", "RootRotation_"]
fileCount = 1

for i in range(len(fileName)):
    for j in range(fileCount):
        f = open(folderName + fileName[i] + str(j + 1) + '.txt', 'r')
        imsi_data[i].append([])
        while True:
            line = f.readline()
            if not line: break
            line = line.replace("(", "").replace(")", "").replace(",", "")
            line = line.split()
            if i < 3:
                line = [float(line[0]), float(line[1]), float(line[2])]
            else:
                line = [math.pi * (float(line[0]) / 180), math.pi * (float(line[1]) / 180), math.pi * (float(line[2]) / 180)]
            imsi_data[i][j].append(line)
        f.close()

foot_train_data = []
foot_train_label = []

for i in range(fileCount):
    left_position_data = imsi_data[0][i][:]
    right_position_data = imsi_data[1][i][:]
    root_position_data = imsi_data[2][i][:]
    left_rotation_data = imsi_data[3][i][:]
    right_rotation_data = imsi_data[4][i][:]
    root_rotation_data = imsi_data[5][i][:]


    for k in range(10, len(left_position_data) - 10):
        a = left_position_data[k - 10:k]
        b = right_position_data[k - 10:k]
        c = root_position_data[k]
        d = left_rotation_data[k - 10:k]
        e = right_rotation_data[k - 10:k]
        f = root_rotation_data[k]

        aa = left_position_data[k:k + 10]
        bb = right_position_data[k:k + 10]
        cc = root_position_data[k]
        dd = left_rotation_data[k:k + 10]
        ee = right_rotation_data[k:k + 10]
        ff = root_rotation_data[k]

        new_foot_data = []
        new_foot_label = []
        for j in range(10):
            new_foot_data.append([a[j][0] - c[0], a[j][1], a[j][2] - c[2], b[j][0] - c[0], b[j][1], b[j][2] - c[2], d[j][0], d[j][1] - f[1], d[j][2], e[j][0], e[j][1] - f[1], e[j][2]])
        for j in range(10):
            new_foot_label.append([aa[j][0] - cc[0], aa[j][1], aa[j][2] - cc[2], bb[j][0] - cc[0], bb[j][1], bb[j][2] - cc[2], dd[j][0], dd[j][1] - ff[1], dd[j][2], ee[j][0], ee[j][1] - ff[1], ee[j][2]])

        # new_foot_data = np.reshape(np.array(new_foot_data), 120)
        new_foot_label = np.reshape(np.array(new_foot_label), 120)
        foot_train_data.append(new_foot_data)
        foot_train_label.append(new_foot_label)

foot_train_data = np.array(foot_train_data)
foot_train_label = np.array(foot_train_label)

print(foot_train_data.shape)
print(foot_train_label.shape)

np.savez('train', foot_train_data = foot_train_data, foot_train_label = foot_train_label)