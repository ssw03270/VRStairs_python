import numpy as np
import math
import random

imsi_data = [[], [], [], []]

folderName = "Data/"
fileName = ["LeftFootPosition_", "RightFootPosition_", "WaistPosition_",  "HeadPosition_"]
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
            line = [float(line[0]), float(line[1]), float(line[2])]
            imsi_data[i][j].append(line)
        f.close()

foot_train_data = []
foot_train_label = []

for i in range(fileCount):
    left_position_data = imsi_data[0][i][:]
    right_position_data = imsi_data[1][i][:]
    waist_position_data = imsi_data[2][i][:]
    head_position_data = imsi_data[3][i][:]


    for k in range(1000, len(left_position_data) - 1000):
        a = left_position_data[k - 10:k]
        b = right_position_data[k - 10:k]
        c = waist_position_data[k]
        d = head_position_data[k]

        new_foot_data = []
        for j in range(10):
            new_foot_data.append([a[j][1], b[j][1]])
        foot_train_label.append([c[1], d[1]])
        new_foot_data = np.reshape(np.array(new_foot_data), 20)
        foot_train_data.append(new_foot_data)

foot_train_data = np.array(foot_train_data)
foot_train_label = np.array(foot_train_label)

print(foot_train_data.shape)
print(foot_train_label.shape)

np.savez('train', foot_train_data = foot_train_data, foot_train_label = foot_train_label)