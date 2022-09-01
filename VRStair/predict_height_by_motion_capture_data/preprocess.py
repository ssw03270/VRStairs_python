import math
import random

import numpy as np
from scipy.signal import savgol_filter

def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data -= mean
    data /= std + 1e-13
    return data

folder_real = 'data'
file_name = ['WaistData', 'Lfootdata', 'Rfootdata', 'MeanData', 'ForwardData']
file_type = '.txt'

height_data = [[], [], [], [], []]

for i, file in enumerate(file_name):
    f = open(folder_real + '/' + file + file_type, 'r')

    while True:
        line = f.readline()
        if not line: break
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        height_data[i].append([float(line[0]), float(line[1]), float(line[2])])
    f.close()

train_data = []
train_label = []

# for i in range(len(height_data[0]) - 10):
#     train_data.append(np.concatenate((height_data[1][i:i+10], height_data[2][i:i+10]), axis=1))
#     train_label.append(height_data[0][i])

height_data = np.array(height_data)

frame_len = 10
for i in range(frame_len, len(height_data[0])):
    forward = np.stack((height_data[4, i-frame_len:i, 0], height_data[4, i-frame_len:i, 2]), axis=1)
    forward = forward.reshape((-1))

    height = np.concatenate((height_data[1][i-frame_len:i], height_data[2][i-frame_len:i]), axis=0)
    height = height[:, 1:2].reshape((-1))

    distance = height_data[1][i-frame_len:i] - height_data[2][i-frame_len:i]
    # distance = np.power(distance[:, 0], 2) + np.power(distance[:, 1], 2) + np.power(distance[:, 2], 2)
    distance = distance.reshape((-1))

    train_data.append(np.concatenate((forward, height, distance), axis=0))

    mean = np.array(height_data[3][i])
    # label = np.array([height_data[0][i][0] - mean[0], height_data[0][i][1], height_data[0][i][2] - mean[2]])
    # train_label.append(label)
    label = np.array([height_data[0][i][1]])
    train_label.append(label)

# for i in range(len(height_data[0])):
#     train_data.append(height_data[1][i] + height_data[2][i])
#     train_label.append(height_data[0][i])

train_data = np.array(train_data)
train_label = np.array(train_label)

imsi_train_data = train_data
imsi_train_label = train_label
input_size = 7
for i in range(1, 21):
    data = np.zeros(frame_len * input_size)
    for j in range(frame_len):
        data[j * input_size + 2] = i / 3
        data[j * input_size + 3] = i / 3
    train_data = np.concatenate((train_data, imsi_train_data + data))

    data2 = i / 3
    train_label = np.concatenate((train_label, imsi_train_label + data2))

# print("height")
# # 높이 데이터 늘리기
# data_len = train_data.shape[0]
# imsi_train_data = train_data[:]
# imsi_train_label = train_label[:]
# for i in range(1, 20):
#     add_vector = i / 10
#     add_vector2 = np.array([0, i / 10])
#     train_data = np.concatenate((train_data, imsi_train_data + add_vector), axis=0)
#     train_label = np.concatenate((train_label, imsi_train_label + add_vector2), axis=0)
#
# print("left right")
# # 좌우 데이터 늘리기
# data_len = train_data.shape[0]
# imsi_train_data = train_data[:]
# imsi_train_label = train_label[:]
# for i in range(-3, 0):
#     add_vector = np.array([i / 3, 0, 0])
#     train_data = np.concatenate((train_data, imsi_train_data + add_vector), axis=0)
#     train_label = np.concatenate((train_label, imsi_train_label), axis=0)
#
# print("forward backward")
# # 앞뒤 데이터 늘리기
# data_len = train_data.shape[0]
# imsi_train_data = train_data[:]
# imsi_train_label = train_label[:]
# for i in range(-3, 0):
#     add_vector = np.array([0, 0, i / 3])
#     train_data = np.concatenate((train_data, imsi_train_data + add_vector), axis=0)
#     train_label = np.concatenate((train_label, imsi_train_label), axis=0)

train_data = train_data.reshape((train_data.shape[0], -1))

print(train_data.shape)
print(train_label.shape)

np.savez('train', foot_train_data = train_data, foot_train_label = train_label)


