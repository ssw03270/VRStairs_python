import numpy as np
import os
import random
import math
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D

l_file = 'LeftFootVelocity_'
r_file = 'RightFootVelocity_'
file_list = ['HeadPosition_1.txt', 'HeadRotation_1.txt', 'HeadVelocity_1.txt', 'LeftFootPosition_1.txt', 'LeftFootRotation_1.txt',
             'LeftFootVelocity_1.txt', 'RightFootPosition_1.txt', 'RightFootRotation_1.txt', 'RightFootVelocity_1.txt', 'RootPosition_1.txt',
             'RootRotation_1.txt', 'RootVelocity_1.txt', 'WaistPosition_1.txt', 'WaistRotation_1.txt', 'WaistVelocity_1.txt']

folder = "Data/"

LposX = []
LposY = []
LposZ = []
RposX = []
RposY = []
RposZ = []

Data = []

for file in file_list:
    imsi = []
    f = open(folder + file, 'r')
    line = f.readline()
    xStep = []
    yStep = []
    zStep = []
    while True:
        line = f.readline()
        if not line: break
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        xStep.append(float(line[0]))
        yStep.append(float(line[1]))
        zStep.append(float(line[2]))

    Data.append(xStep.copy())
    Data.append(yStep.copy())
    Data.append(zStep.copy())

    f.close()

numData = 1
for i in range(1, numData + 1):
    f = open(folder + l_file + str(i) + '.txt', 'r')
    line = f.readline()
    xStep = []
    yStep = []
    zStep = []
    while True:
        line = f.readline()
        if not line: break
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        xStep.append(float(line[0]))
        yStep.append(float(line[1]))
        zStep.append(float(line[2]))
    LposX.append(xStep.copy())
    LposY.append(yStep.copy())
    LposZ.append(zStep.copy())
    f.close()

    f = open(folder + r_file + str(i) + '.txt', 'r')
    line = f.readline()
    xStep = []
    yStep = []
    zStep = []
    while True:
        line = f.readline()
        if not line: break
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        xStep.append(float(line[0]))
        yStep.append(float(line[1]))
        zStep.append(float(line[2]))
    RposX.append(xStep.copy())
    RposY.append(yStep.copy())
    RposZ.append(zStep.copy())

    f.close()

left_gait_start = []
left_gait_end = []
right_gait_start = []
right_gait_end = []

left_last_end = 0
right_last_end = 0

l = np.abs(np.array(LposX[0][10:])) + np.abs(np.array(LposY[0][10:])) + np.abs(np.array(LposZ[0][10:]))
l_smooth = savgol_filter(l, 51, 3)

r = np.abs(np.array(RposX[0][10:])) + np.abs(np.array(RposY[0][10:])) + np.abs(np.array(RposZ[0][10:]))
r_smooth = savgol_filter(r, 51, 3)

limit_start_l = 10
limit_end_l = len(l_smooth) - 10
limit_start_r = 10
limit_end_r = len(r_smooth) - 10

for i in range(10, len(l_smooth) - 10):
    j = len(l_smooth) - i
    if i > limit_start_l and l_smooth[i + 10] - l_smooth[i] > 0.5 and l_smooth[i] < 0.5:
        left_gait_start.append(i)
        limit_start_l = i + 30
    if j < limit_end_l and l_smooth[j - 10] - l_smooth[j] > 0.1 and l_smooth[j] < 0.5:
        left_gait_end.append(j)
        limit_end_l = j - 30
left_gait_end = sorted(left_gait_end)

l_start_i = len(left_gait_start) - 1
l_end_i = len(left_gait_end) - 1

real_left_gait_start = []
real_left_gait_end = []

while True:
    if l_start_i < 0 or l_end_i < 0:
        break
    l_start = left_gait_start[l_start_i]
    l_end = left_gait_end[l_end_i]

    if l_end > l_start:
        real_left_gait_start.append(l_start)
        real_left_gait_end.append(l_end)
        l_end_i -= 1
        l_start_i -= 1
        while l_start < left_gait_end[l_end_i]:
            l_end_i -= 1
            if l_end_i < 0:
                break

for i in range(10, len(r_smooth) - 10):
    j = len(r_smooth) - i
    if i > limit_start_r and r_smooth[i + 10] - r_smooth[i] > 0.5 and r_smooth[i] < 0.5:
        right_gait_start.append(i)
        limit_start_r = i + 30
    if j < limit_end_r and r_smooth[j - 10] - r_smooth[j] > 0.1 and r_smooth[j] < 0.5:
        right_gait_end.append(j)
        limit_end_r = j - 30
right_gait_end = sorted(right_gait_end)

r_start_i = len(right_gait_start) - 1
r_end_i = len(right_gait_end) - 1

real_right_gait_start = []
real_right_gait_end = []

while True:
    if r_start_i < 0 or r_end_i < 0:
        break
    r_start = right_gait_start[r_start_i]
    r_end = right_gait_end[r_end_i]

    if r_end > r_start:
        real_right_gait_start.append(r_start)
        real_right_gait_end.append(r_end)
        r_end_i -= 1
        r_start_i -= 1
        while r_start < right_gait_end[r_end_i]:
            r_end_i -= 1
            if r_end_i < 0:
                break

# foot_time = np.arange(len(LposY[0]))
# plt.figure(figsize=(500, 5))
# plt.title('Trajectory')
# plt.xlabel('Time')
# plt.ylabel('Height')
# plt.plot(foot_time[10:], l_smooth)
# plt.plot(foot_time[10:], r_smooth)
#
#
# real_left_gait_start = sorted(real_left_gait_start)
# real_left_gait_end = sorted(real_left_gait_end)
# real_right_gait_start = sorted(real_right_gait_start)
# real_right_gait_end = sorted(real_right_gait_end)
# for i in range(len(real_left_gait_start)):
#     plt.scatter(real_left_gait_start[i],0, c="b")
#
# for i in range(len(real_left_gait_end)):
#     plt.scatter(real_left_gait_end[i], 0, c="g")
#
#
# for i in range(len(real_right_gait_start)):
#     plt.scatter(real_right_gait_start[i], 0, c="r")
#
# for i in range(len(real_right_gait_end)):
#     plt.scatter(real_right_gait_end[i], 0, c="y")
# plt.show()

data_edited = []
for start, end in zip(real_left_gait_start, real_left_gait_end):
    step = []
    for data in Data:
        imsi = []
        if end - start < 90:
            imsi = data[start + 10:end].copy()
        else:
            imsi = data[start + 10:start + 90].copy()
        step.append(imsi)
    data_edited.append(step)

for start, end in zip(real_right_gait_start, real_right_gait_end):
    step = []
    for data in Data:
        imsi = []
        if end - start < 90:
            imsi = data[start + 10:end].copy()
        else:
            imsi = data[start + 10:start + 90].copy()
        step.append(imsi)
    data_edited.append(step)

# plt.figure(figsize=(5, 5))
# for i in range(30):
#     plt.title('Trajectory')
#     plt.xlabel('Time')
#     plt.ylabel('Height')
#     foot_time = np.arange(len(data_edited[i][10]))
#     plt.plot(foot_time, data_edited[i][10])
#     plt.plot(foot_time, data_edited[i][19])
#     plt.show()

train_data = []
train_label = []
