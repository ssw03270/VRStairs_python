import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import GraphMaker
import GraphMaker as g
import TrajectorySplitter as spl
from GraphMaker import loadData

name = '김미송'
type = 'stair1_50'
num = '2'
files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Lfootdata.txt',
         'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Rfootdata.txt',
         'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistData.txt'}

rotation_files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/LfootRotationData.txt',
         'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/RfootRotationData.txt',
         'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistRotationData.txt'}

output_files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/Lfootdata.txt',
                'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/Rfootdata.txt',
                'neck': 'foot_dataset/user3/' + name + '/' + type + '/WaistData.txt'}

output_rotation_files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/LfootRotationData.txt',
                'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/RfootRotationData.txt',
                'neck': 'foot_dataset/user3/' + name + '/' + type + '/WaistRotationData.txt'}

output_time_file = 'foot_dataset/user3/' + name + '/' + type + '/TimeData.txt'
def test():
    # load position data
    lfoot_data = np.array(loadData(files['lfoot']))
    rfoot_data = np.array(loadData(files['rfoot']))
    neck_data = np.array(loadData(files['neck']))

    # load rotation data
    lfoot_rotation_data = np.array(loadData((rotation_files['lfoot'])))
    rfoot_rotation_data = np.array(loadData((rotation_files['rfoot'])))
    neck_rotation_data = np.array(loadData((rotation_files['neck'])))

    # calculate each step's start & end index
    lfoot_result = spl.FindPoints(lfoot_data[1],NotRemove=False,isDebug=False)
    rfoot_result = spl.FindPoints(rfoot_data[1],NotRemove=False,isDebug=False)
    neck_result = spl.FindPoints(neck_data[1],NotRemove=False,isDebug=False)

    lfoot_result[0] = (150, lfoot_result[0][1])

    rest_frame = 100
    repeat_time = 3

    # make repeat foot data
    new_lfoot_data = np.concatenate((lfoot_data[:,:lfoot_result[0][1]], np.repeat(lfoot_data[:,lfoot_result[0][1]], rest_frame).reshape(3, -1)), axis=1)
    new_rfoot_data = np.concatenate((rfoot_data[:,:rfoot_result[0][1]], np.repeat(rfoot_data[:,rfoot_result[0][1]], rest_frame).reshape(3, -1)), axis=1)

    neck_start_index = []

    foot_repeat_data = lfoot_data[:,lfoot_result[0][0]:lfoot_result[0][1]]
    foot_repeat_data[0] -= foot_repeat_data[0][0]
    foot_repeat_data[1] -= foot_repeat_data[1][0]
    foot_repeat_data[2] -= foot_repeat_data[2][0]

    for i in range(repeat_time):
        current_rright = new_rfoot_data[0][-1]
        current_lright = new_lfoot_data[0][-1]
        current_rheight = new_rfoot_data[1][-1]
        current_lheight = new_lfoot_data[1][-1]
        current_rforward = new_rfoot_data[2][-1]
        current_lforward = new_lfoot_data[2][-1]

        temp_lfoot_data = np.array(foot_repeat_data)
        temp_lfoot_data[0] = current_lright
        temp_lfoot_data[1] += current_lheight
        temp_lfoot_data[2] += current_lforward
        temp_rfoot_data = np.array(foot_repeat_data)
        temp_rfoot_data[0] = current_rright
        temp_rfoot_data[1] += current_rheight
        temp_rfoot_data[2] += current_rforward

        rest_lfoot_data = np.repeat(temp_lfoot_data[:,-1], rest_frame).reshape(3, -1)
        rest_rfoot_data = np.repeat(temp_rfoot_data[:,-1], rest_frame).reshape(3, -1)

        neck_start_index.append(new_rfoot_data.shape[1])
        neck_start_index.append(new_lfoot_data.shape[1])

        new_lfoot_data = np.concatenate((new_lfoot_data, temp_lfoot_data, rest_lfoot_data), axis=1)
        new_rfoot_data = np.concatenate((new_rfoot_data, temp_rfoot_data, rest_rfoot_data), axis=1)

    # make repeat neck data
    neck_repeat_data = np.array(neck_data)[:, rfoot_result[1][0]:rfoot_result[1][1]]

    temp = np.array(neck_data)[2, lfoot_result[0][0]:lfoot_result[0][1]]
    temp = np.concatenate((temp, np.array([temp[-1]])), axis=0)
    neck_repeat_data[2] = temp

    neck_repeat_data[0] -= neck_repeat_data[0][0]
    neck_repeat_data[1] -= neck_repeat_data[1][0]
    neck_repeat_data[2] -= neck_repeat_data[2][0]
    if neck_repeat_data.shape[1] < neck_start_index[1] - neck_start_index[0]:
        rest = (neck_start_index[1] - neck_start_index[0]) - neck_repeat_data.shape[1]
        neck_repeat_data = np.concatenate((neck_repeat_data, np.repeat(neck_repeat_data[:, -1], rest).reshape(3, -1)), axis=1)

    new_neck_data = neck_data[:,:lfoot_result[0][1]]

    for index in neck_start_index:
        current_right = new_neck_data[0][-1]
        current_height = new_neck_data[1][-1]
        current_forward = new_neck_data[2][-1]
        temp_neck_data = np.array(neck_repeat_data)
        temp_neck_data[0] = current_right
        temp_neck_data[1] += current_height
        temp_neck_data[2] += current_forward
        new_neck_data = np.concatenate((new_neck_data[:,:index], temp_neck_data), axis=1)

    # for debug
    plt.clf()
    plt.plot(new_lfoot_data[2])
    plt.plot(new_rfoot_data[2])
    plt.plot(new_neck_data[2])
    plt.show()

    print('start neck y - start rfoot y: ', new_neck_data[1][0] - new_rfoot_data[1][0])
    print('end neck y - end rfoot y: ', new_neck_data[1][-1] - new_rfoot_data[1][-1])

    # make output file
    lfoot_output = convert_data_to_text(new_lfoot_data)
    rfoot_output = convert_data_to_text(new_rfoot_data)
    neck_output = convert_data_to_text(new_neck_data)

    GraphMaker.writeData(output_files['lfoot'], lfoot_output)
    GraphMaker.writeData(output_files['rfoot'], rfoot_output)
    GraphMaker.writeData(output_files['neck'], neck_output)

    # make rotation data init
    new_lfoot_rotation_data = np.zeros((new_lfoot_data.shape))
    new_rfoot_rotation_data = np.zeros((new_rfoot_data.shape))
    new_neck_rotation_data = np.zeros((new_neck_data.shape))
    for i in range(3):
        new_lfoot_rotation_data[i] = lfoot_rotation_data[i][0]
        new_rfoot_rotation_data[i] = rfoot_rotation_data[i][0]
        new_neck_rotation_data[i] = neck_rotation_data[i][0]

    # make output rotation file
    lfoot_rotation_output = convert_data_to_text(new_lfoot_rotation_data)
    rfoot_rotation_output = convert_data_to_text(new_rfoot_rotation_data)
    neck_rotation_output = convert_data_to_text(new_neck_rotation_data)

    GraphMaker.writeData(output_rotation_files['lfoot'], lfoot_rotation_output)
    GraphMaker.writeData(output_rotation_files['rfoot'], rfoot_rotation_output)
    GraphMaker.writeData(output_rotation_files['neck'], neck_rotation_output)

    # make output time file
    time_output = convert_data_to_text(new_lfoot_data, type='time')
    GraphMaker.writeData(output_time_file, time_output)
def convert_data_to_text(data, type='transform'):
    output = ''
    if type == 'transform':
        for i in range(data.shape[1]):
            x = data[0][i]
            y = data[1][i]
            z = data[2][i]

            line = '(' + str(x) + ', ' + str(y) + ', ' + str(z) + ')\n'
            output += line
    elif type == 'time':
        for i in range(data.shape[1]):
            line = str(0.011111) + '\n'
            output += line
    return output

#
# def make():
#     r = g.RecordedData(folder,firstZero=False)
#     f,axes = plt.subplots(2,1)
#
#     #rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) ]
#     yTrajectory = pd.read_csv("test.csv")
#     r.DrawPosAndVelGraph(axes)
#     y50 = yTrajectory[(yTrajectory["bpm"] == 75) & (yTrajectory["method"] == "Ours") & (yTrajectory["stairHeight"] == 0.25)]
#     r.HeightTrajectorySynthesize(np.array(y50["y"]),axes)
#
#     r.writeToTxt1("D:/Desktop/unity/VRStair/footdata/test/")
#     r.DrawPosAndVelGraph(axes)
#     plt.show()

test()