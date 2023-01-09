import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

import GraphMaker
import GraphMaker as g
import TrajectorySplitter as spl
from GraphMaker import loadData

def test(isDebug=False, makeRotation=False):
    # load position data
    lfoot_data = np.array(loadData(files['lfoot']))
    rfoot_data = np.array(loadData(files['rfoot']))
    neck_data = np.array(loadData(files['neck']))

    new_lfoot_data = [[], [], []]
    new_rfoot_data = [[], [], []]
    new_neck_data = [[], [], []]

    # denoise all-axis
    for data, new_data in zip([lfoot_data, rfoot_data, neck_data], [new_lfoot_data, new_rfoot_data, new_neck_data]):
        for i in range(3):
            if i != 1:
                new_data[i] = savgol_filter(data[i], 50, 2)
            else:
                new_data[i] = data[i]

    # change list to nparray
    new_lfoot_data = np.array(new_lfoot_data)
    new_rfoot_data = np.array(new_rfoot_data)
    new_neck_data = np.array(new_neck_data)

    lfoot_result = spl.FindPoints(new_lfoot_data[1], NotRemove=True,isDebug=isDebug)
    rfoot_result = spl.FindPoints(new_rfoot_data[1], NotRemove=True,isDebug=isDebug)
    rfoot_result[1] = rfoot_result[2]

    # denoise x-axis to set offset
    new_rfoot_data[0][lfoot_result[0][0]:lfoot_result[0][1]] = new_rfoot_data[0][lfoot_result[0][0]]
    new_lfoot_data[0][rfoot_result[0][0]:rfoot_result[0][1]] = new_lfoot_data[0][rfoot_result[0][0]]
    new_lfoot_data[0][rfoot_result[1][0]:rfoot_result[1][1]] = new_lfoot_data[0][rfoot_result[1][0]]

    new_lfoot_data[0][lfoot_result[0][1]:] = new_lfoot_data[0][lfoot_result[0][1]]
    new_rfoot_data[0][rfoot_result[1][1]:] = new_rfoot_data[0][rfoot_result[1][1]]

    new_rfoot_data[0] = savgol_filter(new_rfoot_data[0], 20, 2)
    new_lfoot_data[0] = savgol_filter(new_lfoot_data[0], 20, 2)

    # denoise z-axis
    new_rfoot_data[2][lfoot_result[0][0]:lfoot_result[0][1]] = new_rfoot_data[2][lfoot_result[0][0]]
    new_lfoot_data[2][rfoot_result[0][0]:rfoot_result[0][1]] = new_lfoot_data[2][rfoot_result[0][0]]
    new_lfoot_data[2][rfoot_result[1][0]:rfoot_result[1][1]] = new_lfoot_data[2][rfoot_result[1][0]]

    new_lfoot_data[2][lfoot_result[0][1]:] = new_rfoot_data[2][rfoot_result[1][1]]
    new_rfoot_data[2][rfoot_result[1][1]:] = new_rfoot_data[2][rfoot_result[1][1]]

    new_rfoot_data[2] = savgol_filter(new_rfoot_data[2], 20, 2)
    new_lfoot_data[2] = savgol_filter(new_lfoot_data[2], 20, 2)

    # # denoise y-axis
    # new_rfoot_data[1][lfoot_result[0][0]:lfoot_result[0][1]] = new_rfoot_data[1][lfoot_result[0][0]]
    # new_lfoot_data[1][rfoot_result[0][0]:rfoot_result[0][1]] = new_lfoot_data[1][rfoot_result[0][0]]
    # new_lfoot_data[1][rfoot_result[1][0]:rfoot_result[1][1]] = new_lfoot_data[1][rfoot_result[1][0]]
    #
    # new_lfoot_data[1][lfoot_result[0][1]:] = new_lfoot_data[1][lfoot_result[0][1]]
    # new_rfoot_data[1][rfoot_result[1][1]:] = new_rfoot_data[1][rfoot_result[1][1]]

    # for debug
    if isDebug:
        plt.clf()
        plt.plot(new_lfoot_data[2])
        plt.plot(new_rfoot_data[2])
        plt.plot(new_neck_data[2])
        plt.show()

    # make output file
    lfoot_output = convert_data_to_text(new_lfoot_data)
    rfoot_output = convert_data_to_text(new_rfoot_data)
    neck_output = convert_data_to_text(new_neck_data)

    GraphMaker.writeData(output_files['lfoot'], lfoot_output)
    GraphMaker.writeData(output_files['rfoot'], rfoot_output)
    GraphMaker.writeData(output_files['neck'], neck_output)

    # make output time file
    time_output = convert_data_to_text(new_lfoot_data, type='time')
    GraphMaker.writeData(output_time_file, time_output)

    if makeRotation:
        # load rotation data
        lfoot_rotation_data = np.array(loadData((rotation_files['lfoot'])))
        rfoot_rotation_data = np.array(loadData((rotation_files['rfoot'])))
        neck_rotation_data = np.array(loadData((rotation_files['neck'])))

        # make output rotation file
        lfoot_rotation_output = convert_data_to_text(lfoot_rotation_data)
        rfoot_rotation_output = convert_data_to_text(rfoot_rotation_data)
        neck_rotation_output = convert_data_to_text(neck_rotation_data)

        GraphMaker.writeData(output_rotation_files['lfoot'], lfoot_rotation_output)
        GraphMaker.writeData(output_rotation_files['rfoot'], rfoot_rotation_output)
        GraphMaker.writeData(output_rotation_files['neck'], neck_rotation_output)


def convert_data_to_text(data, type='transform'):
    data = np.array(data)
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

if __name__ == "__main__":
    isAllData = False
    if isAllData:

        names = ['김미송', '김성우', '박승준', '박주현', '서민영', '서승원', '이준혁', '이철우', '임수빈', '전민하', '황주영', '황준태']
        types = ['stair1_50', 'stair1_75', 'stair1_100', 'stair2_50', 'stair2_75', 'stair2_100']
        nums = list(range(10))
        for name in names:
            for type in types:
                for num in nums:
                    num = str(num)
                    files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Lfootdata.txt',
                             'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Rfootdata.txt',
                             'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistData.txt'}

                    rotation_files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/LfootRotationData.txt',
                                      'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/RfootRotationData.txt',
                                      'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistRotationData.txt'}

                    output_files = {'lfoot': 'foot_dataset/output/user3/' + name + '/' + type +  '/' + num + '/Lfootdata.txt',
                                    'rfoot': 'foot_dataset/output/user3/' + name + '/' + type +  '/' + num + '/Rfootdata.txt',
                                    'neck': 'foot_dataset/output/user3/' + name + '/' + type +  '/' + num + '/WaistData.txt'}

                    output_rotation_files = {'lfoot': 'foot_dataset/output/user3/' + name + '/' + type +  '/' + num + '/LfootRotationData.txt',
                                             'rfoot': 'foot_dataset/output/user3/' + name + '/' + type +  '/' + num + '/RfootRotationData.txt',
                                             'neck': 'foot_dataset/output/user3/' + name + '/' + type +  '/' + num + '/WaistRotationData.txt'}

                    output_time_file = 'foot_dataset/output/user3/' + name + '/' + type + '/TimeData.txt'
                    try:
                        test()
                    except:
                        print(files['lfoot'] + "is not found")
    else:
        type_list = ['stair1_75_', 'stair1_100_', 'stair2_75_', 'stair2_100_']
        method_list = ['ours', 'nagao', 'seo']

        for type in type_list:
            for method in method_list:
                files = {'lfoot': 'experiment/' + type + method + '/Lfootdata.txt',
                         'rfoot': 'experiment/' + type + method + '/Rfootdata.txt',
                         'neck': 'experiment/' + type + method + '/WaistData.txt'}

                rotation_files = {'lfoot': 'experiment/' + type + method + 'LfootRotationData.txt',
                                  'rfoot': 'experiment/' + type + method + '/RfootRotationData.txt',
                                  'neck': 'experiment/' + type + method + '/WaistRotationData.txt'}

                output_files = {'lfoot': 'output/experiment/' + type + method + '/Lfootdata.txt',
                                'rfoot': 'output/experiment/' + type + method + '/Rfootdata.txt',
                                'neck': 'output/experiment/' + type + method + '/WaistData.txt'}

                output_rotation_files = {
                    'lfoot': 'output/experiment/' + type + method + '/LfootRotationData.txt',
                    'rfoot': 'output/experiment/' + type + method + '/RfootRotationData.txt',
                    'neck': 'output/experiment/' + type + method + '/WaistRotationData.txt'}

                output_time_file = 'output/experiment/' + type + method + '/TimeData.txt'
                test(isDebug=False)