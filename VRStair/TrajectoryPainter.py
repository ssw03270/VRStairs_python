import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from TrajectorySplitter import MakeVelData
from GraphMaker import loadData

posAndVel = True

if posAndVel:
    fixedDeltaTime = 0.011111

    f, axes = plt.subplots(2, 1)

    name = '서승원'
    num = '3'
    type = 'stair1_75'

    files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Lfootdata.txt',
             'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Rfootdata.txt',
             'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistData.txt'}

    lfoot_data = savgol_filter(np.array(loadData(files['lfoot']))[1], 51, 6)
    rfoot_data = savgol_filter(np.array(loadData(files['rfoot']))[1], 51, 6)
    neck_data = savgol_filter(np.array(loadData(files['neck']))[1], 51, 6)

    lfoot_data = np.concatenate((np.zeros(25), lfoot_data))
    rfoot_data = np.concatenate((np.zeros(25), rfoot_data))
    neck_data = np.concatenate((np.zeros(25), neck_data))

    lfoot_data[:26] = lfoot_data[26]
    rfoot_data[:26] = rfoot_data[26]
    neck_data[:26] = neck_data[26]

    end = 280
    lfoot_data = lfoot_data[:end]
    rfoot_data = rfoot_data[:end]
    neck_data = neck_data[:end]

    dataList = [lfoot_data, rfoot_data, neck_data]
    timeList = np.array(list(range(0, len(lfoot_data)))) * fixedDeltaTime

    for posData in dataList:
        velData = np.array(MakeVelData(posData))

        axes[0].plot(timeList, posData)
        axes[1].plot(timeList, velData)

        if (posData == neck_data).all():
            axes[1].plot(timeList, np.array(MakeVelData(lfoot_data)) - velData, '--')
            axes[1].plot(timeList, np.array(MakeVelData(rfoot_data)) - velData, '--')

    axes[0].grid() # 기본 격자 설정
    axes[1].grid() # 기본 격자 설정
    plt.show()

else:
    fixedDeltaTime = 0.011111

    f, axes = plt.subplots(2, 1)

    name = '서승원'
    num = '3'
    type = 'stair1_75'

    files_real = {'lfoot': 'for_figure/stair1_75_real/Lfootdata.txt',
             'rfoot': 'for_figure/stair1_75_real/Rfootdata.txt',
             'neck': 'for_figure/stair1_75_real/WaistData.txt'}

    files_real = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Lfootdata.txt',
             'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Rfootdata.txt',
             'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistData.txt'}

    files_virtual = {'lfoot': 'for_figure/virtual3/1/Lfootdata.txt',
             'rfoot': 'for_figure/virtual3/1/Rfootdata.txt',
             'neck': 'for_figure/virtual3/1/WaistData.txt'}

    t = 0
    tt = 0
    lfoot_data_real = savgol_filter(np.array(loadData(files_real['lfoot']))[1], 51, 6)[t:]
    rfoot_data_real = savgol_filter(np.array(loadData(files_real['rfoot']))[1], 51, 6)[t:]
    neck_data_real = savgol_filter(np.array(loadData(files_real['neck']))[1], 51, 6)[t:]

    lfoot_data_virtual = savgol_filter(np.array(loadData(files_virtual['lfoot']))[1], 51, 6)[tt:]
    rfoot_data_virtual = savgol_filter(np.array(loadData(files_virtual['rfoot']))[1], 51, 6)[tt:]
    neck_data_virtual = savgol_filter(np.array(loadData(files_virtual['neck']))[1], 51, 6)[tt:]

    # neck_data_virtual[177:] += neck_data_virtual[0] - neck_data_virtual[177]
    # neck_data_virtual[:177] = neck_data_virtual[0]
    # neck_data_virtual[300:] = neck_data_virtual[300]

    rfoot_data_virtual[20:120] = rfoot_data_virtual[0:100]
    rfoot_data_virtual[:20] = rfoot_data_virtual[20]

    lfoot_data_virtual -= lfoot_data_virtual[0] - lfoot_data_real[0]
    rfoot_data_virtual -= rfoot_data_virtual[0] - rfoot_data_real[0]
    neck_data_virtual -= neck_data_virtual[0] - neck_data_real[0]

    lfoot_data_real = np.concatenate((np.zeros(25), lfoot_data_real))
    rfoot_data_real = np.concatenate((np.zeros(25), rfoot_data_real))
    neck_data_real = np.concatenate((np.zeros(25), neck_data_real))

    lfoot_data_real[:26] = lfoot_data_real[26]
    rfoot_data_real[:26] = rfoot_data_real[26]
    neck_data_real[:26] = neck_data_real[26]

    end = 280
    lfoot_data_real = lfoot_data_real[:end]
    rfoot_data_real = rfoot_data_real[:end]
    neck_data_real = neck_data_real[:end]
    lfoot_data_virtual = lfoot_data_virtual[:end]
    rfoot_data_virtual = rfoot_data_virtual[:end]
    neck_data_real = neck_data_real[:end]

    data_list_real = [lfoot_data_real, rfoot_data_real, neck_data_real]
    data_list_virtual = [lfoot_data_virtual, rfoot_data_virtual, neck_data_virtual]

    time_list_real = np.array(list(range(0, len(lfoot_data_real)))) * fixedDeltaTime
    time_list_virtual = np.array(list(range(0, len(lfoot_data_virtual)))) * fixedDeltaTime

    for pos_real, pos_virtual in zip(data_list_real, data_list_virtual):
        vel_real = np.array(MakeVelData(pos_real))
        vel_virtual = np.array(MakeVelData(pos_virtual))

        axes[0].plot(time_list_real, pos_real, '--')
        axes[0].plot(time_list_virtual, pos_virtual[:len(lfoot_data_virtual)])

        axes[1].plot(time_list_real, vel_real, '--')
        axes[1].plot(time_list_virtual, vel_virtual[:len(lfoot_data_virtual)])

    axes[0].grid()  # 기본 격자 설정
    axes[1].grid()  # 기본 격자 설정
    plt.show()