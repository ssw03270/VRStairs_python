import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

import GraphMaker
import GraphMaker as g
import TrajectorySplitter as spl
from GraphMaker import loadData

def test(df, stair_type, walk_speed, num="", isDebug=False):
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
                new_data[i] = data[i]

    # change list to nparray
    new_lfoot_data = np.array(new_lfoot_data)
    new_rfoot_data = np.array(new_rfoot_data)
    new_neck_data = np.array(new_neck_data)

    # create net data
    lfoot_net_data = new_lfoot_data[1] - new_neck_data[1]
    rfoot_net_data = new_rfoot_data[1] - new_neck_data[1]
    lfoot_net_data = lfoot_net_data - lfoot_net_data[0] + new_lfoot_data[1][0]
    rfoot_net_data = rfoot_net_data - rfoot_net_data[0] + new_rfoot_data[1][0]

    lfoot_result = spl.FindPoints(new_lfoot_data[1], NotRemove=True,isDebug=True)
    rfoot_result = spl.FindPoints(new_rfoot_data[1], NotRemove=True,isDebug=True)
    rfoot_result[1] = rfoot_result[2]

    # get data for analyze
    netData = [lfoot_net_data, rfoot_net_data]
    netDataIndex = 0
    rangeList = lfoot_result + rfoot_result[1:-1]
    rangeList = sorted(rangeList)
    for start, end in rangeList:
        headZeroIndex = FindHeadZeroPoint(neck_data[1], start, end)
        if isDebug:
            print("현재 발걸음의 시작점과 끝점: ", start, end)
            print("머리 속도가 0이 되는 순간: ", headZeroIndex)

        headMaxSpeedIndex = FindHeadMaxSpeedPoint(neck_data[1], start, end)
        if isDebug:
            print("머리 속도가 최고가 되는 순간: ", headMaxSpeedIndex)

        headMaxSpeedAfterTime = end - headMaxSpeedIndex
        if isDebug:
            print("머리 속도가 최고가 되는 순간부터 끝점까지 걸리는 시간: ", headMaxSpeedAfterTime)

        currentNetData = netData[netDataIndex]
        netDataIndex += 1
        netStartPointIndex = FindFootNetStartPoint(currentNetData, start, end)
        if isDebug:
            print("순수발이 시작하는 순간: ", netStartPointIndex)

        netMaxSpeedPointIndex = FindFootNetMaxSpeedPoint(currentNetData, start, end)
        if isDebug:
            print("순수발이 최고가 되는 순간: ", netMaxSpeedPointIndex)

        if isDebug:
            print(name, stair_type, walk_speed, num)
            print("-------------------------")
        # for debug
        if isDebug:
            plt.scatter(headZeroIndex, neck_data[1][headZeroIndex])
            plt.scatter(headMaxSpeedIndex, neck_data[1][headMaxSpeedIndex])
            plt.scatter(netStartPointIndex, currentNetData[netStartPointIndex])

        new_df = pd.DataFrame(data=[[start, end, headZeroIndex - start, headMaxSpeedIndex - start, headMaxSpeedAfterTime,
                                     netStartPointIndex - start, netMaxSpeedPointIndex - start, stair_type, int(walk_speed)]],
                              columns=["foot_start_index", "foot_end_index", "head_speed_zero_index", "head_speed_max_index",
                                       "head_speed_max_to_foot_end_time", "net_start_index",
                                       "net_speed_max_index", "stair_type", "walk_speed"])
        df = pd.concat([df, new_df])
        df = df.reset_index(drop=True)

    # for debug
    if isDebug:
        plt.plot(new_lfoot_data[1])
        plt.plot(new_rfoot_data[1])
        plt.plot(new_neck_data[1])
        plt.plot(lfoot_net_data)
        plt.plot(rfoot_net_data)
        plt.show()

    return df

# 정해진 범위에서 머리 속도가 0이 되는 index를 구함
def FindHeadZeroPoint(posData, start, end, Th = 0.05):
    velData = spl.MakeVelData(posData)
    for i in range(start + 10, end):
        if velData[i] < Th:
            return i

# 정해진 범위에서 머리 최고 속도인 index를 구함
def FindHeadMaxSpeedPoint(posData, start, end):
    velData = spl.MakeVelData(posData)
    maxSpeed = 0
    maxIndex = 0
    for i in range(start, end):
        if velData[i] > maxSpeed:
            maxSpeed = velData[i]
            maxIndex = i
    return maxIndex

def FindFootNetStartPoint(posData, start, end, Th=0.4):
    velData = spl.MakeVelData(posData)
    for i in range(start, end):
        if velData[i] > Th:
            return i

def FindFootNetMaxSpeedPoint(posData, start, end):
    velData = spl.MakeVelData(posData)
    maxSpeed = 0
    maxIndex = 0
    for i in range(start, end):
        if velData[i] > maxSpeed:
            maxSpeed = velData[i]
            maxIndex = i
    return maxIndex

def analyze(df, isDebug=True):
    df = df.copy()

    groups = df.groupby(['stair_type', 'walk_speed'])
    # method = ['mean', 'std', 'min', 'max', 'median']
    method = ['mean']
    if isDebug:
        print(groups.aggregate(method))
    return groups.aggregate(method)

def exceptList(file_path=""):
    lis = []
    with open(file_path, encoding='UTF8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n', '')
            lis.append(line.split(' '))
    return lis

if __name__ == "__main__":
    isAllData = True
    df = pd.DataFrame(columns=["foot_start_index", "foot_end_index", "head_speed_zero_index", "head_speed_max_index",
                               "head_speed_max_to_foot_end_time", "net_start_index",
                               "net_speed_max_index", "stair_type", "walk_speed"])
    except_list = exceptList("foot_dataset/user3/except.txt")
    except_list_index = 0

    if isAllData:
        names = ['김미송', '김성우', '박승준', '박주현', '서민영', '서승원', '이준혁', '이철우', '임수빈', '전민하', '황주영', '황준태']
        types = ['stair1_50', 'stair1_75', 'stair1_100', 'stair2_50', 'stair2_75', 'stair2_100']
        nums = list(range(10))
        for name in names:
            for type in types:
                for num in nums:
                    num = str(num)

                    if except_list[except_list_index][0] == name and except_list[except_list_index][1] == type.split('_')[0] and \
                            except_list[except_list_index][2] == type.split('_')[1] and num == except_list[except_list_index][3]:
                        except_list_index += 1
                        print(name, type, num, "is excepted data")
                    else:
                        files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Lfootdata.txt',
                                 'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Rfootdata.txt',
                                 'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistData.txt'}

                        rotation_files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/LfootRotationData.txt',
                                          'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/RfootRotationData.txt',
                                          'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistRotationData.txt'}
                        try:
                            t = type.split('_')
                            df = test(df, t[0], t[1], num, isDebug=True)
                        except:
                            print(files['lfoot'] + "is not found")
    else:
        dataList = [['서승원', 'stair1_75', '4'], ['서승원', 'stair2_75', '2'], ['서승원', 'stair2_100', '7'],
                    ['서승원', 'stair1_100', '3']]
        # dataList = [['서승원', 'stair2_75', '2']]
        for name, type, num in dataList:
            files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Lfootdata.txt',
                     'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/Rfootdata.txt',
                     'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistData.txt'}

            rotation_files = {'lfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/LfootRotationData.txt',
                              'rfoot': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/RfootRotationData.txt',
                              'neck': 'foot_dataset/user3/' + name + '/' + type + '/' + num + '/WaistRotationData.txt'}
            t = type.split('_')
            df = test(df, t[0], t[1], isDebug=False)

    df = df.replace(0, np.nan)
    df = df.dropna()
    df = df.sort_values(['stair_type', 'walk_speed'], ascending=True)
    df_index = df.copy()
    df_time = df.copy()

    deltaTime = 0.011111
    df_time['foot_start_index'] *= deltaTime
    df_time['foot_end_index'] *= deltaTime
    df_time['head_speed_zero_index'] *= deltaTime
    df_time['head_speed_max_index'] *= deltaTime
    df_time['head_speed_max_to_foot_end_time'] *= deltaTime
    df_time['net_start_index'] *= deltaTime
    df_time['net_speed_max_index'] *= deltaTime

    df_index.to_csv('csv_output/df_index.csv')
    df_time.to_csv('csv_output/df_time.csv')

    df_index_analyze = analyze(df_index)
    df_time_analyze = analyze(df_time)

    df_index_analyze.to_csv('csv_output/df_index_analyze.csv')
    df_time_analyze.to_csv('csv_output/df_time_analyze.csv')