#수빈
#발 움직임 데이터 그래프


import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#folder = "blendingData/0722-compare/"
folder = "blendingData/realStair/"
#folder = "blendingData/"
rFootName = "RightFootController_"
lFootName = "LeftFootController_"
realName = "realTrajectory"
blendName = "blendedTrajectory"


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return str(self.x) + ", " + str(self.y) + ", " + str(self.z)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector3(x, y, z)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Vector3(x, y, z)

    def __truediv__(self, other):
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __mul__(self, other):
        return Vector3(self.x * other, self.y * other, self.z * other)

    def GetLength(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


def loadPosData(flieName):
    f = open(flieName, 'r')
    # 첫번째 pos 의 x,z값을 0으로 해줌.
    line = f.readline()
    pX = []
    pY = []
    pZ = []
    firstVec = []
    line = f.readline()
    line = line.replace("(", "").replace(")", "").replace(",", "")
    line = line.split()
    firstVec.append(float(line[0]))
    firstVec.append(float(line[1]))
    firstVec.append(float(line[2]))
    pX.append(0)
    pY.append(0)
    pY.append(0)
    while True:
        line = f.readline()
        if not line: break
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        pX.append(float(line[0]))
        #if(float(line[1]) > -1 and float(line[1]) < 1.5) :
        pY.append(float(line[1]) - firstVec[1])
        pZ.append(float(line[2]))
    f.close()
    data = []
    data.append(pX);
    data.append(pY);
    data.append(pZ);
    return data

def loadData(flieName,firstZero = False):
    f = open(flieName, 'r')
    line = f.readline()
    pX = []
    pY = []
    pZ = []
    #line = f.readline()
    line = line.replace("(", "").replace(")", "").replace(",", "")
    line = line.split()
    firstX = firstY = firstZ = 0
    if firstZero:
        firstX = float(line[0])
        firstY = float(line[1])
        firstZ = float(line[2])
    while True:
        line = f.readline()
        if not line: break
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        pX.append(float(line[0]) - firstX)
        pY.append(float(line[1]) - firstY)
        pZ.append(float(line[2]) - firstZ)
    f.close()
    data = []
    data.append(pX);
    data.append(pY);
    data.append(pZ);
    return data

def makeVectorData(data,firstZero = True):
    vecData = []
    px = []
    py = []
    pz = []
    firstVec = []
    for d in data:
        d = d.replace("(", "").replace(")", "").replace(",", "")
        vec = d.split()
        if firstZero:
            if(len(px) == 0):
                px.append(0)
                py.append(0)
                pz.append(0)
                firstVec.append(float(vec[0]))
                firstVec.append(float(vec[1]))
                firstVec.append(float(vec[2]))
                continue
            if len(vec) == 3:
                px.append(float(vec[0]) - firstVec[0])
                py.append(float(vec[1])-firstVec[1])
                pz.append(float(vec[2]) - firstVec[2])
        else:
            if len(vec) == 3:
                px.append(float(vec[0]))
                py.append(float(vec[1]))
                pz.append(float(vec[2]))
    vecData.append(px)
    vecData.append(py)
    vecData.append(pz)
    return vecData


class RecordedFootData():
    def __init__(self,fileName):
        self.realPosData = []
        self.realRotData = []
        self.blendPosData = []
        self.trackerHeightData = []
        self.LoadFootData(fileName)

    def LoadFootData(self,fileName):
        f = open(fileName, 'r')
        data = f.read()
        # 현실 위치, 회전, 조작된 위치, 트래커 높이
        dataList = data.split("####\n")
        self.realPosData = makeVectorData(dataList[0].split('\n'))
        self.realRotData = makeVectorData(dataList[1].split('\n'))
        self.blendPosData = makeVectorData(dataList[2].split('\n'))
        thd = dataList[3].split()
        for t in thd:
            self.trackerHeightData.append(float(t))
        f.close()


class RecordedData():
    def __init__(self,folderName,format = 1):
        self.Format = format

        if(format == 1):
            self.init_1(folderName)
        elif(format == 2):
            self.init_2(folderName)
        elif(format == 3):
            self.init_3(folderName)


    def init_1(self,folderName):
        self.RFootData = RecordedFootData(folderName + "RightFootController.txt")
        self.LFootData = RecordedFootData(folderName + "LeftFootController.txt")
        self.HeadData = [[]]
        self.LoadHeadData(folderName + "otherData.txt")

        rfoot = savgol_filter(self.RFootData.blendPosData[1], 51, 6)
        lfoot = savgol_filter(self.LFootData.blendPosData[1], 51, 6)
        #self.HeadData[1] = savgol_filter(self.HeadData[1], 51, 3)
        self.RVelData = [0]
        self.LVelData = [0]
        self.HeadVelData = [0]

        for i in range(1, len(rfoot)):
            self.RVelData.append((rfoot[i] - rfoot[i - 1]) / 0.0111111)
            self.LVelData.append((lfoot[i] - lfoot[i - 1]) / 0.0111111)
            self.HeadVelData.append((self.HeadData[1][i] - self.HeadData[1][i - 1]) / 0.011111)

    def init_2(self,folderName):
        self.RFootData = loadData(folderName + "Rfootdata.txt",True)
        self.LFootData = loadData(folderName + "Lfootdata.txt",True)
        self.HeadData = loadData(folderName + "WaistData.txt")
        #self.HeadData = np.array(HeadData)

        self.RFootData[1] = savgol_filter(self.RFootData[1], 51, 6)
        self.LFootData[1] = savgol_filter(self.LFootData[1], 51, 6)
        self.HeadData[1] = savgol_filter(self.HeadData[1], 51, 6)
        self.RVelData = [0]
        self.LVelData = [0]
        self.HeadVelData = [0]

        for i in range(1,len(self.RFootData[1])):
            self.RVelData.append((self.RFootData[1][i] - self.RFootData[1][i-1]) / 0.0111111)
            self.LVelData.append((self.LFootData[1][i] - self.LFootData[1][i - 1]) / 0.011111)
            self.HeadVelData.append((self.HeadData[1][i] - self.HeadData[1][i - 1]) / 0.011111)
        self.HighestPoint = []
        self.ChangePoint = []

    def init_3(self,folderName):
        self.RFootData = loadData(folderName + "Rfootdata.txt",True)
        self.LFootData = loadData(folderName + "Lfootdata.txt",True)
        self.HeadData = loadData(folderName + "HeadData.txt")
        self.ankleData = loadData(folderName + "WaistData.txt",True)
        #self.HeadData = np.array(HeadData)

        self.RFootData[1] = savgol_filter(self.RFootData[1], 51, 6)
        self.LFootData[1] = savgol_filter(self.LFootData[1], 51, 6)
        self.HeadData[1] = savgol_filter(self.HeadData[1], 51, 6)
        self.ankleData[1] = savgol_filter(self.ankleData[1], 51, 6)

        self.RVelData = [0]
        self.LVelData = [0]
        self.HeadVelData = [0]
        self.ankleVelData = [0]

        for i in range(1,len(self.RFootData[1])):
            self.RVelData.append((self.RFootData[1][i] - self.RFootData[1][i-1]) / 0.0111111)
            self.LVelData.append((self.LFootData[1][i] - self.LFootData[1][i - 1]) / 0.011111)
            self.HeadVelData.append((self.HeadData[1][i] - self.HeadData[1][i - 1]) / 0.011111)
            self.ankleVelData.append((self.ankleData[1][i] - self.ankleData[1][i - 1]) / 0.011111)
        self.HighestPoint = []
        self.ChangePoint = []

    def LoadHeadData(self,fileName):
        f = open(fileName, 'r')
        data = f.read()
        dataList = data.split("other\n")

        d = dataList[2].split("####\n")
        self.HeadData = makeVectorData(d[0].split('\n'),False)
        f.close()

    def DrawPosAndVelGraph(self,color = None, label = None, startIndex = None, endIndex = None):
        f,axes = plt.subplots(2,1)
        rfoot = []
        lfoot = []

        if(self.Format == 1):
            rfoot = self.RFootData.blendPosData
            lfoot = self.LFootData.blendPosData

        if(self.Format == 2 or self.Format == 3):
            rfoot = self.RFootData
            lfoot = self.LFootData

        if(startIndex == None):
            startIndex= 1
        if(endIndex == None):
            endIndex = -1

        axes[0].plot(rfoot[1][startIndex:endIndex], color=color,label = "Rfoot")
        axes[0].plot(self.HeadData[1][startIndex:endIndex], color=color, label="head")

        if(self.Format != 3):
            axes[0].plot(lfoot[1][startIndex:endIndex], color=color,label = "Lfoot")

        self.HeadVelData = np.array(self.HeadVelData)
        self.RVelData = np.array(self.RVelData)
        self.LVelData = np.array(self.LVelData)

        if self.Format == 3:
            axes[0].plot(self.ankleData[1][startIndex:endIndex], color=color, label="ankle")

        axes[1].plot(self.RVelData[startIndex:endIndex],color=color,label = "RFoot speed")
        if (self.Format != 3):
            axes[1].plot(self.LVelData[startIndex:endIndex],color=color,label = "LFoot speed")
            axes[1].plot(self.LVelData[startIndex:endIndex] - self.HeadVelData[startIndex:endIndex], color=color,
                         label="Lfoot speed - head speed")
        axes[1].plot(self.RVelData[startIndex:endIndex] - self.HeadVelData[startIndex:endIndex],color=color,label = "RFoot speed- head speed")
        axes[1].plot(self.HeadVelData[startIndex:endIndex],label="head speed")


        if self.Format == 3:
            axes[1].plot(self.ankleVelData[startIndex:endIndex], color=color, label="ankle speed")
            axes[1].plot(self.ankleVelData[startIndex:endIndex] - self.HeadVelData[startIndex:endIndex],color=color, label="ankle speed - head speed")

        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].legend()
        axes[1].legend()
        return
    def DrawGrahp(self,x = "Time",color = None, label = None):
        rfoot = []
        lfoot = []
        #plt.axhline(y=0.02, color='r', linewidth=1)
        if(self.Format == 1):
            rfoot = self.RFootData.blendPosData
            lfoot = self.LFootData.blendPosData
        if(self.Format == 2):
            rfoot = self.RFootData
            lfoot = self.LFootData
            self.HeadData[1] = np.array(self.HeadData[1])-0.3

        if(x == "Time"):
            plt.plot(rfoot[1][1:], color=color)
            plt.plot(lfoot[1][1:], color=color)
            plt.plot(self.HeadData[1][1:], color=color, label=label)
            if(self.Format == 1):
                plt.plot(self.RFootData.realPosData[1][1:], color=color)
                plt.plot(self.LFootData.realPosData[1][1:], color=color)
            plt.grid(True)
            #plt.xticks(np.arange(0, 300, 10))
        if x == "Distance":
            Vector3(rfoot[0][1],rfoot[1][1],rfoot[1][2])

            plt.plot(rfoot[2][1:],rfoot[1][1:], color=color)
            plt.plot(lfoot[2][1:],lfoot[1][1:], color=color)
            plt.plot(self.HeadData[2][1:],self.HeadData[1][1:], color=color, label=label)
            rMaxHeight = max(rfoot[1])
            lMaxHeight = max(lfoot[1])
            rindex = rfoot[1].index(rMaxHeight)
            lindex = lfoot[1].index(lMaxHeight)
            plt.scatter(rfoot[2][rindex],rfoot[1][rindex])
            plt.scatter(lfoot[2][lindex],lfoot[1][lindex])

            plt.grid(True)
            #plt.xticks(np.arange(0, 600, 10))

    def CalcVelocity(self):
        velocity_sum = np.zeros(3)
        velocity_count = np.zeros(3)

        for i in range(1, len(self.LFootData[1])):
            velocity = self.LFootData[1][i] - self.LFootData[1][i - 1]
            if velocity > 0:
                velocity_sum[0] += velocity / 0.01
                velocity_count[0] += 1

        for i in range(1, len(self.RFootData[1])):
            velocity = self.RFootData[1][i] - self.RFootData[1][i - 1]
            if velocity > 0:
                velocity_sum[1] += velocity / 0.01
                velocity_count[1] += 1

        for i in range(1, len(self.HeadData[1])):
            velocity = self.HeadData[1][i] - self.HeadData[1][i - 1]
            if velocity > 0:
                velocity_sum[2] += velocity / 0.01
                velocity_count[2] += 1

        return velocity_sum / velocity_count

    def findHighestPoint(self):
        threshold = 0.001
        cnt = 1
        for data in [self.RFootData, self.LFootData, self.HeadData]:
            idx_list = []
            for i in range(cnt, len(data[1]), cnt):
                if math.fabs(data[1][i] - data[1][i - cnt]) / cnt < threshold:
                    idx_list.append(i)

            real_idx_list = []
            for i in range(1, len(idx_list) - 1):
                if data[1][idx_list[i]] > data[1][idx_list[i - 1]] and data[1][idx_list[i]] > data[1][idx_list[i + 1]] :
                    real_idx_list.append(idx_list[i])

            real_idx_list2 = []
            cnt2 = 10
            threshold2 = 0.01
            for idx in real_idx_list:
                if idx + cnt2 < len(data[1]):
                    if data[1][idx] - data[1][idx - cnt2] > threshold2 and data[1][idx] - data[1][idx + cnt2] > threshold2:
                        real_idx_list2.append(idx)

            real_idx_list_y = []
            for idx in real_idx_list2:
                real_idx_list_y.append(data[1][idx])
            plt.scatter(real_idx_list2, real_idx_list_y)

            self.HighestPoint.append(real_idx_list2)

    def findChangePoint(self):
        threshold = 0.001
        cnt = 1
        for data in [self.RFootData, self.LFootData, self.HeadData]:
            idx_list = []
            for i in range(cnt, len(data[1]), cnt):
                if math.fabs(data[1][i] - data[1][i - cnt]) / cnt < threshold:
                    idx_list.append(i)

            real_idx_list = []
            threshold2 = 0.01
            for i in range(1, len(idx_list) - 1):
                if data[1][idx_list[i]] - data[1][idx_list[i - 1]] < threshold2 and data[1][idx_list[i]] + threshold2 < data[1][idx_list[i + 1]] :
                    real_idx_list.append(idx_list[i])

            real_idx_list2 = []
            cnt2 = 10
            for idx in real_idx_list:
                if idx + cnt2 < len(data[1]):
                    if data[1][idx] - data[1][idx - cnt2] < threshold2 and data[1][idx] + threshold2 < data[1][idx + cnt2]:
                        real_idx_list2.append(idx)

            real_idx_list_y = []
            for idx in real_idx_list2:
                real_idx_list_y.append(data[1][idx])
            plt.scatter(real_idx_list2, real_idx_list_y)

            self.ChangePoint.append(real_idx_list2)

    def CalcVelocity2(self):
        value = 100
        # Rfoot up speed
        RfootUpSpeed = []
        RfootNextPoint = [self.HighestPoint[0] + self.ChangePoint[0]]
        RfootNextPoint = sorted(RfootNextPoint[0])
        RfootPair = []
        for i in range(len(RfootNextPoint)):
            if RfootNextPoint[i] in self.HighestPoint[0]:
                RfootPair.append([RfootNextPoint[i - 1], RfootNextPoint[i]])
        for pair in RfootPair:
            speed = (self.RFootData[1][pair[1]] - self.RFootData[1][pair[0]]) / (pair[1] - \
                    pair[0])
            RfootUpSpeed.append(speed * value)

        # Lfoot up speed
        LfootUpSpeed = []
        LfootNextPoint = [self.HighestPoint[1] + self.ChangePoint[1]]
        LfootNextPoint = sorted(LfootNextPoint[0])
        LfootPair = []
        for i in range(len(LfootNextPoint)):
            if LfootNextPoint[i] in self.HighestPoint[1]:
                LfootPair.append([LfootNextPoint[i - 1], LfootNextPoint[i]])
        for pair in LfootPair:
            speed = (self.LFootData[1][pair[1]] - self.LFootData[1][pair[0]]) / (pair[1] - \
                    pair[0])
            LfootUpSpeed.append(speed * value)

        # Head up speed
        HeadUpSpeed = []
        HeadNextPoint = [self.HighestPoint[0] + self.HighestPoint[1] + self.ChangePoint[2]]
        HeadNextPoint = sorted(HeadNextPoint[0])

        HeadPair = []
        for i in range(len(HeadNextPoint)):
            if HeadNextPoint[i] in self.ChangePoint[2]:
                HeadPair.append([HeadNextPoint[i], HeadNextPoint[i +1]])

        for pair in HeadPair:
            speed = (self.HeadData[1][pair[1]] - self.HeadData[1][pair[0]]) / (pair[1] - \
                    pair[0])
            HeadUpSpeed.append(speed * value)

        FootUpSpeed = LfootUpSpeed + RfootUpSpeed
        FootUpSpeed = sorted(FootUpSpeed)
        FootUpSpeed = FootUpSpeed[2:]

        HeadUpSpeed = sorted(HeadUpSpeed)
        HeadUpSpeed = HeadUpSpeed[:3]

        return sum(HeadUpSpeed), len(HeadUpSpeed), sum(FootUpSpeed), len(FootUpSpeed)

#Real traking data
class TrackingData():
    def __init__(self,fileName):
        self.fileName = fileName
        self.posData = []
        self.velData = [[],[],[]]
        self.speed = []
        self.maxX = 0
        self.maxY = 0
        self.maxYIndex = 0
        self.maxZ = 0
        self.validStartIndex = 0
        self.validEndIndex = 0
        self.validTh = 0.4
        self.fixedDeltaTime = 0.01111
        self.length = 0
        self.validMovement = 0
        self.loadData()


    def loadData(self):
        self.valid = False
        self.posData = loadData(self.fileName,True)
        self.length = len(self.posData[0])
        self.posData[1] = savgol_filter(self.posData[1], 51, 6).tolist()
        self.posData = np.array(self.posData)[:,:].tolist()


        for i in range(len(self.posData[0])):
            if i == 0:
                self.velData[0].append(0)
                self.velData[1].append(0)
                self.velData[2].append(0)
                self.speed.append(0)
            else:
                self.velData[0].append( (self.posData[0][i] - self.posData[0][i-1])/ self.fixedDeltaTime )
                self.velData[1].append( (self.posData[1][i] - self.posData[1][i-1])/ self.fixedDeltaTime )
                self.velData[2].append( (self.posData[2][i] - self.posData[2][i-1])/ self.fixedDeltaTime )
                self.speed.append(self.GetVelVector(i).GetLength())
                #print(self.GetVelVector(i).GetLength())
        self.maxX = max(self.posData[0])
        self.maxY = max(self.posData[1])
        self.maxYIndex = self.posData[1].index(self.maxY)
        self.maxZ = max(self.posData[2])
        self.validEndIndex = len(self.posData[0])-1
        if(self.valid):
            for i in range(len(self.speed)):
                if self.speed[i] > self.validTh and self.validStartIndex == 0 and self.velData[1][i] > 0.1:
                    self.validStartIndex = i
                    break
            for i in range(self.posData[1].index(self.maxY),len(self.speed)):
                if self.speed[i] < self.validTh and i > self.validStartIndex:
                    self.validEndIndex = i
                    break
        for i in range(self.validStartIndex,self.validEndIndex):
            self.validMovement += (self.GetVelVector(i) * self.fixedDeltaTime).GetLength()
   #self.speed.index(0.005,self.posData[1].index(maxY))

    def GetAscentVelocity(self):
        vel = self.velData[1][self.validStartIndex:self.maxYIndex]
        if len(vel) == 0: return 0
        return sum(vel)/len(vel)

    def GetAscentHeadVelocity(self, startIndex,endIndex):
        maxY = max(self.posData[1][startIndex:endIndex])
        i = self.posData[1].index(maxY)
        if(i-startIndex <= 0) : return 0;
        return (maxY- self.posData[1][startIndex]) / ((i - startIndex) * self.fixedDeltaTime)

    def GetGetAscentVelocity2(self,startIndex):
        if (self.maxYIndex-startIndex) == 0 :
            return 0
        else:
            return (self.posData[1][self.maxYIndex]-self.posData[1][startIndex])/((self.maxYIndex-startIndex) *self.fixedDeltaTime)

    def GetPosVector(self,i):
        if len(self.posData[0]) > i :
            return Vector3(self.posData[0][i],self.posData[1][i],self.posData[2][i])
        else:
            print("pos data out of range")

    def GetVelVector(self,i):
        if len(self.velData[0]) > i :
            return Vector3(self.velData[0][i],self.velData[1][i],self.velData[2][i])
        else:
            print("vel data out of range")

    def GetMaxHeightIndex(self):
        return self.posData[1].index(self.maxY)

    def DrawPosGraph(self,graph,startIndex = 0, endIndex = 0,color = None,label = None):
        if endIndex == 0:
            endIndex = len(self.posData[0])
        graph.plot(self.posData[1][startIndex:endIndex],color=color,label = label)

    def DrawVelGraph(self,graph,startIndex = 0, endIndex = 0,color = None,label = None):
        if endIndex == 0:
            endIndex = len(self.velData[0])
        graph.plot(self.speed[startIndex:endIndex],color = color,label = label)



def ReadAndDrawGraph(pathR,pathL):
    data = []
    #for path in pathArray:
    data.append(loadPosData(pathL).copy())
    data.append(loadPosData(pathR).copy())
    #for i in range(int(len(data)/2)):
    plt.plot(data[0][1],label = "L")
    plt.plot(data[1][1], label = "R")
    plt.legend(loc='upper right')


def ReadAndDrawGraph2(pathArray,axes,labelName = "compare",LColor = "indigo", RColor = "gold"):
    data = []
    for path in pathArray:
        data.append(RecordedFootData(path))
    axes[0].plot(data[0].blendPosData[1],LColor,label= labelName + "(L)");
    axes[0].plot(data[1].blendPosData[1], RColor,label= labelName + "(R)");
    axes[1].plot(data[0].realPosData[1],LColor);
    axes[1].plot(data[1].realPosData[1], RColor);

    axes[2].plot(data[2].blendPosData[1],LColor);
    axes[2].plot(data[3].blendPosData[1], RColor);
    axes[3].plot(data[2].realPosData[1],LColor);
    axes[3].plot(data[3].realPosData[1], RColor);


def ReadAndDrawGraph3(LeftPath,RightPath,axes,startIndex = 0,labelName = "compare",LColor = "indigo", RColor = "gold"):
    data = []
    data.append(RecordedFootData(LeftPath))
    data.append(RecordedFootData(RightPath))
    axes[startIndex * 2].plot(data[0].blendPosData[1],LColor,label= labelName + "(L)");
    axes[startIndex * 2].plot(data[1].blendPosData[1], RColor,label= labelName + "(R)");
    axes[startIndex * 2 + 1].plot(data[0].realPosData[1],LColor);
    axes[startIndex * 2 + 1].plot(data[1].realPosData[1], RColor);


def DrawRealStairGraph(axes,istwo = False,L=0,R = 1):
    stairPath = "blendingData/realStair/"
    if istwo:
        #d1 = loadPosData(folder+"Lfootdata2.txt")
        #d2 = loadPosData(folder+"Rfootdata2.txt")
        #d3 = loadPosData(folder+"Lfootdata1.txt")
        #d4 = loadPosData(folder +"Rfootdata1.txt")
        d1 = loadPosData(folder +"1/"+ "Lfootdata.txt")
        d2 = loadPosData(folder +"1/"+ "Rfootdata.txt")
        d3 = loadPosData(folder +"2/"+ "Lfootdata.txt")
        d4 = loadPosData(folder +"2/"+ "Rfootdata.txt")
    else:
        d1 = loadPosData(folder+"Lfootdata5.txt")
        d2 = loadPosData(folder+"Rfootdata5.txt")
        d3 = loadPosData(folder+"Lfootdata4.txt")
        d4 = loadPosData(folder+"Rfootdata4.txt")

    axes[L].plot(d2[1],'C0', label="real stair(L)");
    axes[L].plot(d1[1], 'C1',label="real stair(R)");
    axes[R].plot(d4[1],'C0');
    axes[R].plot(d3[1], 'C1');

def DrawCompareGraph():
    return


# f, axes = plt.subplots(2, 1)
# ReadAndDrawGraph3("blendingData/towerStair/L_new.txt","blendingData/towerStair/R_new.txt",axes,"new","C0","C1")
# ReadAndDrawGraph3("blendingData/towerStair/L_t4.txt","blendingData/towerStair/R_t4.txt",axes,"pre","C2","C3")
#
# f.legend(loc='upper right')
# plt.show()

#ReadAndDrawGraph([folder + "test/Lfootdata1.txt",folder + "test/Rfootdata1.txt",folder + "test/Lfootdata2.txt",folder + "test/Rfootdata2.txt" ])

# f1 = RecordedFootData(folder+"left_no_one_L.txt")
# f2 = RecordedFootData(folder+"right_no_one_L.txt")
# f5 = RecordedFootData(folder+"left_no_one_R.txt")
# f6 = RecordedFootData(folder+"right_no_one_R.txt")

#f3 = RecordedFootData(folder+"upForce/"+"Left_L.txt") #왼발 궤적 - 왼발 먼저
#f4 = RecordedFootData(folder+"upForce/"+"Right_L.txt") #오른발 궤적 - 왼발먼저
#f7 = RecordedFootData(folder+"upForce/"+"Left_R.txt") # 왼발 궤적- 오른발먼저
#f8 = RecordedFootData(folder+"upForce/"+"Right_R.txt") # 오른발 궤적 - 오른발 먼저

#f, axes = plt.subplots(4, 1)

#ReadAndDrawGraph3("blendingData/terrain/L1.txt","blendingData/terrain/R1.txt",axes,0,"3","C0","C1")
#ReadAndDrawGraph3("blendingData/terrain/L2.txt","blendingData/terrain/R2.txt",axes)
#ReadAndDrawGraph3("blendingData/terrain/L15.txt","blendingData/terrain/R15.txt",axes,0,"new","C0","C1")
#ReadAndDrawGraph3("blendingData/terrain/L13.txt","blendingData/terrain/R13.txt",axes,0,"new","C2","C3")

#for i in range(0,10):


sIndex = 0
fIndex = 20
#ShowReal_short(s,f,'timeCompare/plane',"C1","C2")
#f, axes = plt.subplots(2, 1)
g = plt

def DrawTrackingDataSet_forVirtual(folderName,color1="C0",color2 = "C1",label = None):
    timeCompareFolder = 'foot_dataset/virtualCompare/'
    avgTime = 0
    avgMaxFootHeight = 0
    avgMaxFootVerticalVelocity = 0
    avgFootVerticalVelocity = 0
    avgMaxFootSpeed = 0
    avgMaxHeadVerticalVelocity = 0
    avgHeadVerticalVelocity = 0
    avgValidMovement = 0
    avgFootSpeed = 0
    avgVerticalMovement = 0
    avgVerticalSpeed = 0

    for i in range(sIndex,fIndex):
        folder_real = timeCompareFolder + folderName + "/"+ str(i)
        RFootData = TrackingData(folder_real + "/Rfootdata.txt")
        LFootData = TrackingData(folder_real + "/Lfootdata.txt")
        WaistData = TrackingData(folder_real + "/HeadData.txt")
        RFootData.DrawPosGraph(g,RFootData.validStartIndex,RFootData.validEndIndex,color=color1,label=label)
        WaistData.DrawPosGraph(g,RFootData.validStartIndex,RFootData.validEndIndex,color=color1,label =label)
        #HeadData.DrawPosGraph(g,RFootData.validStartIndex,RFootData.validEndIndex,color=color1,label =label)

        avgTime += (RFootData.validEndIndex - RFootData.validStartIndex) * 0.01
        avgMaxFootHeight += max(RFootData.posData[1])
        avgFootVerticalVelocity += RFootData.GetAscentVelocity()
        avgMaxFootVerticalVelocity += max(RFootData.velData[1])
        avgMaxFootSpeed += max(RFootData.speed)
        avgHeadVerticalVelocity += WaistData.GetGetAscentVelocity2(RFootData.validStartIndex)
        avgMaxHeadVerticalVelocity += max(WaistData.velData[1])
        avgValidMovement += RFootData.validMovement
        avgFootSpeed += RFootData.validMovement /  ((RFootData.validEndIndex - RFootData.validStartIndex) * 0.01)
        avgVerticalMovement += (RFootData.posData[1][RFootData.validEndIndex] - RFootData.posData[1][RFootData.validStartIndex])
        avgVerticalSpeed += (RFootData.posData[1][RFootData.validEndIndex] - RFootData.posData[1][RFootData.validStartIndex]) / ((RFootData.validEndIndex - RFootData.validStartIndex) * 0.001)

    avgTime /= fIndex
    avgMaxFootHeight /= fIndex
    avgFootVerticalVelocity /= fIndex
    avgMaxFootVerticalVelocity /= fIndex
    avgMaxFootSpeed /= fIndex
    avgHeadVerticalVelocity /= fIndex
    avgMaxHeadVerticalVelocity /= fIndex
    avgFootSpeed  /= fIndex
    avgValidMovement /= fIndex
    avgVerticalMovement /= fIndex
    avgVerticalSpeed /= fIndex

    print("{0}: \n"
          " 체공 시간 : {1}\n"
          " 발의 최대 높이 : {2} \n"
          " 발의 상승 속력 : {3}\n"
          " 발의 최대 수직 속력 : {4} \n"
          " 발의 최대 속력 : {5} \n"
          " 머리 상승 속력 : {6} \n"
          " 머리 최대 수직 속력 : {7} \n"
          " 이동 거리 : {8} \n"
          " 이동 속력 : {9} \n"
          " 최종 이동 수직 거리 : {10}\n"
          " 최종 이동 수직 속력 : {11}".format(folderName,avgTime,avgMaxFootHeight,avgFootVerticalVelocity,avgMaxFootVerticalVelocity,avgMaxFootSpeed,
                                   avgHeadVerticalVelocity,avgMaxHeadVerticalVelocity,avgValidMovement,avgFootSpeed,avgVerticalMovement,avgVerticalSpeed))
    #print(folderName ,":\n" ,avgTime,max(RFootData.velData[1]),max(RFootData.speed),max(WaistData.velData[1]),RFootData.validMovement)

       # RFootData.DrawVelGraph(axes[1],color=color1)
def DrawTrackingDataSet(folderName,sIndex,fIndex, color1="C0", color2="C1", label=None):
        #timeCompareFolder = 'foot_dataset/timeCompare/'
        avgTime = 0
        avgMaxFootHeight = 0
        avgMaxFootVerticalVelocity = 0
        avgFootVerticalVelocity = 0
        avgMaxFootSpeed = 0
        avgMaxHeadVerticalVelocity = 0
        avgHeadVerticalVelocity = 0
        avgValidMovement = 0
        avgFootSpeed = 0
        avgVerticalMovement = 0
        avgVerticalSpeed = 0

        for i in range(0, fIndex):
            folder_real = folderName + "/" + str(i)
            RFootData = TrackingData(folder_real + "/Rfootdata.txt")
            LFootData = TrackingData(folder_real + "/Lfootdata.txt")
            WaistData = TrackingData(folder_real + "/WaistData.txt")
            HeadData = TrackingData(folder_real + "/HeadData.txt")
            #RFootData.DrawPosGraph(g, RFootData.validStartIndex, RFootData.validEndIndex, color=color1, label=label)
            #RFootData.DrawPosGraph(g, RFootData.validStartIndex, RFootData.maxYIndex, color="C4", label=label)

            #WaistData.DrawPosGraph(g, RFootData.validStartIndex , WaistData.maxYIndex,color="C5")
            if i == 1:
                RFootData.DrawPosGraph(g, color=color1, label="RFoot")
                LFootData.DrawPosGraph(g, color=color2, label="LFoot")

            RFootData.DrawPosGraph(g, color=color1)
            LFootData.DrawPosGraph(g, color=color2)
            #LFootData.DrawPosGraph(g, RFootData.validStartIndex, LFootData.length-1, color=color2, label=label)
            WaistData.DrawPosGraph(g,color=color1)
            #WaistData.DrawPosGraph(g, RFootData.validStartIndex, WaistData.maxYIndex, color=color1, label=label)

            #HeadData.DrawPosGraph(g, RFootData.validStartIndex, RFootData.validEndIndex, color=color1, label=label)
            deltaTime = 0.011111
            print((RFootData.validEndIndex, RFootData.validStartIndex))
            avgTime += (RFootData.validEndIndex - RFootData.validStartIndex) * deltaTime
            avgMaxFootHeight += max(RFootData.posData[1])
            avgFootVerticalVelocity += RFootData.GetGetAscentVelocity2(RFootData.validStartIndex)
            avgMaxFootVerticalVelocity += max(RFootData.velData[1])
            avgMaxFootSpeed += max(RFootData.speed)
            avgHeadVerticalVelocity += WaistData.GetAscentHeadVelocity(RFootData.validStartIndex, RFootData.maxYIndex)#WaistData.GetGetAscentVelocity2(RFootData.validStartIndex)
            avgMaxHeadVerticalVelocity += max(WaistData.velData[1])
            avgValidMovement += RFootData.validMovement
            avgFootSpeed += RFootData.validMovement / ((RFootData.validEndIndex - RFootData.validStartIndex) * deltaTime)
            avgVerticalMovement += (RFootData.posData[1][RFootData.validEndIndex] - RFootData.posData[1][RFootData.validStartIndex])
            avgVerticalSpeed += (RFootData.posData[1][RFootData.validEndIndex] - RFootData.posData[1][
                RFootData.validStartIndex]) / ((RFootData.validEndIndex - RFootData.validStartIndex) * deltaTime)

        avgTime /= fIndex
        avgMaxFootHeight /= fIndex
        avgFootVerticalVelocity /= fIndex
        avgMaxFootVerticalVelocity /= fIndex
        avgMaxFootSpeed /= fIndex
        avgHeadVerticalVelocity /= fIndex
        avgMaxHeadVerticalVelocity /= fIndex
        avgFootSpeed /= fIndex
        avgValidMovement /= fIndex
        avgVerticalMovement /= fIndex
        avgVerticalSpeed /= fIndex

        print("{0}: \n"
              " 체공 시간 : {1}\n"
              " 발의 최대 높이 : {2} \n"
              " 발의 상승 속력 : {3}\n"
              " 발의 최대 수직 속력 : {4} \n"
              " 발의 최대 속력 : {5} \n"
              " 머리 상승 속력 : {6} \n"
              " 머리 최대 수직 속력 : {7} \n"
              " 이동 거리 : {8} \n"
              " 이동 속력 : {9} \n"
              " 최종 이동 수직 거리 : {10}\n"
              " 최종 이동 수직 속력 : {11}".format(folderName, avgTime, avgMaxFootHeight, avgFootVerticalVelocity,
                                           avgMaxFootVerticalVelocity, avgMaxFootSpeed,
                                           avgHeadVerticalVelocity, avgMaxHeadVerticalVelocity, avgValidMovement,
                                           avgFootSpeed, avgVerticalMovement, avgVerticalSpeed))

        #WaistData.DrawVelGraph(axes[1],color=color2)
def DrawTrackingDataSet2(folderName):
    timeCompareFolder = 'foot_dataset/timeCompare/'
    ascendingFoot = []
    ascendingHead = []
    for i in range(sIndex,fIndex):
        folder_real = timeCompareFolder + folderName + "/"+ str(i)
        RFootData = TrackingData(folder_real + "/RFootdata.txt")
        LFootData = TrackingData(folder_real + "/LFootdata.txt")
        WaistData =  TrackingData(folder_real + "/WaistData.txt")
        #RFootData.DrawPosGraph(plt,RFootData.validStartIndex,RFootData.validEndIndex,color=color1,label=label)
        #WaistData.DrawPosGraph(plt,RFootData.validStartIndex,RFootData.validEndIndex,color=color1,label =label)
        ascendingFoot.append(max(RFootData.speed))
        ascendingHead.append(WaistData.GetGetAscentVelocity2(RFootData.validStartIndex))
        #print(RFootData.GetAscentVelocity(), WaistData.GetGetAscentVelocity2(RFootData.validStartIndex))
    plt.scatter(ascendingFoot,ascendingHead)
        #print(RFootData.GetAscentVelocity(),WaistData.GetGetAscentVelocity2(RFootData.validStartIndex))



# for i in range(0,20):
#     folder_real =  'foot_dataset/timeCompare/4stair_one/' + str(i)
#     folder_real2 = 'foot_dataset/timeCompare/3stair_one/' + str(i)
#     # folder_real3 = 'foot_dataset/timeCompare/2stair_one/' + str(i)
#     RFootData = TrackingData(folder_real + "/Rfootdata.txt")
#     WaistData = TrackingData(folder_real + "/Waistdata.txt")
#     RFootData2 = TrackingData(folder_real2 + "/Rfootdata.txt")
#     WaistData2 = TrackingData(folder_real2 + "/Waistdata.txt")
#     # RFootData3 = TrackingData(folder_real3 + "/Rfootdata.txt")
#     # WaistData3 = TrackingData(folder_real3 + "/Waistdata.txt")
#     RFootData.DrawPosGraph(plt,RFootData.validStartIndex,color="C1")
#     WaistData.DrawPosGraph(plt,RFootData.validStartIndex,color="C1")
#     RFootData2.DrawPosGraph(plt,RFootData2.validStartIndex,color="C3")
#     WaistData2.DrawPosGraph(plt,RFootData2.validStartIndex,color="C3")
#     # RFootData3.DrawPosGraph(plt,RFootData3.validStartIndex,color="C4")
#     # WaistData3.DrawPosGraph(plt,RFootData3.validStartIndex,color="C4")
#     #print(RFootData.GetAscentVelocity(),RFootData2.GetAscentVelocity(),WaistData.GetAscentVelocity(),WaistData2.GetAscentVelocity())
#     plt.grid(True)
#     plt.show()

#DrawTrackingDataSet("plane","C8","C9")
# DrawTrackingDataSet("1stair","C0","C1")
# DrawTrackingDataSet("2stair_one","C4","C5")
# DrawTrackingDataSet("2stair","C2","C3")
# plt.grid(True)
# plt.show()
#DrawTrackingDataSet("4stair","C6","C7")
#DrawTrackingDataSet("4stair_one","C6","C7")
#DrawTrackingDataSet("3stair_one","C8","C9")

# DrawTrackingDataSet_forVirtual("4stair_one","C0","C1")
# DrawTrackingDataSet_forVirtual("4stair_pre","C4","C5")
# DrawTrackingDataSet("4stair_one","C8","C9")
#DrawTrackingDataSet_forVirtual("2stair","C0","C1")
#DrawTrackingDataSet_forVirtual("2stair_pre","C4","C5")
#DrawTrackingDataSet("slope_down_2_60","C8","C9")
#DrawTrackingDataSet("slope_down_2_70","C2","C3")
# DrawTrackingDataSet_forVirtual("3stair_one","C0","C1")
# DrawTrackingDataSet_forVirtual("3stair_pre","C4","C5")
# DrawTrackingDataSet("3stair_one","C8","C9")

#DrawTrackingDataSet("4stair_one","C6","C7")
#DrawTrackingDataSet2("1stair")
#DrawTrackingDataSet2("2stair")
#DrawTrackingDataSet2("3stair_one")
#DrawTrackingDataSet2("2stair_one")
#DrawTrackingDataSet2("4stair_one")

    #LFootData.DrawVelGraph(axes[1], color="C1")



#ShowReal_short(s,f, 'timeCompare/1stair',"C3","C4")
#ShowReal_short(s,f, 'timeCompare/2stair', "C5", "C6")
#ShowReal_short(s,f, 'timeCompare/2stair_one', "C7", "C8")
#ShowReal_short(s,f, 'timeCompare/3stair_one', "C9", "C10")
#ShowReal_short(s,f, 'timeCompare/4stair_one', "C1", "C2")
#plt.show()

# for i in range(0,4):
#     #ReadAndDrawGraph("foot_dataset/real_short/up/"+str(i)+"/Lfootdata.txt","foot_dataset/real_short/up/"+str(i)+"/Rfootdata.txt" )
#     ReadAndDrawGraph("foot_dataset/virtual_stair/" + str(i) + "/Lfootdata.txt",
#                      "foot_dataset/virtual_stair/" + str(i) + "/Rfootdata.txt")
#     ReadAndDrawGraph("foot_dataset/virtual_stair/up2/" + str(i) + "/Lfootdata.txt",
#                      "foot_dataset/virtual_stair/up2/" + str(i) + "/Rfootdata.txt")
#     #ReadAndDrawGraph("foot_dataset/virtual_stair/"+str(i)+"/HeadData.txt","foot_dataset/virtual_stair/up2/"+str(i)+"/WaistData.txt" )
#     plt.grid(True)
#     plt.xticks(np.arange(0, 500, 10))
#     plt.show()
#f.legend(loc='upper right')