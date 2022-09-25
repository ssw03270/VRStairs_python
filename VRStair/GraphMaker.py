#수빈
#발 움직임 데이터 그래프


import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import csv

fixedDeltaTime = 0.011111
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

class H2F_Data():
    def __init__(self,folderName,onFilter = True):
        self.fileName = folderName
        self.RFootData = loadData(folderName + "Rfootdata.txt",True)
        self.LFootData = loadData(folderName + "Lfootdata.txt",True)
        self.HeadData = loadData(folderName + "WaistData.txt")
        self.RVelData = [[],[],[]]
        self.LVelData = [[],[],[]]
        self.HeadVelData = [[], [], []]
        self.RSpeedData = []
        self.LSpeedData = []

        self.steps : Step = []
        self.stepIndexes = []
        self.validHeads = []
        self.validHeadIndexes = []

        self.beforeFirstStepAvgHeadHeightChange = 0
        self.firstStepHeadHeightChange = 0
        self.afterFirstStepHeadHeight = 0


        if onFilter:
            self.OnFiltering()

        for i in range(1,len(self.RFootData[1])):
            for j in range(0,3):
                self.RVelData[j].append((self.RFootData[j][i] - self.RFootData[j][i - 1]) / fixedDeltaTime)
                self.LVelData[j].append((self.LFootData[j][i] - self.LFootData[j][i - 1]) / fixedDeltaTime)
                self.HeadVelData[j].append((self.HeadData[j][i] - self.HeadData[j][i - 1]) / fixedDeltaTime)
            self.RSpeedData.append(math.sqrt(self.RVelData[0][i-1] **2 +
                                   self.RVelData[1][i-1] **2 +
                                   self.RVelData[2][i-1] **2))
            self.LSpeedData.append(math.sqrt(self.LVelData[0][i-1] ** 2 +
                                   self.LVelData[1][i-1] ** 2 +
                                   self.LVelData[2][i-1] ** 2))

        self.firstHeadHeight = self.HeadData[1][0]

    def WriteHeadHeightChange(self,infoDict):
        firstAvgHead = 0
        for i in range(0,self.steps[0].validStart):
            firstAvgHead += self.HeadData[1][i]

        if self.steps[0].validStart > 0:
            firstAvgHead /= self.steps[0].validStart
        self.beforeFirstStepAvgHeadHeightChange =  firstAvgHead - self.firstHeadHeight
        self.firstStepHeadHeightChange =  self.HeadData[1][self.steps[0].validStart] - self.firstHeadHeight
        self.afterFirstStepHeadHeight = self.HeadData[1][self.steps[0].validEnd] - self.firstHeadHeight

        infoDict["before first step avg head height change"] += self.beforeFirstStepAvgHeadHeightChange
        infoDict["when first step start head height change"] += self.firstStepHeadHeightChange
        infoDict["after first step end head height change"] +=  self.afterFirstStepHeadHeight
        #infoDict["before second step avg head height change"] +=

        return infoDict


    def OnFiltering(self,windowSize = 51, poly = 6):
        self.RFootData[1] = savgol_filter(self.RFootData[1], windowSize, poly)
        self.LFootData[1] = savgol_filter(self.LFootData[1], windowSize, poly)
        self.HeadData[1] = savgol_filter(self.HeadData[1], windowSize, poly)

    def GetFirstHead(self):
        if len(self.validHeads) > 0:
            return self.validHeads[0]
        else:
            print(self.fileName," : error no vaild head")

    def GetLastHead(self):
        if len(self.validHeads) > 0:
            return self.validHeads[len(self.validHeads)-1]
        else:
            print(self.fileName," : error no vaild head")

    def GetFirstStep(self):
        return self.steps[0]

    def GetSecondStep(self):
        if len(self.steps) > 1:
            return self.steps[1]
        else:
            print("this file doesn't have second steps")
            return None;

    def GetLastStep(self):
        if len(self.validHeads) > 0:
            return self.steps[len(self.steps)-1]
        else:
            print(self.fileName, " : error no vaild head")

    def SplitStep(self):
        if(len(self.steps) != 0 ):
            print("already step is splited!")
            return;
        #1 오른발 걸음 분리
        validIndex = self.find_splitPoint(self.RFootData, self.RVelData, self.RSpeedData)
        for i in range(len(validIndex[0])):
            si = validIndex[0][i]
            ei = validIndex[1][i]
            self.stepIndexes.append((si, ei, 1))
        #2 왼발 걸음 분리
        validIndex = self.find_splitPoint(self.LFootData, self.LVelData, self.LSpeedData)
        for i in range(len(validIndex[0])):
            si = validIndex[0][i]
            ei = validIndex[1][i]
            self.stepIndexes.append((si, ei, 0))
        # 순서에 맞게 정렬
        self.stepIndexes = sorted(self.stepIndexes)
        preE = 0
        for s,e,k in self.stepIndexes:
            if preE > s + 30:
                #plt.cla()
                preE = e
                print("error:",self.fileName)
                #self.DrawGrahp()
                #plt.show()
                continue
            if(k == 1): #Right
                self.steps.append(Step(self.RFootData,self.RVelData,s,e))
            else:#Left
                self.steps.append(Step(self.LFootData, self.LVelData, s, e))
            #head
            hindexes= self.find_head_splitPoint(preE,e,self.HeadData,self.HeadVelData)
            if(hindexes[1]-hindexes[0] < 1):
                print(hindexes)
                #self.DrawGrahp()
                print("error head")
                #plt.show()
            h = Step(self.HeadData,self.HeadVelData,s,e,True)#Step(self.HeadData,self.HeadVelData,hindexes[0],hindexes[1],True)
            preE = e
            if h.verticalDistance > 0.06:
                self.validHeads.append(h)
                self.validHeadIndexes.append((s,e))

        # for s in self.steps:
        #     s.DrawStartToMax()
        # for s in self.validHeads:
        #     s.DrawStartToMax()

    def find_head_splitPoint(self,startIndex,endIndex,posData,velData):
        windowSize = 6
        validTH = 0.2
        end = endIndex
        nextFindIsMove = True
        nextIndex = 0
        nextCool = 30 # 머리 떼진 순간 or 발이 닿은 순간을 찾으면 nextCool 프레임 동안은 찾지 않음.
        validStart = []
        validEnd = []
        for i in range(startIndex,endIndex-windowSize):
            curE = i + windowSize
            curSum = 0#sum(velData[1][i:i+windowSize])/windowSize
            for j in range(i,windowSize+i):
                curSum += velData[1][j] ** 2
            curSum = math.sqrt(curSum)
            #print(curSum)
            if nextIndex > 0:
                nextIndex -= 1
                continue
            if(nextFindIsMove): # 머리가 올라가기 시작하는 순간을 찾음.
                if(curSum > validTH and velData[1][i] > 0.09):
                    #plt.scatter(i,posData[1][i])
                    nextFindIsMove = False
                    nextIndex = nextCool
                    validStart.append(i)

            else:  # 머리 움직이지 않는순간
                if (curSum < validTH - 0.1 and velData[1][i] < 0.07 or (velData[1][i + windowSize] > validTH and velData[1][i] < validTH)):
                    #plt.scatter(i, posData[1][i])
                    nextFindIsMove = True
                    nextIndex = nextCool
                    validEnd.append(i)
                    break
        if(not nextFindIsMove): #발이 다시 땅에 닿는 순간을 못찾았다면 맨 마지막 index를 땅에 닿는 인덱스로 넣어줌.
            validEnd.append(end)
            plt.scatter(end, posData[1][end])

        if len(validStart) != len(validEnd):
            print("error - valid index")
            return [[0],[end-1]]

        if len(validStart) == 0:
            return (startIndex,endIndex)

        return (validStart[0],validEnd[0])

    def find_splitPoint(self,posData,velData,speedData):
        windowSize = 17
        validTH = 0.42
        end = len(speedData)
        nextFindIsMove = True
        nextIndex = 0
        nextCool = 30 # 발이 떼진 순간 or 발이 닿은 순간을 찾으면 nextCool 프레임 동안은 찾지 않음.
        validStart = []
        validEnd = []

        for i in range(0,end-windowSize):
            curE = i + windowSize
            curSum = sum(speedData[i:i+windowSize])/windowSize
            curYSum = sum(posData[1][i:i+windowSize])/windowSize
            curVelYSum = sum(velData[1][i:i+windowSize])/windowSize
            if nextIndex > 0:
                nextIndex -= 1
                continue
            if(nextFindIsMove): # 발을 떼기 시작하는 순간을 찾음.
                if(curSum > validTH  and velData[1][i] > 0.07 and posData[1][i] > 0  and (posData[1][i+5] -posData[1][i]) > 0.01):
                   # plt.scatter(i,posData[1][i])
                    nextFindIsMove = False
                    nextIndex = nextCool
                    validStart.append(i)
            else: #발이 다시 땅에 닿는 순간을 찾음.
                if (curSum < validTH - 0.15 or curYSum < 0.2 or ((posData[1][i] - posData[1][i-5]) < 0) and ((posData[1][i]) - posData[1][i+5] < 0 )):
                    #plt.scatter(i, posData[1][i])
                    nextFindIsMove = True
                    nextIndex = nextCool
                    validEnd.append(i)

        if(not nextFindIsMove): #발이 다시 땅에 닿는 순간을 못찾았다면 맨 마지막 index를 땅에 닿는 인덱스로 넣어줌.
            validEnd.append(end)
            #plt.scatter(end, posData[1][end])

        if len(validStart) != len(validEnd):
            print("error - valid index")
            return [[0],[end-1]]

        return [validStart,validEnd]

    def findCrossPoint(self,start,end):
        minDistance = 10000
        minI = 0
        for i in range(start,end):
            cur = abs(self.RFootData[1][i] - self.LFootData[1][i])
            if minDistance == 0:
                return i
            if minDistance > cur:
                minDistance = cur
                minI = i
        else:
            print("not cross")
            return minI


    def DrawGrahp(self,color = None, label = None):
        plt.plot(self.RFootData[1][1:], color=color)
        plt.plot(self.LFootData[1][1:], color=color)
        plt.plot(self.HeadData[1][1:], color=color, label=label)

    def DrawGrahp2(self,start,end,color = None, label = None):
        plt.rc('font', size=10)
        plt.rc('axes', labelsize=16)
        plt.grid(True)
        plt.ylim(0,2)
        plt.xlim(0,end * fixedDeltaTime)
        plt.yticks(np.arange(0, 2,0.5))
        plt.xticks(np.arange(0, end * fixedDeltaTime, 0.5))
        plt.xlabel('time(s)')
        plt.ylabel('height(m)')
        plt.plot(np.array(list(range(start,end)))* fixedDeltaTime,self.HeadData[1][start:end], color=color, label="neak")
        plt.plot(np.array(list(range(start,end)))* fixedDeltaTime,self.LFootData[1][start:end] + 0.05, color=color,label = "left ankle")
        plt.plot(np.array(list(range(start,end)))* fixedDeltaTime,self.RFootData[1][start:end] + 0.05, color=color,label = "right ankle")

    def DrawSectionPosAndVelGraph(self,axes,i,sectionIndex, addedLabel = ""):
        if sectionIndex > len(self.steps) -1:
            sectionIndex = len(self.steps) -1
        curStep = self.steps[sectionIndex]
        startIndex = curStep.validStart
        endIndex = curStep.validEnd + 10

        xAxis = np.array(list(range(0,endIndex - startIndex)))* fixedDeltaTime
        axes[0][i].plot(xAxis,self.HeadData[1][startIndex:endIndex], label="neak" + addedLabel )
        axes[0][i].plot(xAxis,self.LFootData[1][startIndex:endIndex],label = "left ankle"+ addedLabel)
        axes[0][i].plot(xAxis,self.RFootData[1][startIndex:endIndex], label = "right ankle" + addedLabel)

        self.HeadVelData = np.array(self.HeadVelData)
        self.RVelData = np.array(self.RVelData)
        self.LVelData = np.array(self.LVelData)

        axes[1][i].plot(xAxis,self.HeadVelData[1][startIndex:endIndex],label="neak velocity" + addedLabel)
        axes[1][i].plot(xAxis,self.RVelData[1][startIndex:endIndex],label = "left velocity" + addedLabel)
        axes[1][i].plot(xAxis,self.LVelData[1][startIndex:endIndex],label = "right velocity" + addedLabel)
        axes[1][i].plot(xAxis,self.LVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex], label = "Net velocity(L)"+ addedLabel)
        axes[1][i].plot(xAxis,self.RVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex],label = "Net velocity(R)"+ addedLabel)


        axes[0][i].grid(True)
        axes[1][i].grid(True)
        axes[0][i].legend()
        axes[1][i].legend()

    def DrawPosAndVelGraph(self,_axes,color = None, label = None, startIndex = None, endIndex = None):
        axes = _axes
        rfoot = self.RFootData
        lfoot = self.LFootData

        if(startIndex == None):
            startIndex= 0
        if(endIndex == None):
            endIndex = -1

        axes[0].plot(self.HeadData[1][startIndex:endIndex], color=color, label="head")
        axes[0].plot(rfoot[1][startIndex:endIndex], color=color,label = "Rfoot")
        axes[0].plot(lfoot[1][startIndex:endIndex], color=color,label = "Lfoot")


        self.HeadVelData = np.array(self.HeadVelData)
        self.RVelData = np.array(self.RVelData)
        self.LVelData = np.array(self.LVelData)
        for step in self.steps:
            axes[0].scatter(step.validStart, step.posData[1][0])
            axes[0].vlines(step.validStart, 0, 2,colors="black",linestyles="--")

        for step in self.steps:
            axes[0].scatter(step.validEnd, step.posData[1][len(step.posData[1]) - 1])
            axes[0].vlines(step.validEnd,0,2,colors="r",linestyles="--")

        for step in self.steps:
            axes[1].vlines(step.validStart+step.maxYIndex, 0, 2, colors="b", linestyles="--")
            axes[0].vlines(step.validStart + step.maxVelIndex,0, 2,colors="g", linestyles="--")
            axes[1].vlines(step.validStart + step.maxVelIndex, 0, 2, colors="g", linestyles="--")

        for head in self.validHeads:
            axes[0].plot(list(range(head.validStart, head.validEnd)),head.originPos[1][head.validStart:head.validEnd])
            axes[0].vlines(head.validStart+head.maxVelIndex, 0, 2, colors="y", linestyles="--")
            axes[1].vlines(head.validStart + head.maxVelIndex, 0, 2, colors="y", linestyles="--")



        axes[1].plot(self.HeadVelData[1][startIndex:endIndex],label="head velocity")
        axes[1].plot(self.RVelData[1][startIndex:endIndex],color=color,label = "RFoot velocity")
        axes[1].plot(self.LVelData[1][startIndex:endIndex],color=color,label = "LFoot velocity")
        axes[1].plot(self.RVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex],color=color,label = "RFoot velocity- head velocity")
        axes[1].plot(self.LVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex], color=color,label = "LFoot velocity- head velocity")

        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].legend()
        axes[1].legend()
        return



class Step():
    def __init__(self,origin,originVel,validStart,validEnd,isHead = False):
        self.originPos = origin
        self.originVel = originVel
        self.validStart = validStart
        self.validEnd = validEnd
        self.posData = np.array(origin)[:,validStart:validEnd]
        self.velData = np.array(originVel)[:,validStart:validEnd]
        self.length = 0
        self.maxY = 0
        self.maxYIndex = 0
        self.maxVelIndex = 0
        self.maxYVel = 0
        self.ascentVelocity = 0
        self.descentVelocity = 0
        self.verticalDistance = 0
        self.lastY = 0
        self.ascentDistance = 0
        self.isHead = isHead
        self.make_data()

    def make_data(self):
        self.length = len(self.posData[0])
        self.maxY = max(self.posData[1])
        self.maxYVel = max(self.velData[1])
        self.maxYIndex = np.where(self.posData[1] == self.maxY)[0][0]
        self.maxVelIndex = np.where(self.velData[1] == self.maxYVel)[0][0]
        self.ascentDistance = self.maxY - self.posData[1][0]

        if self.maxYIndex != 0:
            self.ascentVelocity = (self.maxY - self.posData[1][0]) / (self.maxYIndex * fixedDeltaTime)
        if self.length - self.maxYIndex != 0:
            self.descentVelocity = (self.maxY-self.posData[1][self.length-1]) / ((self.length - self.maxYIndex) * fixedDeltaTime)
        self.verticalDistance = self.posData[1][self.length - 1] - self.posData[1][0]
        self.lastY = self.posData[1][self.length-1]



    def DrawStartToMax(self):
        plt.plot(list(range(self.validStart,self.validStart + self.maxYIndex)) ,self.originPos[1][self.validStart:self.validStart + self.maxYIndex])

    def Draw(self):
        plt.plot(list(range(self.validStart, self.validEnd)),
                 self.originPos[1][self.validStart:self.validEnd])
        plt.plot(list(range(self.validStart, self.validStart + self.maxYIndex)),
                 self.originPos[1][self.validStart:self.validStart + self.maxYIndex],color="gold")
        plt.scatter(self.validStart-1 + self.maxYIndex,self.maxY)

    def IsOutlier(self,avgDict):
        if not self.isHead:
            if self.length/avgDict["length"] < 0.6 or self.length/avgDict["length"] > 1.4:
                return True
            if self.verticalDistance/avgDict["verticalDistance"] < 0.5 or self.verticalDistance/avgDict["verticalDistance"] > 1.3:
                return True
            if abs(self.lastY - avgDict["lastY"]) > 0.1:
                return True
            if self.maxY / avgDict["maxY"] > 1.2 or  self.maxY / avgDict["maxY"] < 0.8 :
                return True
        if self.isHead:
            if self.length/avgDict["length"] < 0.6 or self.length/avgDict["length"] > 1.5:
                return True
            if self.verticalDistance / avgDict["verticalDistance"] < 0.6 or (self.verticalDistance / avgDict["verticalDistance"])> 1.3:
                return True
        return False

    def WriteInfo(self,dict):
        dict["descentVelocity"]+= self.descentVelocity
        dict["maxVelocity"] += self.maxYVel
        dict["ascentVelocity"] += self.ascentVelocity
        dict["ascentDistance"] += self.ascentDistance
        dict["length"] += self.length
        dict["verticalDistance"] += self.verticalDistance
        dict["lastY"] += self.lastY
        dict["maxY"] += self.maxY

    def WriteSD(self,avgdict,sdDict):
        sdDict["descentVelocity"] += (avgdict["descentVelocity"]-self.descentVelocity)**2
        sdDict["maxVelocity"] += (avgdict["maxVelocity"]-self.maxYVel)**2
        sdDict["ascentVelocity"] += (avgdict["ascentVelocity"]-self.ascentVelocity)**2
        sdDict["ascentDistance"] += (avgdict["ascentDistance"]-self.ascentDistance)**2
        sdDict["length"] += (avgdict["length"]-self.length)**2
        sdDict["verticalDistance"] += (avgdict["verticalDistance"]-self.verticalDistance)**2
        sdDict["lastY"] += (avgdict["lastY"]-self.lastY)**2
        sdDict["maxY"] += (avgdict["maxY"]-self.maxY)**2


class StepAnalyzer():
    def __init__(self,files,isDebug = False,condition ="stair1"):
        self.data : H2F_Data = []
        self.firstSteps = []
        self.secondStpes = []
        self.lastSteps = []
        self.firstHeads = []
        self.lastHeads = []
        self.condition = condition

        self.isDebug = isDebug
        self.avgDicts = []
        self.sdDicts = []

        self.make_steps(files)
        #self.GetHeadHeightChange()

        self.AnalyzeHead()
        print("---------------First Foot------------------")
        self.AnalyzeFirstStep()
        print("---------------Second Foot------------------")
        self.AnalyzeSecondStep()
        print("---------------Last Foot------------------")
        self.AnalyzeLastStep()


    def GetHeadHeightChange(self):
        infoDict = {"before first step avg head height change":0 ,"when first step start head height change" : 0,"after first step end head height change":0}
        plt.cla()
        for d in self.data:
            d.WriteHeadHeightChange(infoDict)
            d.DrawGrahp2(0,d.steps[1].validEnd)
            plt.vlines(d.steps[1].validStart * fixedDeltaTime,0,2,colors="black",linestyles="--")
            #crossPoint = d.findCrossPoint(d.steps[1].validStart,d.steps[1].validEnd)
            #plt.scatter(d.steps[1].validStart, d.RFootData[1][d.steps[1].validStart])
            #plt.scatter(crossPoint * fixedDeltaTime,d.RFootData[1][crossPoint],color = "r",zorder= 3)
            plt.legend(loc="upper right")
            plt.show()
        for v in infoDict.keys():
            infoDict[v] /= len(self.data)

        print(infoDict)

    def GetResultList(self):
        return [self.avgDicts,self.sdDicts]

    def writeCSV(self):
        with open(self.condition +".csv",'w',encoding="UTF-8") as f:
            w = csv.writer(f)
            order = ["Head 1","Head 2", "First Foot", "Second Foot", "Last Foot"]
            i = 0
            for o in order:
                w.writerow([o])
                w.writerow(self.avgDicts[i].keys())
                w.writerow(self.avgDicts[i].values())
                w.writerow(self.sdDicts[i].keys())
                w.writerow(self.sdDicts[i].values())
                i += 1


    def make_steps(self,files):
        for file in files:
            if not os.path.exists(file):
                print(file, ": not exists.")
                continue
            data = H2F_Data(file)
            if self.isDebug:
                data.DrawGrahp()
                #data.DrawPosAndVelGraph()
            data.SplitStep()
            self.data.append(data)
            #data.DrawGrahp()
        if(self.isDebug):
            plt.show()
        #plt.show()
        return

    def AnalyzeHead(self):
        for data in self.data:
            fh = data.GetFirstHead()
            lh = data.GetLastHead()
            if fh:
                self.firstHeads.append(fh)
            if lh:
                self.lastHeads.append(lh)

        print("-------------Head movement1------------")
        if self.isDebug:
            for h in self.firstHeads:
                h.Draw()
            plt.show()
        self.AnalyzeStep(self.firstHeads)

        print("-------------Head movement2------------")
        if self.isDebug:
            for h in self.lastHeads:
                h.Draw()
            plt.show()
        self.AnalyzeStep(self.lastHeads)

    def AnalyzeFirstStep(self):
        for data in self.data:
            s = data.GetFirstStep()
            if s:
                self.firstSteps.append(s)
        if self.isDebug:
            for s in self.firstSteps:
                s.Draw()
            plt.show()
        self.AnalyzeStep(self.firstSteps)


    def AnalyzeSecondStep(self):
        for data in self.data:
            s = data.GetSecondStep()
            if s:
                self.secondStpes.append(s)
        if self.isDebug:
            for s in self.secondStpes:
                s.Draw()
            plt.show()
        self.AnalyzeStep(self.secondStpes)


    def AnalyzeLastStep(self):
        for data in self.data:
            self.lastSteps.append(data.GetLastStep())
        if self.isDebug:
            for s in self.lastSteps:
                s.Draw()
            plt.show()
        self.AnalyzeStep(self.lastSteps)

    def GetAvgInfo(self,steps):
        infoDict = {"descentVelocity" : 0,
                    "maxVelocity" : 0,
                    "ascentVelocity" : 0,
                    "ascentDistance" :0,
                    "length" : 0,
                    "verticalDistance" : 0,
                    "lastY":0,
                    "maxY": 0}
        for s in steps:
            s.WriteInfo(infoDict)
        for v in infoDict.keys():
            infoDict[v] /= len(steps)
        return infoDict

    def GetSDInfo(self,avgInfo,steps):
        infoDict = {"descentVelocity" : 0,
                    "maxVelocity": 0,
                    "ascentVelocity" : 0,
                    "ascentDistance" : 0,
                    "length" : 0,
                    "verticalDistance" : 0,
                    "lastY":0,
                    "maxY": 0}
        for s in steps:
            s.WriteSD(avgInfo,infoDict)
        for v in infoDict.keys():
            infoDict[v] /= len(steps)
            infoDict[v] = math.sqrt(infoDict[v])
        return infoDict


    def AnalyzeStep(self,steps):
        infoDict = self.GetAvgInfo(steps)
        print("Before remove OutLier:" ,infoDict)
        #self.RemoveOutlier(steps,infoDict)

        if(self.isDebug):
            for s in steps:
                s.Draw()
            plt.title("After")
            plt.show()
        else:
            plt.cla()
        infoDict = self.GetAvgInfo(steps)
        print("After remove OutLier:", infoDict)
        infoDict["total count"] = len(steps)
        SDDict = self.GetSDInfo(infoDict,steps)
        print("SD",SDDict)
        self.avgDicts.append(infoDict)
        self.sdDicts.append(SDDict)




    def RemoveOutlier(self,steps,infoDict):
        rList =[]
        for i in range(len(steps)):
            if(steps[i].IsOutlier(infoDict)):
                steps[i].Draw()
                rList.append(i)
        if self.isDebug:
            plt.title("OutLier")
            plt.show()
        plt.cla()
        rList.sort(reverse=True)
        for i in rList:
            del steps[i]






class RecordedData():
    def __init__(self,folderName,format = 1):
        self.Format = format
        self.RFootData = None
        self.LFootData = None

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
        self.RVelData = [[],[],[]]
        self.LVelData = [[],[],[]]
        self.HeadVelData = []

        for i in range(1,len(self.RFootData[1])):
            for j in range(0,3):
                self.RVelData[j].append((self.RFootData[j][i] - self.RFootData[j][i-1]) / 0.0111111)
                self.LVelData[j].append((self.LFootData[j][i] - self.LFootData[j][i - 1]) / 0.011111)
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
        self.RVelData = [[],[],[]]
        self.LVelData = [[],[],[]]
        self.HeadVelData = []
        self.ankleVelData = [[],[],[]]

        for i in range(1,len(self.RFootData[1])):
            for j in range(0,3):
                self.RVelData[j].append((self.RFootData[j][i] - self.RFootData[j][i-1]) / 0.0111111)
                self.LVelData[j].append((self.LFootData[j][i] - self.LFootData[j][i - 1]) / 0.011111)
                self.ankleVelData.append((self.ankleData[1][i] - self.ankleData[1][i - 1]) / 0.011111)
            self.HeadVelData.append((self.HeadData[1][i] - self.HeadData[1][i - 1]) / 0.011111)

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

        axes[0].plot(self.HeadData[1][startIndex:endIndex], color=color, label="head")
        axes[0].plot(rfoot[1][startIndex:endIndex], color=color,label = "Rfoot")
        if(self.Format != 3):
            axes[0].plot(lfoot[1][startIndex:endIndex], color=color,label = "Lfoot")

        self.HeadVelData = np.array(self.HeadVelData)
        self.RVelData = np.array(self.RVelData)
        self.LVelData = np.array(self.LVelData)

        if self.Format == 3:
            axes[0].plot(self.ankleData[1][startIndex:endIndex], color=color, label="ankle")

        axes[1].plot(self.HeadVelData[startIndex:endIndex],label="head speed")
        axes[1].plot(self.RVelData[1][startIndex:endIndex],color=color,label = "RFoot speed")
        if (self.Format != 3):
            axes[1].plot(self.LVelData[1][startIndex:endIndex],color=color,label = "LFoot speed")
            #axes[1].plot(self.LVelData[1][startIndex:endIndex] - self.HeadVelData[startIndex:endIndex], color=color,
        #axes[1].plot(self.RVelData[1][startIndex:endIndex] - self.HeadVelData[startIndex:endIndex],color=color,label = "RFoot speed- head speed")



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
        self.valid = True
        self.posData = loadData(self.fileName,True)
        self.length = len(self.posData[0])
        self.posData[1] = savgol_filter(self.posData[1], 51, 6).tolist()
        self.posData = np.array(self.posData)[:,:140].tolist()


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
                if self.speed[i] < self.validTh-0.14 and i > self.validStartIndex:
                    self.validEndIndex = i
                    break
        for i in range(self.validStartIndex,self.validEndIndex):
            self.validMovement += (self.GetVelVector(i) * self.fixedDeltaTime).GetLength()
   #self.speed.index(0.005,self.posData[1].index(maxY))
    def GetDescentVelocity(self):
        if(self.validEndIndex - self.maxYIndex == 0) : return 0
        return (self.maxY - self.posData[1][self.validEndIndex]) / ((self.validEndIndex - self.maxYIndex) * self.fixedDeltaTime)
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
        avgFootDescentSpeed =0

        for i in range(0, fIndex):
            folder_real = folderName + "/" + str(i)
            RFootData = TrackingData(folder_real + "/Rfootdata.txt")
            LFootData = TrackingData(folder_real + "/Lfootdata.txt")
            WaistData = TrackingData(folder_real + "/WaistData.txt")
            HeadData = TrackingData(folder_real + "/HeadData.txt")
            RFootData.DrawPosGraph(g, RFootData.validStartIndex, color=color1)
            RFootData.DrawPosGraph(g, RFootData.validStartIndex, RFootData.validEndIndex, color="b", label=label)
            RFootData.DrawPosGraph(g, RFootData.validStartIndex, RFootData.maxYIndex, color="r", label=label)

            #WaistData.DrawPosGraph(g, RFootData.validStartIndex , WaistData.maxYIndex,color="C5")
            if i == 1:
                RFootData.DrawPosGraph(g, color=color1, label="RFoot")
                LFootData.DrawPosGraph(g, color=color2, label="LFoot")


            #LFootData.DrawPosGraph(g, color=color2)
            #LFootData.DrawPosGraph(g, RFootData.validStartIndex, LFootData.length-1, color=color2, label=label)
            #WaistData.DrawPosGraph(g,color=color1)
            #WaistData.DrawPosGraph(g, RFootData.validStartIndex, WaistData.maxYIndex, color=color1, label=label)

            #HeadData.DrawPosGraph(g, RFootData.validStartIndex, RFootData.validEndIndex, color=color1, label=label)
            deltaTime = 0.011111
            print((RFootData.validEndIndex, RFootData.validStartIndex))
            avgTime += (RFootData.validEndIndex - RFootData.validStartIndex) * deltaTime
            avgMaxFootHeight += max(RFootData.posData[1])
            avgFootVerticalVelocity += RFootData.GetGetAscentVelocity2(RFootData.validStartIndex)
            avgFootDescentSpeed += RFootData.GetDescentVelocity()
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
        avgFootDescentSpeed /= fIndex
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
              " 최종 이동 수직 속력 : {11}\n"
              " 발 하강 속력 : {12}".format(folderName, avgTime, avgMaxFootHeight, avgFootVerticalVelocity,
                                           avgMaxFootVerticalVelocity, avgMaxFootSpeed,
                                           avgHeadVerticalVelocity, avgMaxHeadVerticalVelocity, avgValidMovement,
                                           avgFootSpeed, avgVerticalMovement, avgVerticalSpeed,avgFootDescentSpeed))

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


