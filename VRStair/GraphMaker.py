#수빈
#발 움직임 데이터 그래프


import numpy as np
import os
import random
from define import *
from utility import *
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import csv
import seaborn as sns
import TrajectorySplitter as spl

defalutDF  =  pd.DataFrame({"index":[],"time":[],"y":[],"velY":[]})


class RecordedFootData():
    def __init__(self,fileName,firstZero=True):
        self.realPosData = []
        self.realVelData = [[0],[0],[0]]
        self.realRotData = []
        self.blendPosData = []
        self.trackerHeightData = []
        self.firstZero = firstZero

        self.LoadFootData(fileName)

    def LoadFootData(self,fileName):
        f = open(fileName, 'r')
        data = f.read()
        # 현실 위치, 회전, 조작된 위치, 트래커 높이
        dataList = data.split("####\n")
        self.realPosData = makeVectorData(dataList[0].split('\n'),firstZero=self.firstZero)
        self.realRotData = makeVectorData(dataList[1].split('\n'),firstZero=self.firstZero)
        self.blendPosData = makeVectorData(dataList[2].split('\n'),firstZero=self.firstZero)
        thd = dataList[3].split()
        self.onFilter = True
        for t in thd:
            self.trackerHeightData.append(float(t))
        for i in range(1,len(self.realPosData[0])):
            for j in range(0,3):
                    self.realVelData[j].append((self.realPosData[j][i] - self.realPosData[j][i-1]) / fixedDeltaTime)
        self.realVelData[1] = savgol_filter(self.realVelData[1], filterSize, 6)
        f.close()

    def to_dataframe(self):
        df = pd.DataFrame({"posX":self.realPosData[0],
                           "posY":self.realPosData[1],
                           "posZ":self.realPosData[2],
                           "RotX":self.realRotData[0],
                           "RotY":self.realRotData[1],
                           "RotZ":self.realRotData[2],
                           "vPosX":self.blendPosData[0],
                           "vPosY":self.blendPosData[1],
                           "vPosZ":self.blendPosData[2],
                           "initH":self.trackerHeightData})
        return df



    def to_txt(self,start=0,end=-1):
        dataTxt = Vector3ArrayToString(self.realPosData,start,end) + "####\n"\
                  + Vector3ArrayToString(self.realRotData,start,end) + "####\n" \
                  + Vector3ArrayToString(self.blendPosData,start,end) + "####\n" + FloatArrayToString(self.trackerHeightData,start,end)
        return dataTxt


class H2F_Data():
    def __init__(self,folderName,onFilter = False):
        self.fileName = folderName
        self.RFootData = loadData(folderName + "Rfootdata.txt",True)
        self.LFootData = loadData(folderName + "Lfootdata.txt",True)
        self.HeadData = loadData(folderName + "WaistData.txt",True)
        self.RRotData = np.array(loadData(folderName + "RfootRotationData.txt",True,defalutRotationR))
        self.LRotData = np.array(loadData(folderName + "LfootRotationData.txt",True,defalutRotationL))
        self.RVelData = [[0],[0],[0]]
        self.LVelData = [[0],[0],[0]]
        self.HeadVelData = [[0], [0], [0]]
        self.RSpeedData = []
        self.LSpeedData = []
        self.RNetPosData = [[],[],[]]
        self.LNetPosData = [[],[],[]]
        self.RNetVelData = [[],[],[]]
        self.LNetVelData = [[], [], []]

        self.steps : Step = []
        self.stepIndexes = []
        self.validHeads : Step = []
        self.validHeadIndexes = []

        self.netSteps : Step = []

        self.beforeFirstStepAvgHeadHeightChange = 0
        self.firstStepHeadHeightChange = 0
        self.afterFirstStepHeadHeight = 0
        self.isOutLier = False
        self.myVelTest = True

        self.headTrajectoryDataFrame = pd.DataFrame()
        self.allTrajectoryDataFrame = pd.DataFrame()
        self.trajectoryLength = 0

        if onFilter:
            self.OnFiltering()

        for i in range(0, len(self.RFootData[1])):
            for j in range(0,3):
                self.RNetPosData[j].append(self.RFootData[j][i] - (self.HeadData[j][i]-self.HeadData[j][0] ) )
                self.LNetPosData[j].append(self.LFootData[j][i] - (self.HeadData[j][i]-self.HeadData[j][0] ) )

        for i in range(1,len(self.RFootData[1])):
            for j in range(0,3):
                a = 1
                if self.myVelTest:
                    a = 0.5
                    # w_c = 2 * math.pi *
                    # tau = 1/w_c
                    # a = tau/tau+1
                self.RVelData[j].append((1-a)*self.RVelData[j][i-1] + a * (self.RFootData[j][i] - self.RFootData[j][i - 1]) / fixedDeltaTime)
                self.LVelData[j].append((1-a)*self.LVelData[j][i-1] + a  * (self.LFootData[j][i] - self.LFootData[j][i - 1]) / fixedDeltaTime)
                self.HeadVelData[j].append((1-a)*self.HeadVelData[j][i-1] + a * (self.HeadData[j][i] - self.HeadData[j][i - 1]) / fixedDeltaTime)
                self.RNetVelData[j].append( (self.RVelData[j][i - 1] - self.HeadVelData[j][i - 1]))
                self.LNetVelData[j].append((self.LVelData[j][i - 1] - self.HeadVelData[j][i - 1]))

            self.RSpeedData.append(math.sqrt(self.RVelData[0][i-1] **2 +
                                   self.RVelData[1][i-1] **2 +
                                   self.RVelData[2][i-1] **2))
            self.LSpeedData.append(math.sqrt(self.LVelData[0][i-1] ** 2 +
                                   self.LVelData[1][i-1] ** 2 +
                                   self.LVelData[2][i-1] ** 2))


    def get_after_lp_velocity(self,pre):
        self.firstHeadHeight = self.HeadData[1][0]
        return

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


    def OnFiltering(self,windowSize = filterSize, poly = 6):
        for i in range(0,3):
            self.RFootData[i] = savgol_filter(self.RFootData[i], windowSize, poly)
            self.LFootData[i] = savgol_filter(self.LFootData[i], windowSize, poly)
            self.HeadData[i] = savgol_filter(self.HeadData[i], windowSize, poly)

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

        #이상한 데이터
        if(len(self.stepIndexes) == 0):
            self.isOutLier = True
            return

        preE = 0
        for s,e,k in self.stepIndexes:
            if preE > s +5:
                #preE = e
                #print("error- 뛰어가는중!!!!!!:",self.fileName)
                if len(self.validHeads) > 0:
                    #print("error- 고치자!!!!!!:",self.fileName)
                    self.validHeads[len(self.validHeads)-1].validEnd = s
                    self.validHeads[len(self.validHeads)-1].make_data()
                #continue
            curStep = None
            if(k == 1): #Right
                curStep = Step(self,self.RFootData,self.RVelData,s,e)
                self.steps.append(curStep)
                self.netSteps.append(Step(self,self.RNetPosData,self.RNetVelData,s,e))
            else:#Left
                curStep = Step(self,self.LFootData, self.LVelData, s, e)
                self.steps.append(curStep)
                self.netSteps.append(Step(self,self.LNetPosData, self.LNetVelData, s, e))
            #head
            #hindexes= self.find_head_splitPoint(preE,e,self.HeadData,self.HeadVelData)
            h = Step(self,self.HeadData,self.HeadVelData,s,e,True,curStep)
            preE = e
            if h.verticalDistance > 0.06:
                self.validHeads.append(h)
                self.validHeadIndexes.append((s,e))
        if (len(self.validHeadIndexes) == 0):
            self.isOutLier = True
            return
        start = max( self.validHeadIndexes[0][0] - 10,0)

        dic = {"time" : np.array(list(range(0, len(self.HeadData[1]) - start))) * fixedDeltaTime,
               "index" : np.array(list(range(0, len(self.HeadData[1]) - start))),
               "y" : self.HeadData[1][start:],
               "velY" : self.HeadVelData[1][start:]}


        self.headTrajectoryDataFrame = pd.DataFrame(dic)
        start = self.stepIndexes[0][0]
        #print(start,len(np.array(list(range(0, len(self.RFootData[0]) - start)))),len(np.array(self.RFootData[1][start:])) )

        rFoot = self.RFootData
        lFoot = self.LFootData
        rRot = self.RRotData
        lRot = self.LRotData

        if self.stepIndexes[0][2] != 1:
            rFoot = self.LFootData
            lFoot = self.RFootData
            rRot = self.LRotData
            lRot = self.RRotData


        Td = {"index" : np.array(list(range(0, len(self.HeadData[0]) - start))),
              "Hx" : np.array(self.HeadData[0][start:]),
              "Hy" : np.array(self.HeadData[1][start:]),
              "Hz" :np.array(self.HeadData[2][start:]),
              "Rx" :  np.array(rFoot[0][start:]),
              "Ry" :  np.array(rFoot[1][start:]),
              "Rz" :  np.array(rFoot[2][start:]),
              "Lx" :  np.array(lFoot[0][start:]),
              "Ly" :  np.array(lFoot[1][start:]),
              "Lz" :  np.array(lFoot[2][start:]),
              "RRx": np.array(rRot[0][start:]),
              "RRy": np.array(rRot[1][start:]),
              "RRz": np.array(rRot[2][start:]),
              "LRx": np.array(lRot[0][start:]),
              "LRy": np.array(lRot[1][start:]),
              "LRz": np.array(lRot[2][start:])
        }
        self.allTrajectoryDataFrame = pd.DataFrame(Td)
        self.trajectoryLength = len(self.RFootData[0]) - start


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
            #plt.scatter(end, posData[1][end])

        if len(validStart) != len(validEnd):
            print("error - valid index")
            return [[0],[end-1]]

        if len(validStart) == 0:
            return (startIndex,endIndex)

        return (validStart[0],validEnd[0])

    def find_splitPoint(self,posData,velData,speedData):
        maxVelocity = max(velData[1])
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
                #or curYSum < 0.2 or ((posData[1][i] - posData[1][i-5]) < 0) and ((posData[1][i]) - posData[1][i+5] < 0 )
                if (curSum < validTH - 0.15 and abs(velData[1][i]) < 0.1):
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
        startIndex = curStep.validStart-10
        endIndex = curStep.validEnd + 10

        xAxis = np.array(list(range(0,endIndex - startIndex)))* fixedDeltaTime
        axes[0][i].plot(xAxis,self.HeadData[1][startIndex:endIndex], label="neak" + addedLabel )
        axes[0][i].plot(xAxis,self.LFootData[1][startIndex:endIndex],label = "ankle (L)"+ addedLabel)
        axes[0][i].plot(xAxis,self.RFootData[1][startIndex:endIndex], label = "ankle (R)" + addedLabel)

        axes[1][i].plot(self.HeadData[2][startIndex:endIndex],self.HeadData[1][startIndex:endIndex], label="neak" + addedLabel )
        axes[1][i].plot(self.LFootData[2][startIndex:endIndex],self.LFootData[1][startIndex:endIndex],label = "ankle (L)"+ addedLabel)
        axes[1][i].plot(self.RFootData[2][startIndex:endIndex],self.RFootData[1][startIndex:endIndex], label = "ankle (R)" + addedLabel)

        # self.HeadVelData = np.array(self.HeadVelData)
        # self.RVelData = np.array(self.RVelData)
        # self.LVelData = np.array(self.LVelData)
        # axes[1][i].plot(xAxis,self.HeadVelData[1][startIndex:endIndex],label="neak velocity" + addedLabel)
        # axes[1][i].plot(xAxis,self.RVelData[1][startIndex:endIndex],label = "ankle velocity(L)" + addedLabel)
        # axes[1][i].plot(xAxis,self.LVelData[1][startIndex:endIndex],label = "right velocity" + addedLabel)
        # axes[1][i].plot(xAxis,self.LVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex], label = "net velocity(L)"+ addedLabel)
        # axes[1][i].plot(xAxis,self.RVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex],label = "net velocity(R)"+ addedLabel)


        axes[0][i].grid(True)
        axes[1][i].grid(True)
        axes[0][i].legend()
        axes[1][i].legend()

    def DrawSimplePosAndVelGraph(self,_axes,color = None, label = None, startIndex = None, endIndex = None):
        axes = _axes
        rfoot = self.RFootData
        lfoot = self.LFootData

        if(startIndex == None):
            startIndex= 0
        if(endIndex == None):
            endIndex = -1
        startIndex = self.validHeads[0].validStart
        xAxis = np.array(list(range(0, len(self.RFootData[0])-startIndex-1 ))) * fixedDeltaTime

        axes[0].plot(xAxis,self.HeadData[1][startIndex:endIndex], color="r",alpha =0.2)

        self.HeadVelData = np.array(self.HeadVelData)
        self.RVelData = np.array(self.RVelData)
        self.LVelData = np.array(self.LVelData)

        xAxis = np.array(list(range(0, len(self.RFootData[0]) - startIndex - 1))) * fixedDeltaTime
        axes[1].plot(xAxis,self.HeadVelData[1][startIndex:endIndex], color="r",alpha =0.2)

        for head in self.validHeads:
            xAxis = np.array(list(range(head.startVelIndex- startIndex ,head.startVelIndex + startDeltaLength- startIndex))) * fixedDeltaTime


        axes[0].grid(True)
        axes[1].grid(True)

    def DrawPosAndVelGraph(self,_axes,color = None, label = None, startIndex = None, endIndex = None):
        axes = _axes
        rfoot = self.RFootData
        lfoot = self.LFootData

        if(startIndex == None):
            startIndex= 0
        if(endIndex == None):
            endIndex = -1

        xAxis = np.array(list(range(0, len(self.RFootData[0])-startIndex-1 ))) * fixedDeltaTime

        axes[0].plot(xAxis,self.HeadData[1][startIndex:endIndex], color=color, label="neak")
        axes[0].plot(xAxis,lfoot[1][startIndex:endIndex], color=color,label = "ankle(L)")
        axes[0].plot(xAxis,rfoot[1][startIndex:endIndex], color=color,label = "ankle(R)")
        # axes[0].plot(xAxis,self.LNetPosData[1][startIndex:endIndex], color=color, label="net ankle(L)")
        # axes[0].plot(xAxis, self.RNetPosData[1][startIndex:endIndex], color=color, label="net ankle(R)")

        self.HeadVelData = np.array(self.HeadVelData)
        self.RVelData = np.array(self.RVelData)
        self.LVelData = np.array(self.LVelData)
        for step in self.steps:
            axes[0].scatter(step.validStart * fixedDeltaTime, step.posData[1][0])
            axes[0].vlines(step.validStart * fixedDeltaTime, 0, 2,colors="black",linestyles="--")

        for step in self.steps:
            axes[0].scatter(step.validEnd * fixedDeltaTime, step.posData[1][len(step.posData[1]) - 1])
            # axes[0].vlines(step.validEnd * fixedDeltaTime,0,2,colors="r",linestyles="--")

        for step in self.steps:
            axes[1].vlines((step.validStart + step.maxYIndex) * fixedDeltaTime, 0, 2, colors="b", linestyles="--")
            axes[0].vlines((step.validStart + step.maxYIndex) * fixedDeltaTime, 0, 2, colors="b", linestyles="--")
            axes[0].vlines((step.validStart + step.maxVelIndex) * fixedDeltaTime,0, 2,colors="g", linestyles="--")
            axes[1].vlines((step.validStart + step.maxVelIndex) * fixedDeltaTime, 0, 2, colors="g", linestyles="--")


        xAxis = np.array(list(range(0, len(self.RFootData[0]) - startIndex - 1))) * fixedDeltaTime
        axes[1].plot(xAxis,self.HeadVelData[1][startIndex:endIndex], color="r",label="neck velocity")
        axes[1].plot(xAxis,self.LVelData[1][startIndex:endIndex],color=color,label = "ankle velocity(L)")
        axes[1].plot(xAxis,self.RVelData[1][startIndex:endIndex],color=color,label = "ankle velocity(R)")
        # axes[1].plot(xAxis,self.LVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex], color=color,label = "net velocity(L)")
        # axes[1].plot(xAxis,self.RVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex],color=color,label = "net velocity(R)")

        for head in self.validHeads:
            #axes[1].vlines(head.startVelIndex * fixedDeltaTime,0,2,colors="y",linestyles="--")
            axes[0].plot(np.array(list(range(head.validStart, head.validEnd)))* fixedDeltaTime,head.originPos[1][head.validStart:head.validEnd],color="indigo")
            axes[0].scatter(head.validEnd * fixedDeltaTime, head.posData[1][head.length-1])
            axes[0].vlines((head.validStart+head.maxVelIndex)* fixedDeltaTime, 0, 2, colors="y", linestyles="--")
            axes[1].vlines((head.validStart + head.maxVelIndex)* fixedDeltaTime, 0, 2, colors="y", linestyles="--")
            xAxis = np.array(list(range(head.startVelIndex- startIndex ,head.startVelIndex + startDeltaLength- startIndex))) * fixedDeltaTime
            axes[0].plot(xAxis, head.originPos[1][head.startVelIndex:head.startVelIndex + startDeltaLength], color="y")
            axes[1].plot(xAxis,head.originVel[1][head.startVelIndex:head.startVelIndex + startDeltaLength],color="y")

        for s in self.steps:
            xAxis = np.array(list(range(s.validStart + s.startDeltaIndex - startIndex, s.validStart + s.startDeltaIndex + startDeltaLength - startIndex))) * fixedDeltaTime
            axes[0].plot(xAxis, s.posData[1][s.startDeltaIndex:s.startDeltaIndex + startDeltaLength], color="black")
            axes[1].plot(xAxis,s.velData[1][s.startDeltaIndex:s.startDeltaIndex+startDeltaLength],color="black")
            xAxis = np.array(list(range(s.startVelIndex- startIndex ,s.startVelIndex + startDeltaLength- startIndex))) * fixedDeltaTime
            axes[0].plot(xAxis, s.originPos[1][s.startVelIndex:s.startVelIndex + startDeltaLength], color="y")
            axes[1].plot(xAxis,s.originVel[1][s.startVelIndex:s.startVelIndex + startDeltaLength],color="y")

        #
        # for s in self.netSteps:
        #     xAxis = np.array(list(range(s.validStart + s.startDeltaIndex, s.validStart + s.startDeltaIndex + startDeltaLength))) * fixedDeltaTime
        #     axes[0].plot(xAxis, s.posData[1][s.startDeltaIndex:s.startDeltaIndex + startDeltaLength], color="black")
        #     axes[1].plot(xAxis,s.velData[1][s.startDeltaIndex:s.startDeltaIndex+startDeltaLength],color="black")

        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].legend()
        axes[1].legend()


    def DrawPosAndVelGraph_z(self,_axes,color = None, label = None, startIndex = None, endIndex = None):
        axes = _axes
        rfoot = self.RFootData
        lfoot = self.LFootData

        if(startIndex == None):
            startIndex= 0
        if(endIndex == None):
            endIndex = -1

        #xAxis = np.array(list(range(0, len(self.RFootData[0])-startIndex-1 ))) * fixedDeltaTime

        axes[0].plot(self.HeadData[2][startIndex:endIndex],self.HeadData[1][startIndex:endIndex], color=color, label="neak")
        axes[0].plot(lfoot[2][startIndex:endIndex],lfoot[1][startIndex:endIndex], color=color,label = "ankle(L)")
        axes[0].plot(rfoot[2][startIndex:endIndex],rfoot[1][startIndex:endIndex], color=color,label = "ankle(R)")
        # axes[0].plot(xAxis,self.LNetPosData[1][startIndex:endIndex], color=color, label="net ankle(L)")
        # axes[0].plot(xAxis, self.RNetPosData[1][startIndex:endIndex], color=color, label="net ankle(R)")

        self.HeadVelData = np.array(self.HeadVelData)
        self.RVelData = np.array(self.RVelData)
        self.LVelData = np.array(self.LVelData)
        # for step in self.steps:
        #     axes[0].scatter(step.validStart * fixedDeltaTime, step.posData[1][0])
        #     axes[0].vlines(step.validStart * fixedDeltaTime, 0, 2,colors="black",linestyles="--")
        #
        # for step in self.steps:
        #     axes[0].scatter(step.validEnd * fixedDeltaTime, step.posData[1][len(step.posData[1]) - 1])
        #     # axes[0].vlines(step.validEnd * fixedDeltaTime,0,2,colors="r",linestyles="--")
        #
        # for step in self.steps:
        #     axes[1].vlines((step.validStart + step.maxYIndex) * fixedDeltaTime, 0, 2, colors="b", linestyles="--")
        #     axes[0].vlines((step.validStart + step.maxYIndex) * fixedDeltaTime, 0, 2, colors="b", linestyles="--")
        #     axes[0].vlines((step.validStart + step.maxVelIndex) * fixedDeltaTime,0, 2,colors="g", linestyles="--")
        #     axes[1].vlines((step.validStart + step.maxVelIndex) * fixedDeltaTime, 0, 2, colors="g", linestyles="--")


        xAxis = np.array(list(range(0, len(self.RFootData[0])-startIndex-1 ))) * fixedDeltaTime

        axes[1].plot(xAxis,self.HeadData[1][startIndex:endIndex], color=color, label="neak")
        axes[1].plot(xAxis,lfoot[1][startIndex:endIndex], color=color,label = "ankle(L)")
        axes[1].plot(xAxis,rfoot[1][startIndex:endIndex], color=color,label = "ankle(R)")
        # axes[1].plot(xAxis,self.LVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex], color=color,label = "net velocity(L)")
        # axes[1].plot(xAxis,self.RVelData[1][startIndex:endIndex] - self.HeadVelData[1][startIndex:endIndex],color=color,label = "net velocity(R)")
        #
        # for head in self.validHeads:
        #     #axes[1].vlines(head.startVelIndex * fixedDeltaTime,0,2,colors="y",linestyles="--")
        #     axes[0].plot(np.array(list(range(head.validStart, head.validEnd)))* fixedDeltaTime,head.originPos[1][head.validStart:head.validEnd],color="indigo")
        #     axes[0].scatter(head.validEnd * fixedDeltaTime, head.posData[1][head.length-1])
        #     axes[0].vlines((head.validStart+head.maxVelIndex)* fixedDeltaTime, 0, 2, colors="y", linestyles="--")
        #     axes[1].vlines((head.validStart + head.maxVelIndex)* fixedDeltaTime, 0, 2, colors="y", linestyles="--")
        #     xAxis = np.array(list(range(head.startVelIndex- startIndex ,head.startVelIndex + startDeltaLength- startIndex))) * fixedDeltaTime
        #     axes[0].plot(xAxis, head.originPos[1][head.startVelIndex:head.startVelIndex + startDeltaLength], color="y")
        #     axes[1].plot(xAxis,head.originVel[1][head.startVelIndex:head.startVelIndex + startDeltaLength],color="y")
        #
        # for s in self.steps:
        #     xAxis = np.array(list(range(s.validStart + s.startDeltaIndex - startIndex, s.validStart + s.startDeltaIndex + startDeltaLength - startIndex))) * fixedDeltaTime
        #     axes[0].plot(xAxis, s.posData[1][s.startDeltaIndex:s.startDeltaIndex + startDeltaLength], color="black")
        #     axes[1].plot(xAxis,s.velData[1][s.startDeltaIndex:s.startDeltaIndex+startDeltaLength],color="black")
        #     xAxis = np.array(list(range(s.startVelIndex- startIndex ,s.startVelIndex + startDeltaLength- startIndex))) * fixedDeltaTime
        #     axes[0].plot(xAxis, s.originPos[1][s.startVelIndex:s.startVelIndex + startDeltaLength], color="y")
        #     axes[1].plot(xAxis,s.originVel[1][s.startVelIndex:s.startVelIndex + startDeltaLength],color="y")

        #
        # for s in self.netSteps:
        #     xAxis = np.array(list(range(s.validStart + s.startDeltaIndex, s.validStart + s.startDeltaIndex + startDeltaLength))) * fixedDeltaTime
        #     axes[0].plot(xAxis, s.posData[1][s.startDeltaIndex:s.startDeltaIndex + startDeltaLength], color="black")
        #     axes[1].plot(xAxis,s.velData[1][s.startDeltaIndex:s.startDeltaIndex+startDeltaLength],color="black")

        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].legend()
        axes[1].legend()
        return


class Step():
    def __init__(self,origin,originPos,originVel,validStart,validEnd,isHead = False,pairFoot = None):
        self.origin : H2F_Data = origin
        self.originPos = originPos
        self.originVel = originVel
        self.validStart = validStart
        self.validEnd = validEnd
        self.posData = np.array(originPos)[:,validStart:validEnd]
        self.velData = np.array(originVel)[:,validStart:validEnd]
        self.pairFoot : Step = pairFoot
        self.length = 0
        self.maxY = 0
        self.maxVelTime = 0
        self.maxVelSlope = 0
        self.maxYIndex = 0
        self.maxVelIndex = 0
        self.maxYVel = 0
        self.ascentVelocity = 0
        self.descentVelocity = 0
        self.verticalDistance = 0
        self.lastY = 0
        self.ascentDistance = 0
        self.ascentTime = 0
        self.isHead = isHead
        self.startVelIndex = 0
        self.totalDistance = 0
        self.afterCrossVelocity = 1
        self.startDeltaTimeVelocity = 1
        self.startDeltaIndex = 1
        self.frame = {"time":[],"y":[],"velY":[]}
        self.make_data()

    def make_data(self):
        self.posData = np.array(self.originPos)[:,self.validStart:self.validEnd]
        self.velData = np.array(self.originVel)[:,self.validStart:self.validEnd]
        self.length = len(self.posData[0])
        self.maxY = max(self.posData[1])
        self.maxYVel = max(self.velData[1])
        self.maxYIndex = np.where(self.posData[1] == self.maxY)[0][0]
        self.maxVelIndex = np.where(self.velData[1] == self.maxYVel)[0][0]
        self.ascentDistance = self.maxY - self.posData[1][0]
        self.ascentTime = self.maxYIndex * fixedDeltaTime

        for i in range(len(self.posData[1])):
            self.frame["y"].append(self.posData[1][i])
            self.frame["velY"].append(self.velData[1][i])
            self.frame["time"].append(i * fixedDeltaTime)

        if self.maxYIndex != 0:
            self.ascentVelocity = (self.maxY - self.posData[1][0]) / (self.maxYIndex * fixedDeltaTime)
        if self.length - self.maxYIndex != 0:
            self.descentVelocity = (self.maxY-self.posData[1][self.length-1]) / ((self.length - self.maxYIndex) * fixedDeltaTime)
        self.verticalDistance = self.posData[1][self.length - 1] - self.posData[1][0]
        self.lastY = self.posData[1][self.length-1]

        startIndex = self.validStart
        if not self.isHead :
            # if abs(self.velData[1][0]) > 0.1:
            #     for i in range(self.validStart,0,-1):
            #         if abs(self.originVel[1][i]) < 0.1:
            #             startIndex = i;
            #             break
            # elif abs(self.velData[1][0]) < 0.05:
            #     for i in range(self.validStart,self.validEnd,1):
            #         if abs(self.originVel[1][i]) > 0.075:
            #             startIndex = i;
            #             break
            for i in range(self.validStart + 2,self.validEnd,1):
                if (self.originPos[1][i] - self.originPos[1][self.validStart] > 0.02 \
                        and self.originPos[1][i-1] - self.originPos[1][self.validStart] > 0.02 \
                        and self.originPos[1][i-2] - self.originPos[1][self.validStart] > 0.02) or self.originPos[1][i] - self.originPos[1][self.validStart] > 0.02 * 4 :
                    startIndex = i;
                    break;
        else:
            startIndex = self.pairFoot.startVelIndex


        self.maxVelTime = (self.validStart + self.maxVelIndex - startIndex) * fixedDeltaTime
        self.startVelIndex = startIndex

        if self.maxVelTime < 0.01:
            self.maxVelSlope = 0
        else:
            self.maxVelSlope = (self.maxYVel / self.maxVelTime)
        for i in range(1,self.length):
             curDistance = (self.posData[0][i] -self.posData[0][i-1])**2 \
                                  + (self.posData[1][i] -self.posData[1][i-1])**2 \
                                    + (self.posData[1][i] - self.posData[1][i - 1]) ** 2
             self.totalDistance += math.sqrt(curDistance)

        self.startDeltaTimeVelocity = self.originVel[1][self.startVelIndex]#(self.originPos[1][self.startVelIndex + startDeltaLength]- self.originPos[1][self.startVelIndex]) / startDeltaLength / fixedDeltaTime

        if(not self.isHead):
            startIndex = 0
            for i in range(self.validStart,self.validEnd - startDeltaLength):
                if abs(self.origin.RFootData[1][i] -self.origin.LFootData[1][i]) < 0.01:
                    startIndex = i - self.validStart
                    break
            self.startDeltaIndex = startIndex
            self.afterCrossVelocity = (self.posData[1][startIndex + startDeltaLength]- self.posData[1][startIndex]) / fixedDeltaTime/startDeltaLength


    def DrawStartToMax(self):
        plt.plot(list(range(self.validStart,self.validStart + self.maxYIndex)) ,self.originPos[1][self.validStart:self.validStart + self.maxYIndex])

    def Draw(self):
        plt.plot(self.originPos[1][0:],color="r")
        plt.plot(list(range(self.validStart, self.validEnd)),
                 self.originPos[1][self.validStart:self.validEnd])
        plt.plot(list(range(self.validStart, self.validStart + self.maxYIndex)),
                 self.originPos[1][self.validStart:self.validStart + self.maxYIndex],color="gold")

        plt.scatter(self.validStart-1 + self.maxYIndex,self.maxY)

    def IsOutlier(self,avgDict,SDDict):
        if not self.isHead:
            if self.length/avgDict["length"] < 0.4 or self.length/avgDict["length"] > 2 or abs(self.length - avgDict["length"]) > 3 * SDDict["length"]:
                print("length out lier")
                return True
            if self.verticalDistance/avgDict["verticalDistance"] < 0.3 \
                    or self.verticalDistance/avgDict["verticalDistance"] > 2 \
                    or abs(self.verticalDistance - avgDict["verticalDistance"]) > 3 * SDDict["verticalDistance"]:
                print("verticalDistance out lier")
                return True
            if abs(self.maxY - avgDict["maxY"]) > 3 * SDDict["maxY"] :
                print("maxY out lier")
                return True
        if self.isHead:
            if self.length/avgDict["length"] < 0.5 or self.length/avgDict["length"] > 2 or abs(self.length - avgDict["length"]) > 3 * SDDict["length"]:
                print("length out lier")
                return True
            if self.verticalDistance / avgDict["verticalDistance"] < 0.3 \
                    or (self.verticalDistance / avgDict["verticalDistance"])> 2 or \
                    abs(self.verticalDistance - avgDict["verticalDistance"]) > 3 * SDDict["verticalDistance"]:
                print("verticalDistance out lier")
                return True
        return False

    def WriteInfo(self,dict):
        dict["descentVelocity"]+= self.descentVelocity
        dict["ascentVelocity"] += self.ascentVelocity
        dict["maxVelocity"] += self.maxYVel
        dict["maxVelTime"] += self.maxVelTime
        dict["maxVelSlope"] += self.maxVelSlope
        dict["ascentDistance"] += self.ascentDistance
        dict["ascentTime"] += self.ascentTime
        dict["length"] += self.length
        dict["verticalDistance"] += self.verticalDistance
        dict["totalDistance"] += self.totalDistance
        dict["lastY"] += self.lastY
        dict["maxY"] += self.maxY
        dict["startDeltaTimeVelocity"] += self.startDeltaTimeVelocity
        dict["afterCrossVelocity"] += self.afterCrossVelocity

    def WriteSD(self,avgdict,sdDict):
        sdDict["descentVelocity"] += (avgdict["descentVelocity"]-self.descentVelocity)**2
        sdDict["ascentVelocity"] += (avgdict["ascentVelocity"]-self.ascentVelocity)**2
        sdDict["maxVelTime"] += (avgdict["maxVelTime"]-self.maxVelTime)**2
        sdDict["maxVelSlope"] += (avgdict["maxVelSlope"]-self.maxVelSlope)**2
        sdDict["maxVelocity"] += (avgdict["maxVelocity"]-self.maxYVel)**2
        sdDict["ascentDistance"] += (avgdict["ascentDistance"]-self.ascentDistance)**2
        sdDict["ascentTime"] += (avgdict["ascentTime"]-self.ascentTime)**2
        sdDict["length"] += (avgdict["length"]-self.length)**2
        sdDict["verticalDistance"] += (avgdict["verticalDistance"]-self.verticalDistance)**2
        sdDict["totalDistance"] += (avgdict["totalDistance"] - self.totalDistance) ** 2
        sdDict["lastY"] += (avgdict["lastY"]-self.lastY)**2
        sdDict["maxY"] += (avgdict["maxY"]-self.maxY)**2
        sdDict["startDeltaTimeVelocity"] +=  (avgdict["startDeltaTimeVelocity"]-self.startDeltaTimeVelocity)**2
        sdDict["afterCrossVelocity"] += (avgdict["afterCrossVelocity"]-self.afterCrossVelocity)**2


class StepAnalyzer():
    order = ["Head 1", "Head 2", "First Foot", "Second Foot", "Last Foot", "Net speed(second)",
             "Net speed(last)"]
    def __init__(self,files,isDebug = False,condition ="stair1"):
        self.data : H2F_Data = []
        self.firstHeads = []
        self.lastHeads = []
        self.condition = condition

        self.isDebug = isDebug
        self.avgDicts = []
        self.sdDicts = []
        self.stepDict = dict()
        self.HeadFootRate = []
        self.dataFrameDict = dict()
        self.dataFrame = pd.DataFrame({"time":[],"index":[],"y":[],"velY":[]})
        self.make_steps(files)
        #self.GetHeadHeightChange()
        self.AnalyzeHead()
        self.AnalyzeFoot()
        self.AnalyzeNetStep()

    def MakeHeadFullTrajectoryData(self):
        df = pd.DataFrame()
        for d in self.data:
            df = pd.concat([df,d.headTrajectoryDataFrame])
        return df

    def MakeAllTrajectoryToCSV(self,path):
        df = pd.DataFrame()
        meanLength = 0
        for d in self.data:
            df = pd.concat([df, d.allTrajectoryDataFrame])
            meanLength += (d.trajectoryLength) /len(self.data)

        meanLength = int(meanLength)
        meanData = {"Hx":[],"Hy":[],"Hz":[],"Rx":[],"Ry":[],"Rz":[],"Lx":[],"Ly":[],"Lz":[],"RRx":[],"RRy":[],"RRz":[],"LRx":[],"LRy":[],"LRz":[]}

        for i in range(meanLength):
            for k in meanData.keys():
                if(len(df.loc[df['index'] == i][k]) == 0):
                   break
                else:
                    m = df.loc[df['index'] == i][k].mean()
                meanData[k].append(m)

        resultDF = pd.DataFrame(meanData)

        HeadTxt = ""
        RFootTxt = ""
        LFootTxt = ""
        rRotationTxt = ""
        lRotationTxt = ""
        timeDataTxt = ""
        for i in range(len(meanData["Hx"])-10):
            HeadTxt += (HeadOffset + Vector3(meanData["Hx"][i],meanData["Hy"][i],meanData["Hz"][i])).ToString() + "\n"
            RFootTxt += (RfootOffset + Vector3(meanData["Rx"][i],meanData["Ry"][i],meanData["Rz"][i])).ToString() + "\n"
            LFootTxt += (LfootOffset + Vector3(meanData["Lx"][i],meanData["Ly"][i],meanData["Lz"][i])).ToString() + "\n"
            rRotationTxt +=  Vector3(meanData["RRx"][i],meanData["RRy"][i],meanData["RRz"][i]).ToString() + "\n"
            lRotationTxt += Vector3(meanData["LRx"][i], meanData["LRy"][i], meanData["LRz"][i]).ToString() + "\n"
            timeDataTxt += str(fixedDeltaTime) + "\n"

        os.makedirs(path, exist_ok=True)
        resultDF.to_csv(path + "trajectory.csv")
        writeData(path+"WaistData.txt",HeadTxt)
        writeData(path + "Rfootdata.txt", RFootTxt)
        writeData(path + "Lfootdata.txt", LFootTxt)
        writeData(path + "RfootRotationData.txt", rRotationTxt)
        writeData(path + "LfootRotationData.txt", lRotationTxt)
        writeData(path+"TimeData.txt",timeDataTxt)

    def DrawAVGHeadGraph(self,axes):
        print(self.dataFrame)
        sns.lineplot(x="time", y="velY",hue="type", data=self.dataFrame, ax=axes[1])
        sns.lineplot(x="time",y="y",hue="type", data=self.dataFrame,ax=axes[0])

    def DrawAllHeadGraph(self,axes):
        for d in self.data:
            d.DrawSimplePosAndVelGraph(axes)

    def DrawLengthPerAscent(self,axes,color,num,label=None,marker = None):
        #color1 = color
        #color2 = "C" + str(num+1)
        isFirst = True

        for d in self.data:
            if not d.isOutLier and len(d.steps) > 2 and len(d.validHeads) > 1:
                if isFirst:
                    axes[0].scatter(d.validHeads[0].maxYVel,d.steps[1].startDeltaTimeVelocity,color = color,label = label,marker = marker)
                    #axes[0].scatter(d.validHeads[1].maxYVel, d.steps[2].startDeltaTimeVelocity, color = color2,label = label+"(half)")
                    isFirst= False
                else:
                    axes[0].scatter(d.validHeads[0].maxYVel,d.steps[1].startDeltaTimeVelocity,color = color,marker = marker)

    def DrawLengthPerAscent1(self,axes,color,num,label=None,marker = None):
        #color1 = color
        #color2 = "C" + str(num+1)
        isFirst = True
        for d in self.data:
            if not d.isOutLier and len(d.steps) > 2 and len(d.validHeads) > 1:
                curColor = self.getColor(d.validHeads[0].length)
                axes[0].scatter(d.steps[1].startDeltaTimeVelocity,d.validHeads[0].maxYVel,color = curColor,marker = marker)
                curColor = self.getColor(d.validHeads[1].length)
                axes[0].scatter(d.steps[2].startDeltaTimeVelocity,d.validHeads[1].maxYVel, color=curColor, marker=marker)

    def getColor(self,length):
        c = {45:"lightcoral",50:"red",55:"tomato",60:"orange",65:"gold",70:"yellow",75:"yellowgreen",80:"green",90:"mediumspringgreen",95:"blue",100:"mediumpurple",105:"purple"}
        color = "gray"
        nearest = abs(115-length)
        p = (length - 30) / (120 -  30)
        if p > 1: p =1
        if p < 0: p = 0
        # for i in c.keys():
        #     if nearest > abs(length - i):
        #         nearest = abs(length - i)
        #         color = c[i]
        return (p,0.1,0.1)


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
            i = 0
            for o in self.order:
                w.writerow([o])
                w.writerow(self.avgDicts[i].keys())
                w.writerow(self.avgDicts[i].values())
                w.writerow(self.sdDicts[i].keys())
                w.writerow(self.sdDicts[i].values())
                i += 1
    def GetHeadFootRate(self):
        cDict = {"ascentVelocity":0,"maxVelocity":0}

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

    def AnalyzeFoot(self):
        curSteps = [[],[],[]]
        df = [defalutDF.copy()] * 3

        for data in self.data:
            if len(data.steps) > 2:
                for i in range(0,3):
                    s = data.steps[i]
                    if s:
                        curSteps[i].append(s)


        for i in range(len(data.steps)):
            print("-------------Foot" + str(i) + "------------")
            if self.isDebug:
                for s in curSteps[i]:
                    s.Draw()
                plt.show()
            self.AnalyzeStep(curSteps[i]) # 아웃라이어 제거 및 평균, 표준편차 구하기

        for i in range(len(curSteps)):
            for s in curSteps[i]:
                df[i] = pd.concat([df[i], pd.DataFrame(s.frame)])

        self.dataFrameDict["First Foot"] = df[0]
        self.dataFrameDict["Second Foot"] = df[1]
        self.dataFrameDict["Last Foot"] = df[2]

    def AnalyzeNetStep(self):
        netSteps =[[],[]]
        for data in self.data:
            if not data.isOutLier:
                netSteps[0].append(data.netSteps[1])
                netSteps[1].append(data.netSteps[len(data.netSteps)-1])

        print("-------------Net speed2------------")
        if self.isDebug:
            for s in netSteps[0]:
                s.Draw()
            plt.show()
        self.AnalyzeStep(netSteps[0],False)
        if self.isDebug:
            for s in netSteps[1]:
                s.Draw()
            plt.show()
        print("-------------Net speed3------------")
        self.AnalyzeStep(netSteps[1],False)


    def AnalyzeHead(self):
        df1  = pd.DataFrame(pd.DataFrame({"time":[],"y":[],"velY":[]}))
        df2 = pd.DataFrame(pd.DataFrame({"time":[],"y":[],"velY":[]}))
        for data in self.data:
            fh = data.GetFirstHead()
            lh = data.GetLastHead()
            if fh:
                self.firstHeads.append(fh)
                fh.frame["type"] = ["Head 1"] * len(fh.frame["time"])
                df1 = pd.concat([df1,pd.DataFrame(fh.frame)])
            if lh:
                self.lastHeads.append(lh)
                lh.frame["type"] = ["Head 2"] * len(lh.frame["time"])
                df2 = pd.concat([df2, pd.DataFrame(lh.frame)])

        self.dataFrameDict["Head 1"] = df1
        self.dataFrameDict["Head 2"] = df2

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

    def GetAvgInfo(self,steps):
        infoDict = {"ascentVelocity" : 0,
                    "ascentDistance" :0,
                    "ascentTime" : 0,
                    "descentVelocity": 0,
                    "length" : 0,
                    "verticalDistance" : 0,
                    "totalDistance" : 0,
                    "maxVelocity": 0,
                    "maxVelTime": 0,
                    "maxVelSlope": 0,
                    "lastY":0,
                    "maxY": 0,
                    "startDeltaTimeVelocity":0,
                    "afterCrossVelocity" : 0}
        for s in steps:
            s.WriteInfo(infoDict)
        for v in infoDict.keys():
            infoDict[v] /= len(steps)
        return infoDict

    def GetSDInfo(self,avgInfo,steps):
        infoDict = {"ascentVelocity" : 0,
                    "ascentDistance" :0,
                    "ascentTime" : 0,
                    "descentVelocity": 0,
                    "length" : 0,
                    "verticalDistance" : 0,
                    "totalDistance" : 0,
                    "maxVelocity": 0,
                    "maxVelTime": 0,
                    "maxVelSlope": 0,
                    "lastY":0,
                    "maxY": 0,
                    "startDeltaTimeVelocity":0,
                    "afterCrossVelocity" : 0}
        for s in steps:
            s.WriteSD(avgInfo,infoDict)
        for v in infoDict.keys():
            infoDict[v] /= len(steps)
            infoDict[v] = math.sqrt(infoDict[v])
        return infoDict


    def AnalyzeStep(self,steps,removeOutLier = True):
        infoDict = self.GetAvgInfo(steps)
        SDDict = self.GetSDInfo(infoDict, steps)
        print("Before remove OutLier:" ,infoDict)
        if removeOutLier:
            self.RemoveOutlier(steps,infoDict,SDDict)

        if(self.isDebug):
            for s in steps:
                s.Draw()
            plt.title("After")
            plt.show()
        # else:
        #     plt.cla()
        infoDict = self.GetAvgInfo(steps)
        print("After remove OutLier:", infoDict)
        infoDict["total count"] = len(steps)
        SDDict = self.GetSDInfo(infoDict,steps)
        print("SD",SDDict)
        self.avgDicts.append(infoDict)
        self.sdDicts.append(SDDict)


    def RemoveOutlier(self,steps,infoDict,SDDict):
        rList =[]
        for i in range(len(steps)):
            if(steps[i].IsOutlier(infoDict,SDDict)):
                steps[i].Draw()
                print("out Lier : ",steps[i].origin.fileName)
                steps[i].origin.isOutLier = True
                rList.append(i)
        if self.isDebug:
            plt.title("OutLier")
            plt.show()
            # plt.cla()
        rList.sort(reverse=True)
        for i in rList:
            del steps[i]


class RecordedData():
    def __init__(self,folderName,format = 1,firstZero = True):
        self.Format = format
        self.RFootData = None
        self.LFootData = None
        self.firstZero = firstZero
        if(format == 1):
            self.init_1(folderName)
        elif(format == 2):
            self.init_2(folderName)
        elif(format == 3):
            self.init_3(folderName)


    def init_1(self,folderName):
        self.RFootData = RecordedFootData(folderName + "RightFootController.txt",self.firstZero)
        self.LFootData = RecordedFootData(folderName + "LeftFootController.txt",self.firstZero)
        self.HeadData = [[]]
        self.HeadRotation = [[]]
        self.testData = [[]]
        self.testVelData = [0]
        self.LoadHeadData(folderName + "otherData.txt")

        rfoot = []
        lfoot = []
        for i in range(3):
            #rfoot.append(savgol_filter(self.RFootData.blendPosData[i], filterSize, 6))
            #lfoot.append(savgol_filter(self.LFootData.blendPosData[i], filterSize, 6))
            rfoot.append(self.RFootData.blendPosData[i])
            lfoot.append(self.LFootData.blendPosData[i])

        #self.HeadData[1] = savgol_filter(self.HeadData[1], filterSize, 6)

        self.RVelData = [[0],[0],[0]]
        self.LVelData = [[0],[0],[0]]
        self.HeadVelData = [0]

        for i in range(1,len(rfoot[0])):
            for j in range(0,3):
                self.RVelData[j].append((rfoot[j][i] - rfoot[j][i-1]) / fixedDeltaTime)
                self.LVelData[j].append((lfoot[j][i] - lfoot[j][i - 1]) / fixedDeltaTime)
            self.HeadVelData.append((self.HeadData[1][i] - self.HeadData[1][i - 1]) / fixedDeltaTime)
            self.testVelData.append((self.testData[1][i] - self.testData[1][i - 1]) / fixedDeltaTime)

    def init_2(self,folderName):
        self.RFootData = loadData(folderName + "Rfootdata.txt",True)
        self.LFootData = loadData(folderName + "Lfootdata.txt",True)
        self.HeadData = loadData(folderName + "WaistData.txt")
        #self.HeadData = np.array(HeadData)

        self.RFootData[1] = savgol_filter(self.RFootData[1], filterSize, 6)
        self.LFootData[1] = savgol_filter(self.LFootData[1], filterSize, 6)
        self.HeadData[1] = savgol_filter(self.HeadData[1], filterSize, 6)
        self.RVelData = [[0],[0],[0]]
        self.LVelData = [[0],[0],[0]]
        self.HeadVelData = [0]

        for i in range(1,len(self.RFootData[1])):
            for j in range(0,3):
                self.RVelData[j].append((self.RFootData[j][i] - self.RFootData[j][i-1]) / fixedDeltaTime)
                self.LVelData[j].append((self.LFootData[j][i] - self.LFootData[j][i - 1]) / fixedDeltaTime)
            self.HeadVelData.append((self.HeadData[1][i] - self.HeadData[1][i - 1]) / fixedDeltaTime)
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
                self.RVelData[j].append((self.RFootData[j][i] - self.RFootData[j][i-1]) / fixedDeltaTime)
                self.LVelData[j].append((self.LFootData[j][i] - self.LFootData[j][i - 1]) / fixedDeltaTime)
                self.ankleVelData.append((self.ankleData[1][i] - self.ankleData[1][i - 1]) / fixedDeltaTime)
            self.HeadVelData.append((self.HeadData[1][i] - self.HeadData[1][i - 1]) / fixedDeltaTime)

        self.HighestPoint = []
        self.ChangePoint = []

    def LoadHeadData(self,fileName):
        f = open(fileName, 'r')
        data = f.read()
        dataList = data.split("other\n")
        d = dataList[2].split("####\n")
        self.HeadData = makeVectorData(d[0].split('\n'),False)
        self.HeadRotation = makeVectorData(d[1].split('\n'),False)

        d1 = dataList[1].split('####\n')
        self.testData = makeVectorData(d1[0].split("\n"),False)
        f.close()

    def makeTimeString(self,start=0,end=-1):
        dataString = ""
        if end == -1:
            end = len(self.HeadData[0])
        ran = range(start, end)
        for i in ran:
            dataString += str(fixedDeltaTime) + "\n"
        return dataString


    def makeOtherDataString(self,start=0,end=-1):
        dataTxt = "other\n"+Vector3ArrayToString(self.testData,start,end) + "####\n"+Vector3ArrayToString(self.HeadRotation,start,end)
        return dataTxt

    def MatchMid(self,s,e,originLists,newLists):
        newS = s; newE = e;
        mid1 = s + FindMaxIndex(originLists[s:e])
        mid2 = FindMaxIndex(newLists)
        transX = (s + mid2) - mid1

        if(transX > s):
            transX = s
        if(e - transX > len(originLists)-1):
            transX = e - (len(originLists)-1)

        newS = newS - transX
        newE = newE - transX

        print(transX,s,e,newS,newE,len(originLists))

        return newS,newE

    def FindStartAndEndIndex(self):
        rightS = self.findStartPoint2(self.RFootData.realPosData[1])
        leftS = self.findStartPoint2(self.LFootData.realPosData[1])
        rightE = spl.FindEndPoint(self.RFootData.realPosData[1])
        leftE = spl.FindEndPoint(self.LFootData.realPosData[1])
        print("end:", rightE*fixedDeltaTime,leftE * fixedDeltaTime)
        return min(rightS,leftS),max(rightE,leftE)


    def HeightTrajectorySynthesize(self,newYData,axes):
        s = max(self.findStartPoint2(self.RFootData.realPosData[1]) ,0)
        end = min(s + len(newYData)-1,len(self.RFootData.realPosData[0])-1)
        axes[0].scatter(s * fixedDeltaTime, self.RFootData.realPosData[1][s])
        axes[0].scatter(end * fixedDeltaTime, self.RFootData.realPosData[1][end])

        s,end = self.MatchMid(s,end,self.RFootData.realPosData[1],newYData)

        for i in range(s,end):
            self.RFootData.realPosData[1][i] = self.RFootData.realPosData[1][s] + newYData[i-s]
        self.RFootData.realPosData[1][end:] = [self.RFootData.realPosData[1][s] + newYData[i-s]] * len(self.RFootData.realPosData[1][end:])

        s = self.findStartPoint2(self.LFootData.realPosData[1])
        end = min(s + len(newYData)-1,len(self.LFootData.realPosData[0])-1)

        axes[0].scatter(s * fixedDeltaTime, self.LFootData.realPosData[1][s])
        axes[0].scatter(end * fixedDeltaTime, self.LFootData.realPosData[1][end])

        s,end = self.MatchMid(s,end,self.LFootData.realPosData[1],newYData)

        for i in range(s,end):
            self.LFootData.realPosData[1][i] = self.LFootData.realPosData[1][s] + newYData[i-s]
        self.LFootData.realPosData[1][end:] = [self.LFootData.realPosData[1][s] + newYData[i - s]] * len(self.LFootData.realPosData[1][end:])

        s = s + 50 + self.findStartPoint2(self.RFootData.blendPosData[1][s+50:])
        end = min(s + len(newYData)-1,len(self.RFootData.realPosData[0])-1)

        s, end = self.MatchMid(s, end, self.RFootData.blendPosData[1], newYData)

        print("l:",len(self.RFootData.realPosData[1]),"e:",end)
        axes[0].scatter(s * fixedDeltaTime, self.RFootData.realPosData[1][s])
        axes[0].scatter(end * fixedDeltaTime, self.RFootData.realPosData[1][end])

        for i in range(s,end):
            self.RFootData.realPosData[1][i] = self.RFootData.realPosData[1][s] + newYData[i-s]
        xA = np.array(list(range(0, len(self.RFootData.realPosData[1]) ))) * fixedDeltaTime

        axes[0].plot(xA,self.RFootData.realPosData[1], color="r", label="add(R)")
        axes[0].plot(xA,self.LFootData.realPosData[1], color="b", label="add(L)")

    def writeToTxt1(self,path,start=0,end=-1):
        if end > len(self.RFootData.blendPosData[0]) - 1:
            end = len(self.RFootData.blendPosData[0]) - 1
        writeData(path + "RightFootController.txt",self.RFootData.to_txt(start,end))
        writeData(path + "LeftFootController.txt",self.LFootData.to_txt(start,end))
        writeData(path + "otherData.txt",self.makeOtherDataString(start, end))
        writeData(path + "timeData.txt",self.makeTimeString(start,end))

    def writeToTxt(self,path,start=0,end=-1):
        if end > len(self.RFootData.blendPosData[0])-1:
            end = len(self.RFootData.blendPosData[0])-1
        writeData(path + "Rfootdata.txt",Vector3ArrayToString(self.RFootData.blendPosData,start,end))
        writeData(path + "Lfootdata.txt", Vector3ArrayToString(self.LFootData.blendPosData,start,end))
        writeData(path + "RfootRotationData.txt", Vector3ArrayToString(self.RFootData.realRotData,start,end))
        writeData(path + "LfootRotationData.txt", Vector3ArrayToString(self.LFootData.realRotData,start,end))
        writeData(path + "WaistData.txt", Vector3ArrayToString(self.HeadData,start,end))


    def DrawHeadGraph(self,axes,color = None,additionalLabel = "", startIndex = None, endIndex = None,avgInfo = None,addtionalHeight = 0,transX = 0):
        startIndex = max(self.findStartPoint(self.HeadVelData) -10 ,0)

        startHeight = self.HeadData[1][startIndex]

        xAxis = np.array(list(range(0, len(self.HeadData[1])-startIndex) )) * fixedDeltaTime

        meanData = []

        for i in list(range(0, len(self.HeadData[1])-startIndex) ):
            if(len(avgInfo.loc[avgInfo['index'] == i]["velY"]) == 0):
                m = 0
            else:
                m = avgInfo.loc[avgInfo['index'] == i]["velY"].mean()
            meanData.append(m)

        axes[0].plot(xAxis, np.array(self.HeadData[1][startIndex:endIndex]) -startHeight, color=color,label="head" + additionalLabel)
        axes[1].plot(xAxis,self.HeadVelData[startIndex:endIndex] - np.array(self.testVelData[startIndex:endIndex]),label="head speed"+ additionalLabel)
        axes[1].plot(xAxis,np.array(meanData),label = "test")
        print(dtw(meanData,self.HeadVelData[startIndex:endIndex]))


        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].legend(loc = "upper right")
        axes[1].legend(loc = "upper right")

    def DrawPosAndVelGraph(self,axes,color = None,additionalLabel = "", startIndex = None, endIndex = None,addtionalHeight = 0,transX = 0):
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
        rfoot = np.array(rfoot)
        lfoot = np.array(lfoot)


        xAxis = np.array(list(range(startIndex+1 + transX, len(self.HeadData[1]) +transX))) * fixedDeltaTime
        axes[0].plot(xAxis,np.array(self.HeadData[1][startIndex:endIndex]) + addtionalHeight, color=color, label="head" + additionalLabel)
        axes[0].plot(xAxis, np.array(self.testData[1][startIndex:endIndex]) + addtionalHeight, color=color,
                     label="test" + additionalLabel)
        xAxis = np.array(list(range(startIndex + 1 + transX, len(rfoot[1]) + transX))) * fixedDeltaTime
        axes[0].plot(xAxis,rfoot[1][startIndex:endIndex], color=color,label = "Rfoot"+ additionalLabel)
        axes[0].plot(xAxis, lfoot[1][startIndex:endIndex], color=color, label="Lfoot" + additionalLabel)

        if self.Format == 1:
            #axes[0].plot(xAxis,self.RFootData.realPosData[1][startIndex:endIndex], color="indigo",label = "Lfoot(input)"+ additionalLabel)
            #axes[0].plot(xAxis,self.LFootData.realPosData[1][startIndex:endIndex], color="gold",label = "Rfoot(input)"+ additionalLabel)
            axes[0].plot(xAxis, self.RFootData.realPosData[1][startIndex:endIndex], color="indigo",
                         label="Lfoot(real)" + additionalLabel)
            axes[0].plot(xAxis, self.LFootData.realPosData[1][startIndex:endIndex], color="gold",
                         label="Rfoot(real)" + additionalLabel)

        self.HeadVelData = np.array(self.HeadVelData)
        self.RVelData = np.array(self.RVelData)
        self.LVelData = np.array(self.LVelData)


        xAxis = np.array(list(range(startIndex+1 + transX, len(self.HeadData[1]) + transX ))) * fixedDeltaTime
        axes[1].plot(xAxis,self.HeadVelData[startIndex:endIndex],label="head speed"+ additionalLabel)
        xAxis = np.array(list(range(startIndex + 1 + transX, len(rfoot[1]) + transX))) * fixedDeltaTime
        axes[1].plot(xAxis,self.RVelData[1][startIndex:endIndex],color=color,label = "RFoot speed"+ additionalLabel)
        axes[1].plot(xAxis,self.LVelData[1][startIndex:endIndex],color=color,label = "LFoot speed"+ additionalLabel)

        # if self.Format == 1:
        #     axes[1].plot(xAxis,self.RFootData.realVelData[1][startIndex:endIndex], color="indigo",label = "Lfoot(input)"+ additionalLabel)
        #     axes[1].plot(xAxis,self.LFootData.realVelData[1][startIndex:endIndex], color="gold",label = "Rfoot(input)"+ additionalLabel)
        # axes[1].plot(self.LVelData[1][startIndex:endIndex] - self.HeadVelData[startIndex:endIndex], color=color,label = "LFoot speed- head speed")
        # axes[1].plot(self.RVelData[1][startIndex:endIndex] - self.HeadVelData[startIndex:endIndex],color=color,label = "RFoot speed- head speed")

        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].legend(loc = "upper right")
        axes[1].legend(loc = "upper right")
        return

    def findStartPoint(self,Veldata):
        for i in range(10,len(Veldata)):
            check = True
            for j in range(i,i+10):
                if Veldata[j] < 0.05 or Veldata[j] > 1:
                    check = False
                    break
            if check:
                return i
    def findStartPoint2(self,posData):
        for i in range(len(posData)-3):
            if (posData[i]-posData[0]) > 0.02 and (posData[i+1]-posData[0]) > 0.02 and (posData[i+2]-posData[0]) > 0.02:
                print(i)
                return i
        print("not found")
        return i


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
        self.fixedDeltaTime = fixedDeltaTime
        self.length = 0
        self.validMovement = 0
        self.loadData()


    def loadData(self):
        self.valid = True
        self.posData = loadData(self.fileName,True)
        self.length = len(self.posData[0])
        self.posData[1] = savgol_filter(self.posData[1], filterSize, 6).tolist()
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


def DrawGraph(path):
    data = []
    #for path in pathArray:
    data.append(loadPosData(path).copy())
    plt.plot(data[0][1],label = "L")


def ReadAndDrawGraph(pathR,pathL,rcolor="r",lcolor="b"):
    data = []
    #for path in pathArray:
    if(not os.path.isfile(pathR)):
        print("not exist: "+pathR)
        return
    data.append(loadData(pathL,True).copy())
    data.append(loadData(pathR,True).copy())
    #for i in range(int(len(data)/2)):
    plt.plot(data[0][1],label = "L",color=rcolor)
    plt.plot(data[1][1], label = "R",color=lcolor)
    #plt.plot(data[0][0],label = "L",color=rcolor)
    #plt.plot(data[1][0], label = "R",color=lcolor)

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


sIndex = 0
fIndex = 20
#ShowReal_short(s,f,'timeCompare/plane',"C1","C2")
#f, axes = plt.subplots(2, 1)
g = plt

def GetHeadDataFrame(condition):
    f  = os.getcwd().replace("\\","/",10)
    df = pd.read_csv(f + "/dataFrame/" + condition +"/" + "HeadFull.csv")
    return df


def DrawAVGHeadGraph(axes,condition):
    d1 : pd.DataFrame = GetHeadDataFrame(condition)
    #sns.lineplot(x="time", y="velY", data=d1,ax=axes[1],label = condition+"(avg)")
    sns.lineplot(x="time", y="y", data=d1,ax=axes[0],label = condition+"(avg)")
    return d1


def dtw(s,t):
    n,m = len(s),len(t)
    dtw_matrix = np.zeros((n+1,m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i,j] = np.inf
    dtw_matrix[0,0] = 0

    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = abs(s[i-1]-t[j-1])
            last_min = np.min([dtw_matrix[i-1,j],dtw_matrix[i,j-1],dtw_matrix[i-1,j-1]])
            dtw_matrix[i,j] = cost + last_min
    return dtw_matrix[n,m]