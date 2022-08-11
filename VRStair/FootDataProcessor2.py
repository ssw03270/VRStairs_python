# 작성자 임수빈
# 데이터 처리용

import torch
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt

isUseOtherData = False

rFootName = "RightFootController"
lFootName = "LeftFootController"
posFile = "_pos.txt"
velFile = "_velocity.txt"
rotationFile = "_rotation.txt"
otherPosFile = "_otherPos.txt"
otherVelFile = "_otherVelocity.txt"
otherRotationFile = "_otherRotation.txt"
timeFile = "_timedata.txt"

dataFolder = 'sooData/'
# curDataFolder = "plane05/"
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getRotMatrix(angle):
    rotMat = [[math.cos(angle), 0, math.sin(angle)],
              [0, 1, 0],
              [-math.sin(angle), 0, math.cos(angle)]]
    return np.array(rotMat)


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

    def __rmul__(self,other):
        return Vector3(self.x * other, self.y * other, self.z * other)


class StepData():
    MIN = 1
    MAX = 0
    AvgAngle = [0] * 10
    typeCount = [0] * 10

    def __init__(self, folderName='', walkingType='', isLeft=False, loadWithNPZ=False):
        self.folderName = folderName
        self.inputData = []
        self.posData = []
        self.velData = []
        self.rotData = []
        self.otherPosData = []
        self.otherVelData = []
        self.otherRotData = []
        self.walkingType = walkingType
        self.totalFrame = 0
        self.stepTime = 0
        self.isLeft = isLeft
        self.posOffset = Vector3(0, 0, 0)
        self.dAngle = 0
        self.maxHeight = 0
        self.onLPF = True
        self.filterWeight = 0.5

        if loadWithNPZ == False:
            self.loadPosData()
            self.loadVelData()
            self.loadRotData()
            self.checkTotalTime()
        if isUseOtherData:
            self.loadOtherData()
        # self.getWalkingDirection(6)

    '''
    승원이가 만들어준 데이터로 로드하는거
    0~2 : pos
    3~5 : rot
    6~8 : vel
    '''
    def loadDataWithNPZ(self, data):
        self.totalFrame = len(data[0])
        for i in range(len(data[0])):
            self.posData.append(Vector3(data[0][i], data[1][i], data[2][i]))
            self.velData.append(Vector3(data[3][i], data[4][i], data[5][i]))
            self.rotData.append(Vector3(data[6][i], data[7][i], data[8][i]))

    def checkMInMax(self, vec):
        data = [vec.x, vec.y, vec.z]
        isUpdate = False
        for value in data:
            if value > StepData.MAX:
                StepData.MAX = value
                isUpdate = True
            if value < StepData.MIN:
                StepData.MIN = value
                isUpdate = True
        return isUpdate

    def getObjName(self):
        if self.isLeft:
            return lFootName
        else:
            return rFootName

    def lastXZ(self, isRot=False):
        l = self.totalFrame - 1
        pos = [self.posData[l].x, self.posData[l].z]
        if isRot:
            rotPos = getRotMatrix(-self.rotData[l].y) @ np.array(
                [self.posData[l].x, self.posData[l].y, self.posData[l].z])
            pos = [rotPos[0], rotPos[2]]
        return pos

    def lastPosition(self, isRot=False):
        l = self.totalFrame - 1
        pos = [self.posData[l].x, self.posData[l].y, self.posData[l].z]
        if isRot:
            rotPos = getRotMatrix(-self.rotData[l].y) @ np.array(
                [self.posData[l].x, self.posData[l].y, self.posData[l].z])
            pos = [rotPos[0], rotPos[1], rotPos[2]]
        return pos

    def makeInputData(self, start, seq, isRot=True):
        inputData = []
        if seq > self.totalFrame:
            print(self.folderName)
        if isRot:
            for i in range(start, seq):
                rotPos = getRotMatrix(-self.rotData[i].y) @ np.array(
                    [self.posData[i].x, self.posData[i].y, self.posData[i].z])
                rotVel = getRotMatrix(-self.rotData[i].y) @ np.array(
                    [self.velData[i].x, self.velData[i].y, self.velData[i].z])
                X = [rotPos[0], rotPos[1], rotPos[2],
                     rotVel[0], rotVel[1], rotVel[2],
                     self.rotData[i].x, i / self.totalFrame]
                # X = np.array(X)
                # X = (X-StepData.MIN)/(StepData.MAX-StepData.MIN)
            inputData.append(X)
        else:
            for i in range(start, seq):
                X = [self.posData[i].x, self.posData[i].y, self.posData[i].z,
                     self.velData[i].x, self.velData[i].y, self.velData[i].z,
                     self.rotData[i].x, self.rotData[i].y]
                inputData.append(X)
        return inputData

    '''
    LSTM에 사용할 데이터
    seq : seq의 길이
    end : 마지막번째 궤적의 인덱스.
    0부터 end까지 seq만큼의 position 을 뽑아서 input data로 사용. 
    '''

    def makeInputData1(self, seq, end, isRot=False):
        inputData = []

        if isRot:
            rotPos = getRotMatrix(-self.rotData[0].y) @ np.array(
                [self.posData[0].x, self.posData[0].y, self.posData[0].z])
            rotVel = getRotMatrix(-self.rotData[0].y) @ np.array(
                [self.velData[0].x, self.velData[0].y, self.velData[0].z])
            X = [rotPos[0], rotPos[1], rotPos[2],
                 rotVel[0], rotVel[1], rotVel[2],
                 self.rotData[0].x]
            inputData.append(X)

            rotPos = getRotMatrix(-self.rotData[end].y) @ np.array(
                [self.posData[end].x, self.posData[end].y, self.posData[end].z])
            rotVel = getRotMatrix(-self.rotData[end].y) @ np.array(
                [self.velData[end].x, self.velData[end].y, self.velData[end].z])
            X = [rotPos[0], rotPos[1], rotPos[2],
                 rotVel[0], rotVel[1], rotVel[2],
                 self.rotData[end].x]
            inputData.append(X)
            i = 1
            k = 0
            for j in range(1, seq - 1):
                rotPos = getRotMatrix(-self.rotData[i].y) @ np.array(
                    [self.posData[i].x, self.posData[i].y, self.posData[i].z])
                rotVel = getRotMatrix(-self.rotData[i].y) @ np.array(
                    [self.velData[i].x, self.velData[i].y, self.velData[i].z])
                X = [rotPos[0], rotPos[1], rotPos[2],
                     rotVel[0], rotVel[1], rotVel[2],
                     self.rotData[i].x]
                k += (end / (seq - 2))
                i = int(k)
                # X = np.array(X)
                # X = (X-StepData.MIN)/(StepData.MAX-StepData.MIN)
            inputData.append(X)
        else:
            i = 0
            k = 0
            X = [self.posData[0].x, self.posData[0].y, self.posData[0].z,
                 self.velData[0].x, self.velData[0].y, self.velData[0].z,
                 self.rotData[0].x, self.rotData[0].y]
            inputData.append(X)

            X = [self.posData[end].x, self.posData[end].y, self.posData[end].z,
                 self.velData[end].x, self.velData[end].y, self.velData[end].z,
                 self.rotData[end].x, self.rotData[end].y]
            inputData.append(X)

            for j in range(1, seq - 1):
                X = [self.posData[i].x, self.posData[i].y, self.posData[i].z,
                     self.velData[i].x, self.velData[i].y, self.velData[i].z,
                     self.rotData[i].x, self.rotData[i].y]
                inputData.append(X)
                k += (end / (seq - 2))
                i = int(k)
        return inputData

    # MLP에서 쓸 데이터
    # 시작점과 끝점, 경과시간, 속도를 input으로 학습
    def makeInputData2(self, end, isRot=False):
        if end >= self.totalFrame:
            end = self.totalFrame - 1
        if isRot:
            rotPos = getRotMatrix(-self.rotData[0].y) @ np.array(
                [self.posData[0].x, self.posData[0].y, self.posData[0].z])
            rotVel = getRotMatrix(-self.rotData[0].y) @ np.array(
                [self.velData[0].x, self.velData[0].y, self.velData[0].z])
            rotPos1 = getRotMatrix(-self.rotData[end].y) @ np.array(
                [self.posData[end].x, self.posData[end].y, self.posData[end].z])
            rotVel1 = getRotMatrix(-self.rotData[end].y) @ np.array(
                [self.velData[end].x, self.velData[end].y, self.velData[end].z])
            inputData = [rotPos[0], rotPos[1], rotPos[2],
                         rotVel[0], rotVel[1], rotVel[2],
                         self.rotData[0].x,
                         rotPos1[0], rotPos1[1], rotPos1[2],
                         rotVel1[0], rotVel1[1], rotVel1[2],
                         self.rotData[end].x]
        else:
            '''

            inputData = [self.posData[1].x, self.posData[1].y,self.posData[1].z,
                         self.velData[1].x, self.velData[1].y,self.velData[1].z,
                         self.rotData[1].x, self.rotData[1].y,
                         self.posData[end].x, self.posData[end].y,self.posData[end].z,
                         self.velData[end].x, self.velData[end].y,self.velData[end].z,
                         self.rotData[end].x, self.rotData[end].y]

                         self.posData[end - 1].x, self.posData[end - 1].y, self.posData[end - 1].z,
            #                         self.velData[end-1].x, self.velData[end-1].y,self.velData[end-1].z,
            inputData = [self.otherPosData[1].x, self.otherPosData[1].y, self.otherPosData[1].z,
                         self.otherRotData[1].x, self.otherRotData[1].y,
                         self.posData[1].x, self.posData[1].y, self.posData[1].z,
                         self.velData[1].x, self.velData[1].y, self.velData[1].z,
                         self.rotData[1].x, self.rotData[1].y,
                         self.posData[end - 1].x, self.posData[end - 1].y, self.posData[end - 1].z,
                         self.posData[end].x, self.posData[end].y, self.posData[end].z,
                         self.velData[end].x, self.velData[end].y, self.velData[end].z,
                         self.rotData[end].x, self.rotData[end].y]
                                                  self.velData[end - 1].x, self.velData[end - 1].y, self.velData[end - 1].z,
                         self.rotData[1].x, self.rotData[1].y,
                                 
                                     inputData = [self.posData[1].x, self.posData[1].y,self.posData[1].z,
                         self.velData[1].x, self.velData[1].y, self.velData[1].z,
                         self.rotData[1].x, self.rotData[1].y,
                         self.posData[end - 1].x, self.posData[end - 1].y, self.posData[end - 1].z,
                         self.posData[end].x, self.posData[end].y,self.posData[end].z,
                         self.velData[end].x, self.velData[end].y,self.velData[end].z,
                         self.rotData[end].x, self.rotData[end].y]
                         
                        inputData = [self.posData[1].x, self.posData[1].y,self.posData[1].z,
                         self.velData[1].x, self.velData[1].y, self.velData[1].z,
                         self.posData[end - 1].x, self.posData[end - 1].y, self.posData[end - 1].z,
                         self.velData[end - 1].x, self.velData[end - 1].y, self.velData[end - 1].z,
                         self.posData[end].x, self.posData[end].y,self.posData[end].z,
                         self.velData[end].x, self.velData[end].y,self.velData[end].z]
                         
 
                              inputData = [self.posData[1].x, self.posData[1].y, self.posData[1].z,
                         self.velData[1].x, self.velData[1].y, self.velData[1].z,
                         self.rotData[1].x, self.rotData[1].y,self.rotData[1].z,
                         self.posData[end - 1].x, self.posData[end - 1].y, self.posData[end - 1].z,
                         self.posData[end].x, self.posData[end].y, self.posData[end].z,
                         self.velData[end].x, self.velData[end].y, self.velData[end].z,
                         self.rotData[end].x, self.rotData[end].y,self.rotData[end].z]
            '''

            inputData = [self.posData[1].x, self.posData[1].y, self.posData[1].z,
                         self.velData[1].x, self.velData[1].y, self.velData[1].z,
                         self.rotData[1].x, self.rotData[1].y, self.rotData[1].z,
                         self.posData[end - 1].x, self.posData[end - 1].y, self.posData[end - 1].z,
                         self.velData[end - 1].x, self.velData[end - 1].y, self.velData[end - 1].z,
                         self.rotData[end - 1].x, self.rotData[end - 1].y, self.rotData[end - 1].z,
                         self.posData[end].x, self.posData[end].y, self.posData[end].z,
                         self.velData[end].x, self.velData[end].y, self.velData[end].z,
                         self.rotData[end].x, self.rotData[end].y, self.rotData[end].z]

        return inputData

    '''
    MLP에서 쓸 데이터
    seq : seq의 길이
    end : 마지막번째 궤적의 인덱스.
    0부터 end까지 seq만큼의 position 을 뽑아서 input data로 사용. 
    '''

    def makeInputData3(self, seq, end):
        inputData = []
        i = 1
        k = 0
        X = [self.posData[0].x, self.posData[0].y, self.posData[0].z,
             self.velData[0].x, self.velData[0].y, self.velData[0].z,
             self.rotData[0].x, self.rotData[0].y]
        inputData.extend(X)

        for j in range(1, seq - 1):
            X = [self.posData[i].x, self.posData[i].y, self.posData[i].z,
                 self.velData[i].x, self.velData[i].y, self.velData[i].z,
                 self.rotData[i].x, self.rotData[i].y]
            inputData.extend(X)
            k += ((end - 2) / (seq - 2))
            i = int(k)

        X = [self.posData[end].x, self.posData[end].y, self.posData[end].z,
             self.velData[end].x, self.velData[end].y, self.velData[end].z,
             self.rotData[end].x, self.rotData[end].y]
        inputData.extend(X)

        return inputData

    def getLineToVector3(self, line):
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        vec = Vector3(float(line[0]), float(line[1]), float(line[2]))

        return vec

    # 발이보고 있는 방향과 진행방향 사이의 각도 차이를 구함.
    # loadPosData,loadRotData 다음에 호출.
    def getWalkingDirection(self, seq):
        index = min(len(self.posData) - 1, seq)
        walkingDir = self.posData[index]
        magnitude = math.sqrt(walkingDir.x ** 2 + walkingDir.z ** 2)
        # print(walkingDir.x,walkingDir.z,magnitude)
        walkingDir = (walkingDir.x / magnitude, walkingDir.z / magnitude)
        angle = math.atan2(walkingDir[0], walkingDir[1])
        dAngle = (angle - self.rotData[0].y) * (180 / math.pi)
        dAngle %= 360
        if (dAngle > 180):
            dAngle = dAngle - 360
        # print(self.walkingType, ":" , dAngle)
        self.dAngle = dAngle * (math.pi / 180)
        # print("dir",walkingDir[0],walkingDir[1], " dff angle: ",dAngle)

    def loadOtherData(self):
        objName = self.getObjName()
        posf = open(self.folderName + "/" + objName + otherPosFile, 'r')
        velf = open(self.folderName + "/" + objName + otherVelFile, 'r')
        rotf = open(self.folderName + "/" + objName + otherRotationFile, 'r')

        while True:
            posLine = posf.readline()
            velLine = velf.readline()
            rotLine = rotf.readline()
            if not posLine: break

            self.otherPosData.append(self.getLineToVector3(posLine) / 100 - self.posOffset)
            self.otherVelData.append(self.getLineToVector3(velLine) / 100)
            self.otherRotData.append(self.getLineToVector3(rotLine) * (math.pi / 180))

        posf.close()
        velf.close()
        rotf.close()

    def loadPosData(self):
        objName = self.getObjName()
        f = open(self.folderName + "/" + objName + posFile, 'r')
        # 첫번째 pos 의 x,z값을 0으로 해줌.
        line = f.readline()
        self.posOffset = self.getLineToVector3(line) / 100
        self.posData.append(Vector3(0, 0, 0))
        while True:
            line = f.readline()
            if not line: break
            line = line.replace("(", "").replace(")", "").replace(",", "")
            line = line.split()
            data = (Vector3(float(line[0]), float(line[1]), float(line[2])) / 100) - self.posOffset
            if(data.y > self.maxHeight):
                self.maxHeight = data.y
            if self.onLPF:
                data = self.filterWeight * self.posData[len(self.posData)-1] + (1-self.filterWeight) * data
            self.posData.append(data)

        f.close()

    def loadVelData(self):
        objName = self.getObjName()
        f = open(self.folderName + "/" + objName + velFile, 'r')

        line = f.readline()

        data = Vector3(0, 0, 0)
        self.velData.append(data)
        while True:
            line = f.readline()
            if not line: break

            line = line.replace("(", "").replace(")", "").replace(",", "")
            line = line.split()
            data = Vector3(float(line[0]), float(line[1]), float(line[2])) / 100

            if self.onLPF:
                data = self.filterWeight * self.velData[len(self.velData)-1] + (1-self.filterWeight) * data

            self.velData.append(data)
        f.close()

    def loadRotData(self):
        objName = self.getObjName()
        f = open(self.folderName + "/" + objName + rotationFile, 'r')
        while True:
            line = f.readline()
            if not line: break

            line = line.replace("(", "").replace(")", "").replace(",", "")
            line = line.split()
            data = Vector3(float(line[0]), float(line[1]), float(line[2]))
            data *= (math.pi / 180)
            #low pass filter
            if self.onLPF:
                if(len(self.rotData) >  0):
                    data = self.filterWeight *  self.rotData[len(self.rotData)-1]  + (1-self.filterWeight) * data

            self.rotData.append(data)
        f.close()

    def checkTotalTime(self):
        objName = self.getObjName()
        f = open(self.folderName + "/" + objName + timeFile, 'r')
        while True:
            line = f.readline()
            if not line: break
            self.stepTime += float(line)
            self.totalFrame += 1
        f.close()

    def GetStepDistance(self):
        x,y = self.lastXZ()
        return math.sqrt(x**2 + y**2)



# 너무 짧거나 너무 긴 데이터를 버림
def isValidData(stepData, seq):
    isValid = True
    if stepData.stepTime < 0.3 or stepData.stepTime > 1.5 or stepData.totalFrame < 12:
        isValid = False
    return isValid


def ReadFootData(steps, path, minIndex, maxIndex, walkingType, seq):
    for i in range(minIndex, maxIndex):
        lStep = StepData(path + str(i + 1), walkingType, True)
        rStep = StepData(path + str(i + 1), walkingType, False)
        if isValidData(lStep, seq):
            steps.append(lStep)
        if isValidData(rStep, seq):
            steps.append(rStep)


def drawData():
    steps = []
    dataNumbers = []
    folderNames = os.listdir(dataFolder)

    fi = 0
    for folder in folderNames:
        step = []
        curPath = dataFolder + folder + "/"
        print(curPath)
        maxStepCount = len(os.listdir(curPath))
        ReadFootData(step, curPath, 0, maxStepCount, fi, 10)
        steps.append(step.copy())
        fi += 1

    for j in range(0, 5):
        for s1, s2 in zip(steps[j], steps[j + 5]):
            i = 0
            yPos1 = []
            timeData1 = []
            yPos2 = []
            timeData2 = []
            for p1, p2 in zip(s1.posData, s2.posData):
                yPos1.append(p1.y)
                yPos2.append(p2.y)
                timeData1.append(i * (s1.stepTime / s1.totalFrame))
                timeData2.append(i * (s2.stepTime / s2.totalFrame))
                i += 1
                plt.plot(timeData1, yPos1, 'r');
                plt.plot(timeData2, yPos2, 'g');
                plt.legend(('plane', 'stair'))
        plt.xlabel('time')
        plt.ylabel('y')
        plt.show()


def draw3d():
    steps = []
    dataNumbers = []
    folderNames = os.listdir(dataFolder)

    fi = 0
    for folder in folderNames:
        step = []
        curPath = dataFolder + folder + "/"
        print(curPath)
        maxStepCount = len(os.listdir(curPath))
        ReadFootData(step, curPath, 0, maxStepCount, fi, 10)
        steps.append(step.copy())
        fi += 1

    for j in range(0, 5):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        for step, step1 in zip(steps[j], steps[j + 1]):
            xPos1 = []
            yPos1 = []
            zPos1 = []
            xPos2 = []
            yPos2 = []
            zPos2 = []
            for inputs1, inputs2 in zip(step.makeInputData(0, step.totalFrame),
                                        step1.makeInputData(0, step1.totalFrame)):
                # for inputs1,inputs2 in zip(step.posData,step1.posData):
                xPos1.append(inputs1[0])
                yPos1.append(inputs1[1])
                zPos1.append(inputs1[2])
                xPos2.append(inputs2[0])
                yPos2.append(inputs2[1])
                zPos2.append(inputs2[2])
                # xPos1.append(inputs1.x)
                # yPos1.append(inputs1.y)
                # zPos1.append(inputs1.z)
                # xPos2.append(inputs2.x)
                # yPos2.append(inputs2.y)
                # zPos2.append(inputs2.z)
            ax.plot(xPos1.copy(), yPos1.copy(), zPos1.copy(), 'r')
            ax.plot(xPos2.copy(), yPos2.copy(), zPos2.copy(), 'g')
        ax.set_xlabel('axis_x')
        ax.set_ylabel('axis_y')
        ax.set_zlabel('axis_z')
        # plt.title(step.folderName)
        # if(step.isLeft): lr = 'L'
        # else : lr = 'R'
        plt.show()
        # plt.savefig('./img/'+step.folderName.replace('/','_')+lr +'.png')
        # plt.cla()
        # plt.close(fig)


# LSTM 발걸음 분류
def MakeData(seq=10):
    steps = []
    dataNumbers = []
    folderNames = os.listdir(dataFolder)

    fi = 0
    for folder in folderNames:
        curPath = dataFolder + folder + "/"
        print(curPath)
        maxStepCount = len(os.listdir(curPath))
        ReadFootData(steps, curPath, 0, maxStepCount, fi, seq)
        fi += 1
    # drawData(steps)


'''
    input_batch = []
    target_batch = []
    random.shuffle(steps)

    #print("min,max,steps",StepData.MIN,StepData.MAX,len(steps))
    for step in steps:
        #for s in range(0,step.totalFrame - seq +1):
        input_batch.append(step.makeInputData(s,seq).copy())
        target_batch.append(step.walkingType)

    total = list(zip(input_batch,target_batch))
    random.shuffle(total)
    input_batch, target_batch = zip(*total)

    trainCount = (int)(len(input_batch) * 0.1)
    print(trainCount)

    test_batch = input_batch[:trainCount]
    answer_batch= target_batch[:trainCount]

    input_batch = torch.tensor(input_batch[trainCount:], dtype=torch.float32, requires_grad=True).to(Device)
    target_batch = torch.tensor(target_batch[trainCount:], dtype=torch.int64).to(Device)

    test_batch = torch.tensor(test_batch, dtype=torch.float32, requires_grad=True).to(Device)
    answer_batch = torch.tensor(answer_batch, dtype=torch.int64).to(Device)

    return input_batch,target_batch ,test_batch , answer_batch

def getSampleData(seq = 8):
    steps = []
    dataNumbers = []
    folderNames = os.listdir(dataFolder)
    input_batch = []
    target_batch = []

    fi = 0
    for folder in folderNames:
        curPath = dataFolder + folder + "/"
        maxStepCount = len(os.listdir(curPath))
        ReadFootData(steps,curPath,0,1,fi,seq)
        fi += 1
        break

    for step in steps:
        input_batch.append(step.makeInputData(0,seq).copy())
        target_batch.append(step.walkingType)

    input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True).to(Device)    
    return input_batch
'''


# drawData()
# draw3d()


# LSTM
def MakeData2(seq=10):
    steps = []
    folderNames = os.listdir(dataFolder)

    fi = 0
    for folder in folderNames:
        curPath = dataFolder + folder + "/"
        print(curPath)
        maxStepCount = len(os.listdir(curPath))
        ReadFootData(steps, curPath, 0, maxStepCount, fi, seq)
        fi += 1
        if fi > 9: break

    input_batch = []
    target_batch = []
    pos_data = []
    random.shuffle(steps)
    trainCount = (int)(len(steps) * 0.1)

    for step in steps:
        for s in range(seq, step.totalFrame - 1):
            input_batch.append(step.makeInputData1(seq, s, False).copy())
            target_batch.append(step.lastXZ(False))

    total = list(zip(input_batch, target_batch))
    random.shuffle(total)
    input_batch, target_batch = zip(*total)

    test_batch = []
    answer_batch = []
    for step in steps[:trainCount]:
        num = seq
        for s in range(seq, step.totalFrame - 1):
            test_batch.append(step.makeInputData1(seq, s, False).copy())
            answer_batch.append(step.lastXZ(False))
            pos_data.append((step.posData.copy(), num))
            num += 1

    total = list(zip(input_batch, target_batch))
    random.shuffle(total)
    input_batch, target_batch = zip(*total)

    # input_batch = torch.tensor(input_batch[trainCount:], dtype=torch.float32, requires_grad=True).to(Device)
    # target_batch = torch.tensor(target_batch[trainCount:], dtype=torch.float32).to(Device)
    input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True).to(Device)
    target_batch = torch.tensor(target_batch, dtype=torch.float32).to(Device)

    test_batch = torch.tensor(test_batch, dtype=torch.float32, requires_grad=False).to(Device)
    answer_batch = torch.tensor(answer_batch, dtype=torch.float32).to(Device)

    return input_batch, target_batch, test_batch, answer_batch, pos_data


# MLP 중간값 샘프링
def MakeData3(seq=10):
    steps = []
    folderNames = os.listdir(dataFolder)

    fi = 0
    for folder in folderNames:
        curPath = dataFolder + folder + "/"
        print(curPath)
        maxStepCount = len(os.listdir(curPath))
        ReadFootData(steps, curPath, 0, maxStepCount, fi, seq)

    input_batch = []
    target_batch = []
    pos_data = []
    random.shuffle(steps)

    trainCount = (int)(len(steps) * 0.1)

    for step in steps[trainCount:]:
        for s in range(seq, step.totalFrame - 1):
            input_batch.append(step.makeInputData3(10, s).copy())
            target_batch.append(step.lastXZ(False))

    test_batch = []
    answer_batch = []
    for step in steps[:trainCount]:
        num = seq
        for s in range(seq, step.totalFrame - 1):
            test_batch.append(step.makeInputData3(10, s).copy())
            answer_batch.append(step.lastXZ(False))
            pos_data.append((step.posData.copy(), num))
            num += 1

    total = list(zip(input_batch, target_batch))
    random.shuffle(total)
    input_batch, target_batch = zip(*total)

    # input_batch = torch.tensor(input_batch[trainCount:], dtype=torch.float32, requires_grad=True).to(Device)
    # target_batch = torch.tensor(target_batch[trainCount:], dtype=torch.float32).to(Device)
    input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True).to(Device)
    target_batch = torch.tensor(target_batch, dtype=torch.float32).to(Device)

    test_batch = torch.tensor(test_batch, dtype=torch.float32, requires_grad=False).to(Device)
    answer_batch = torch.tensor(answer_batch, dtype=torch.float32).to(Device)

    return input_batch, target_batch, test_batch, answer_batch, pos_data


# MLP , 시작점 끝점
def MakeData4(seq=10):
    steps = []
    folderNames = os.listdir(dataFolder)

    fi = 0
    for folder in folderNames:
        '''
        fi += 1
        if fi > 2:
            break
                   '''
        curPath = dataFolder + folder + "/"
        print(curPath)
        maxStepCount = len(os.listdir(curPath))
        ReadFootData(steps, curPath, 0, maxStepCount, fi, seq)

    print(len(steps))

    input_batch = []
    target_batch = []
    pos_data = []
    random.shuffle(steps)

    trainCount = (int)(len(steps) * 0.1)

    for step in steps[trainCount:]:
        for s in range(seq, step.totalFrame - 1):
            input_batch.append(step.makeInputData2(s).copy())
            target_batch.append(step.lastXZ(False))

    test_batch = []
    answer_batch = []
    for step in steps[:trainCount]:
        num = seq
        for s in range(seq, step.totalFrame - 1):
            test_batch.append(step.makeInputData2(s).copy())
            answer_batch.append(step.lastXZ(False))
            pos_data.append((step.posData.copy(), num))
            num += 1

    total = list(zip(input_batch, target_batch))
    random.shuffle(total)
    input_batch, target_batch = zip(*total)
    trainInput = np.array(input_batch)
    trainTarget = np.array(target_batch)

    testInput = np.array(test_batch)
    testTarget = np.array(answer_batch)

    #np.savez("footTrain",input = trainInput,target = trainTarget)
    #np.savez("footTest", input = testInput, target = testTarget)


    # input_batch = torch.tensor(input_batch[trainCount:], dtype=torch.float32, requires_grad=True).to(Device)
    # target_batch = torch.tensor(target_batch[trainCount:], dtype=torch.float32).to(Device)
    input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True).to(Device)
    target_batch = torch.tensor(target_batch, dtype=torch.float32).to(Device)

    test_batch = torch.tensor(test_batch, dtype=torch.float32, requires_grad=False).to(Device)
    answer_batch = torch.tensor(answer_batch, dtype=torch.float32).to(Device)

    return input_batch, target_batch, test_batch, answer_batch, pos_data


# train_data
# 0~2 : pos
# 3~5 : rot
# 6~8 : vel
def MakeDataWithNPZ(seq=10):
    npzFile = np.load('train1.npz')
    trainData = npzFile['train_data']
    steps = []
    for data in trainData:
        step = StepData(loadWithNPZ=True)
        step.loadDataWithNPZ(data)
        steps.append(step)
    npzFile.close()

    input_batch = []
    target_batch = []
    pos_data = []
    random.shuffle(steps)

    trainCount = (int)(len(steps) * 0.1)

    input_batch = []
    target_batch = []
    pos_data = []
    random.shuffle(steps)

    trainCount = (int)(len(steps) * 0.1)

    for step in steps[trainCount:]:
        for s in range(seq, step.totalFrame - 1):
            input_batch.append(step.makeInputData2(s).copy())
            target_batch.append(step.lastXZ(False))

    test_batch = []
    answer_batch = []
    for step in steps[:trainCount]:
        num = seq
        for s in range(seq, step.totalFrame - 1):
            test_batch.append(step.makeInputData2(s).copy())
            answer_batch.append(step.lastXZ(False))
            pos_data.append((step.posData.copy(), num))
            num += 1

    total = list(zip(input_batch, target_batch))
    random.shuffle(total)
    input_batch, target_batch = zip(*total)

    # input_batch = torch.tensor(input_batch[trainCount:], dtype=torch.float32, requires_grad=True).to(Device)
    # target_batch = torch.tensor(target_batch[trainCount:], dtype=torch.float32).to(Device)
    input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True).to(Device)
    target_batch = torch.tensor(target_batch, dtype=torch.float32).to(Device)

    test_batch = torch.tensor(test_batch, dtype=torch.float32, requires_grad=False).to(Device)
    answer_batch = torch.tensor(answer_batch, dtype=torch.float32).to(Device)

    return input_batch, target_batch, test_batch, answer_batch, pos_data



def CalStepsMeanHeightAndWidth():
    folderNames = os.listdir(dataFolder)
    for folder in folderNames:
        steps = []
        curPath = dataFolder + folder + "/"
        #print(curPath)
        maxStepCount = len(os.listdir(curPath))
        ReadFootData(steps, curPath, 0, maxStepCount, 0, 10)

        minMaxHeight = 20000
        maxMaxHeight = 0
        meanHeight = 0
        meanDistance = 0
        for step in steps:
            if step.maxHeight < minMaxHeight:
                minMaxHeight = step.maxHeight
            if step.maxHeight > maxMaxHeight:
                maxMaxHeight = step.maxHeight
            meanHeight += step.maxHeight
            meanDistance += step.GetStepDistance()
        meanHeight /= len(steps)
        meanDistance /= len(steps)
        print(folder,"\n minMaxHeight : ", minMaxHeight,
                    "\n maxMaxHeight : ",
              maxMaxHeight,"\n mean Height : ", meanHeight,"\n mean Distance : ", meanDistance, "\n step size : ",
              len(steps))

CalStepsMeanHeightAndWidth()

#MakeData4()