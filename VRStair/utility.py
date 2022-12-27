
import math
import os
import numpy as np
from scipy.signal import savgol_filter
from define import *

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

    def ToString(self):
        return "("+str(self.x)+", "+str(self.y)+", "+str(self.z)+")"

RfootOffset = Vector3(0.08,0.1,0)
LfootOffset = Vector3(-0.08,0.1,0)
HeadOffset = Vector3(0,1.6,0)
defalutRotationL = Vector3(22,-44,-40) * math.pi/180
defalutRotationR = Vector3(10.7,53,140.7)* math.pi/180

'''
위치 리스트를 넣어주면, 속도 리스트로 만들어줌.
fixedDeltaTime = 0.011111 (define.py 참고) 
 input 
  - posData [] 
 output 
  - velData [] 
'''
def MakeVelData(posData,smoothON = False):
    velData = [0]
    if smoothON :
        if len(posData) > filterSize:
            posData = savgol_filter(posData, filterSize, 6)
        else:
            print("MakeVelData - warning : filterSize(51) 보다 position 길이가 작으서 필터링 못함.")

    for i in range(1,len(posData)):
        velData.append((posData[i] - posData[i-1])/fixedDeltaTime)
    return velData.copy()


def writeData(path,content):
    file = open(path, "w")
    file.write(content)
    file.close()


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

def loadData(flieName,firstZero = False,addedVector = Vector3(0,0,0)):
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
        pX.append(float(line[0]) - firstX + addedVector.x)
        pY.append(float(line[1]) - firstY + addedVector.y)
        pZ.append(float(line[2]) - firstZ + addedVector.z)
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

def FloatArrayToString(info,start=0,end=-1):
    dataTxt= ""
    if end == -1:
        end = len(info)
    ran = range(start,end)
    for i in ran:
        dataTxt += str(info[i]) + "\n"
    return dataTxt


def Vector3ArrayToString(info,start = 0,end = -1):
    dataTxt = ""
    if end == -1:
        end = len(info[0])
    ran = range(start,end)
    for i in ran:
        dataTxt += Vector3(info[0][i],info[1][i],info[2][i]).ToString() + "\n"
    return dataTxt

def FindMaxIndex(arr):
    m = max(arr)
    return list(arr).index(m)



'''
ex) 
input : stair1_50 => output : 0.125, 50
input : stair2_75 -> output : 0.25, 75
'''
def GetHeightAndBPM(case : str):
    s = case.split("_")
    height = 0.125
    bpm = s[1]
    if "2" in s[0]:
        height = 0.25
    return height,bpm


'''
폴더 형식이 다음과 같이 저장 되어있을 때 
 - /Ours/김미송/stair1_100/0/
 -/method/이름/height_bpm/count/
 input --> str: "D:/Desktop/VRStairs_python/VRStair/foot_dataset/ex3/"
->['D:/Desktop/VRStairs_python/VRStair/foot_dataset/ex3/Ours/김미송/stair1_100/0/', 'D:/Desktop/VRStairs_python/VRStair/foot_dataset/ex3/Ours/김미송/stair1_100/1/'...'']
이런 식으로 폴더 리스트 만들어 줌.
'''
def GetFolderList(folderName,methodList = ["Ours", "Seo", "Nagao"],count = 5):
    if not folderName.endswith("/"):
        folderName += "/"
    folderList = []
    for m in methodList:
        names = os.listdir(folderName + m + "/")
        for n in names:
            stair_cases = os.listdir(folderName + m + "/" + n + "/")
            for s in stair_cases:
                for i in range(count):
                    folderList.append(folderName + m + "/" + n + "/" + s + "/" + str(i) + "/")
    return folderList