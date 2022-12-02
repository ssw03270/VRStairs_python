
import math
import numpy as np

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

