import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt

'''
t1 : 원래꺼 : 48.524487305319155 7.193363368089595 11.540539478723405
t2 : 원래꺼의 그림자
t3 : 10 ,input data 24 : 48.524487305319155 8.835406196680436 10.973120765957448
t4 : 6 , input data 24
t5 : 8 , input data 24
t6 : 8, input data 24
t7 : 10, input data 19 
t8 : 10, input data 24 - 82 : 48.524487305319155 7.6535967521271235 11.72353889361702
t9 : 10, input data 24 - 81
t19 :10, input data 24 - 86 : 48.524487305319155 11.46452932712856 12.181774170212766
lpf2 : low pass filter 쓴거 : 54.66941827712417 9.635120410208136 10.808038653594775 0.525732899360813 0.17624333153440283

errorData_new_xyz_3 : original(19) : 54.66941827712417 7.087343468996433 10.929636849673205 0.6035105509415302 0.12964000152827773

all lpf1 : 위치, 속도, 회전 모두 lowpass filter 쓴거. 테스트 데이터 아니고 내가 따로 걸은거

errorData_xy_19_lpf: lpf.  x,y 좌표만 쓰고 이전 프레임 정보는 포지션 값만 넣거(input size 19) 
54.66941827712417 7.77033664677128 11.018904092156863 0.5653386589788917 0.14213315033613036 

errorData_lpf_xyz_preOnlyPos3: 54.66941827712417 7.651289196379545 11.185873705882354 0.543717118463312 0.13995556268029916


평균 걸음거리 , 마지막 오차 거리, 평균 오차 거리, 첫 오차율, 마지막 오차율.

0617
input27 - 54.66941827712417 7.943811009999662 10.852078104575165 0.5449927972980299 0.14530630214010642
input27-iter2000 - 54.66941827712417 10.079779674407089 10.789062437908495 0.4920741740061944 0.1843769330654258
input21-1 - 54.66941827712417 10.97396351275096 11.132879222222224 0.5676473757263741 0.20073313122014502
input21-3 - 54.66941827712417 14.190315089254184 11.314598790849672 0.5988813709694445 0.2595658694834141
input27-iter2000 - 54.66941827712417 10.079779674407089 10.789062437908495 0.4920741740061944 0.1843769330654258
input27-iter2000-2 - 54.66941827712417 11.085913596381289 10.721295960784312 0.5292956326597787 0.20278089553076645
relu - 54.66941827712417 16.586193736020263 13.384800241830064 0.6337142695378339 0.3033907120054464
relu - 54.66941827712417 13.655992893190977 12.54109800653595 0.6183352421625776 0.2497921749225047


totalPaths = ["errorData/data1/계단/forward1/",
         "errorData/data1/계단/forward2/",
         "errorData/data1/계단/side/",
         "errorData/data1/계단/back1/",
         "errorData/data1/계단/back2/",
         "errorData/data1/경사/forward1/",
         "errorData/data1/경사/forward2/",
         "errorData/data1/경사/side/",
         "errorData/data1/경사/back1/",
         "errorData/data1/경사/back2/",
         "errorData/data1/울퉁불퉁2/"]
'''

#path = "errorData/data1/울퉁불퉁2/"
#totalPaths = ["errorData/data1/울퉁불퉁2/"]


totalPaths = [

         "errorData/data1/울퉁불퉁2/"]


class ErrorData:
    def __init__(self,_framePerError,_MSE,_tDistance,_heightFromGround,_heightFromGroundError):
        self.framePerError = _framePerError
        self.MSE = _MSE
        self.totalMovingDistance = _tDistance
        self.heightFromGround =_heightFromGround
        self.heightFromGroundError = abs(_heightFromGroundError)


errorDataList = []
def makeErrorList(paths):
    for path in paths:
        file_list = os.listdir(path)
        for file in file_list:
            f = open(path + file, 'r')
            data = f.read()
            dataList = data.split("MSE ")
            d1 = dataList[0].split()
            d2 = dataList[1].split()
            #print(d1)
            #print(d2)
            errorList = []
            for errorT in d1:
                errorList.append(float(errorT))
            if(len(errorList) > 15 and len(errorList) < 150):
                errorDataList.append(ErrorData(errorList,float(d2[0]),float(d2[2]),float(d2[4]),float(d2[6]) ))
makeErrorList(totalPaths)
print(len(errorDataList))

'''
for data in errorDataList:
    f, axes = plt.subplots(1, 2)
    n = 0
    for e in data.framePerError:
        axes[0].bar(n,e)
        #print(data.MSE, data.totalMovingDistance)
        axes[1].bar(n,e/data.totalMovingDistance)
        n += 1
    plt.show()
'''

sampleSize = 100
errorSampleList = [0] * sampleSize
meanTotalMoving = 0
meanMSE = 0
meanHeightFromGround = 0
meanHeightFromGroundError = 0
heightErrorList = []
minHeight = 100
maxHeight = 0

for data in errorDataList:
    for i in range(0,sampleSize):
        errorSampleList[i] += (data.framePerError[int((len(data.framePerError)-1) * (i+1)/sampleSize)] / data.totalMovingDistance)
        #print(data.MSE, data.totalMovingDistance)
    meanTotalMoving += data.totalMovingDistance
    meanMSE += data.MSE
    meanHeightFromGround += data.heightFromGround
    meanHeightFromGroundError += data.heightFromGroundError
    heightErrorList.append(data.heightFromGroundError)
    if minHeight > data.heightFromGroundError:
        minHeight = data.heightFromGroundError
    if maxHeight < data.heightFromGroundError:
        maxHeight = data.heightFromGroundError

xAxis = []
errorCount = [0] * 1001
for HE in heightErrorList:
    curV = (HE-minHeight)/(maxHeight-minHeight)
    errorCount[int(curV*1000)] += 1


for i in range(0,1001):
    #plt.bar(i, errorCount[i], color = 'y')
    xAxis.append((i/1000.0) * (maxHeight-minHeight)+minHeight)

plt.plot(xAxis,errorCount)
plt.show()
print(minHeight, maxHeight)
'''





dataSize = len(errorDataList)
meanTotalMoving /= dataSize
meanHeightFromGround /= dataSize
meanHeightFromGroundError /= dataSize

print(meanTotalMoving * 100, meanTotalMoving * errorSampleList[sampleSize-1]/dataSize * 100 , meanMSE/dataSize * 100 ,errorSampleList[0]/dataSize ,errorSampleList[sampleSize-1]/dataSize )
print(meanHeightFromGround, meanHeightFromGroundError)


n = 0
x_values = []
y_values = []

for e in errorSampleList:
    x_values.append(n)
    y_values.append(e/dataSize)
    n += 1

plt.plot(x_values,y_values)
plt.show()

'''