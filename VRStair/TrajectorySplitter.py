import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from define import *



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




'''
자른 step의 궤적이 유효한지 검사. 
잘못 잘린건지 검사하는 함수임.
'''
def isValidStep(data,Th):
    if max(data) - min(data) < Th:
        return False
    if data[0] - data[-1] > Th * 0.5:
        return False
    else:
        return True

def FindStartPoint(posData):
    velData = MakeVelData(posData)
    startIndex = 0
    for i in range(len(posData)-3):
        if (velData[i] > 0.02 and velData[i+1] > 0.02 and velData[i+2] > 0.02):
            startIndex = i;
            break;
    return startIndex


'''
높이 궤적이 주어지면, 궤적이 올라가는 시작점과 끝 부분을 찾아줌.
input <- (posdata [] , Th : float, interval : int , NotRemove : bool , isDebug : bool )
 - posData : 궤적 데이터 
 - Th : 변곡점을 찾을 속도 Th
 - interval : 시작점과 끝점간의 최소 거리
 - NotRemove : 찾아진 점을 정제하지 않고 전부 return.
 - isDebug : 궤적과 찾아진 시작점과 끝점을 그래프로 그려줌.
output -> [(s1,e2),,,,(s_n,e_n)] 
'''
def FindPoints(posData,Th = 0.05,interval = 25,NotRemove = False ,isDebug = False):
    f,axes = plt.subplots(3,1)
    posData = savgol_filter(posData,51,6)
    velData = MakeVelData(posData)
    aData = MakeVelData(velData,True)
    i = 0
    NextCoolTime = interval
    curCool = 0
    pointList = []
    aTh = max(aData)/5
    isStartPoint = True
    for v in velData:
        if curCool < 0:
            if isStartPoint:
                if (v > Th and abs(aData[i]) > aTh):
                    isStartPoint = not isStartPoint
                    curCool = NextCoolTime
                    pointList.append(i)
            else:
                if (abs(v) < Th  and abs(aData[i]) < aTh):
                    isStartPoint = not isStartPoint
                    curCool = NextCoolTime
                    pointList.append(i)
        else:
            curCool -= 1
        i += 1
    if not isStartPoint:
        pointList.append(i-1)
    resultList = []
    validTh = max(posData) * 0.5
    for j in range(len(pointList) - 1):
        start = pointList[j];
        end = pointList[j + 1]
        if NotRemove :
            resultList.append((start, end))
        else:
            if isValidStep(posData[start:end], validTh):
                sIndex = start + FindStartPoint(posData[start:end])
                resultList.append((sIndex, end))

    if isDebug:
        print(resultList)
        axes[0].plot(posData)
        axes[1].plot(velData)
        axes[2].plot(aData)
        i = 0
        for s,e in resultList:
            i += 1
            axes[0].scatter(s,posData[s],color = "r")
            axes[0].text(s,posData[s],i,color ="r")
            axes[0].scatter(e,posData[e],color = "b")
            axes[0].text(e, posData[e]*1.1, i,color="b")

        plt.show()

    return resultList

