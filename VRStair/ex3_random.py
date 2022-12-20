
'''
ex3에서 보여줄 궤적 랜덤 순서 만들기

후보1
한 블록당 : 2 (계단 높이 조건) * 2(속도 조건) * 6 (4가지 방법의 조합) = 24

후보2
한 블록당 : 2 (계단 높이 조건) * 2(속도 조건) * 12 (4가지 방법의 순열) = 48

후보3
 - 4개의 서로 다른 조건을 4개의 블록으로 : (1,75),(1,100),(2,75),(2,100)
 - 4개의 블록 counterbalanced random.
 - 한 블록 당 : 12 (4가지 방법의 순열) * n (반복)
    - 5번 반복 시 시간 : 12 * 5 * 10(s) = 600초  = 10분
    - 총 40분? 킹만한데?

'''

import numpy as np
import random
from itertools import permutations
import csv
import pandas as pd
import matplotlib.pyplot as plt


blockCondition = ["stair1_75","stair1_100","stair2_75","stair2_100"]
methodCondition = ["ours","real","nagao","seo"]
methodOrder = list(permutations(methodCondition, 2))
allOrder = list(permutations(blockCondition, 4))
repeatNumber = 5


def makeCSV():
    f = open('ex3_blockOrder.csv', 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(["block1","block2","block3","block4"])
    for order in allOrder:
        wr.writerow(order)
    f.close()

    '''각 참가자(24)에 따른 블록 순서(12),반복(5) 랜덤 case'''
    f = open('ex3_random.csv', 'w', newline='')
    wr = csv.writer(f)
    d = {"participant number":[],"current block":[],"block number":[],"case number":[],"first":[],"second":[],"chose":[]}
    wr.writerow(list(d.keys()))
    for i,c in enumerate(allOrder):
        print(i,c)
        for blockNum,curBlockName in enumerate(c):
            for n in range(repeatNumber):
                curMethodOrder = methodOrder.copy()
                np.random.shuffle(curMethodOrder)
                for o,curMethod in enumerate(curMethodOrder):
                    wr.writerow([i,curBlockName,blockNum,(n * len(curMethodOrder)) + o, curMethod[0],curMethod[1],""])
                    print([i,curBlockName,blockNum,(n * len(curMethodOrder)) + o, curMethod[0],curMethod[1]])
    f.close()

makeCSV()

