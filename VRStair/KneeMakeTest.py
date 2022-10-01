
import GraphMaker as g
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
folder = "C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"




def reader(folderName):
    result = dict()
    #condition = ["stair1","stair2","stair1_60","stair2_60","stair1_85","stair2_85"]

    condition = ["stair1_50", "stair1_75", "stair1_100", "stair2_50", "stair2_75", "stair2_100"]
    for c in condition:
        file_list = os.listdir(folderName)
        stepFiles = []
        for i in range(0,10):
            for name in file_list:
                stepFiles.append(folderName + name + "/" + c + "/" + str(i)+ "/")
        print(c)
        result[c] = g.StepAnalyzer(stepFiles,False,c).GetResultList()