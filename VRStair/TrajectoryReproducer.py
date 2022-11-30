import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import GraphMaker as g
import TrajectorySplitter as spl

folder = "D:/Desktop/unity/VRStair/footdata/ex3/our/임수빈/stair2_100/4/"

def test():
    r = g.RecordedData(folder, firstZero=False)
    spl.FindPoints(r.RFootData.realPosData[1],NotRemove=False,isDebug=True)


def make():
    r = g.RecordedData(folder,firstZero=False)
    f,axes = plt.subplots(2,1)

    #rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) ]
    yTrajectory = pd.read_csv("test.csv")
    r.DrawPosAndVelGraph(axes)
    y50 = yTrajectory[(yTrajectory["bpm"] == 75) & (yTrajectory["method"] == "Ours") & (yTrajectory["stairHeight"] == 0.25)]
    r.HeightTrajectorySynthesize(np.array(y50["y"]),axes)

    r.writeToTxt1("D:/Desktop/unity/VRStair/footdata/test/")
    r.DrawPosAndVelGraph(axes)
    plt.show()

test()