
import os

startDeltaLength = 6
fixedDeltaTime = 0.011111
#folder = "blendingData/0722-compare/"
#folder = "blendingData/realStair/"
#folder = "blendingData/"
rFootName = "RightFootController_"
lFootName = "LeftFootController_"
realName = "realTrajectory"
blendName = "blendedTrajectory"
filterSize = 51
ProjectFolder = os.getcwd()  #"C:/Users/user/Desktop/Unity/VRStair/footdata/"#"C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
ProjectFolder = ProjectFolder.replace("\\","/",10)