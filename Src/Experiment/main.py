import cv2
import threading
import time
from playsound import playsound
from datetime import datetime
import numpy as np

def runExperiment(numTrials, trialLength):
    data = []
    experimentStartTime = datetime.now()
    eyesClosed = cv2.imread(".\\Resources\\eyesClosed.jpg")
    eyesOpen = cv2.imread(".\\Resources\\eyesOpen.jpg")
    for i in range(numTrials):
        trialStartTime = datetime.now()
        dp = {}
        dp["trialStartTime"] = trialStartTime.strftime("%d-%b-%Y %H:%M:%S.%f")
        if i % 2 == 0:
            dp["trialType"] = "eyesOpen"
            t = threading.Thread(target=playsound, args=[".\\Resources\\doorbellTone.wav"]).start()
            cv2.imshow("window", eyesOpen)
            cv2.waitKey(trialLength * 1000)
            cv2.destroyAllWindows()
        else:
            dp["trialType"] = "eyesClosed"
            cv2.imshow("window", eyesClosed)
            cv2.waitKey(trialLength * 1000)
            cv2.destroyAllWindows()
        trialEndTime = datetime.now()
        dp["trialEndTime"] = trialEndTime.strftime("%d-%b-%Y %H:%M:%S.%f")
        data.append(dp)
        
    for i in range(len(data)):
        print(data[i])


if __name__ == "__main__":
    threading.Thread(target=runExperiment, args=[15, 5]).start()