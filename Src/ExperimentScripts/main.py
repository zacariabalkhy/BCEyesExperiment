import cv2
import threading
import time
from playsound import playsound
from datetime import datetime
import numpy as np
import json

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
        
    file = open("TrialData.json", "w")
    file.write(json.dumps(data))
    file.close()
    for i in range(len(data)):
        print(data[i])
    return data
    


if __name__ == "__main__":
    threading.Thread(target=runExperiment, args=[60, 6]).start()