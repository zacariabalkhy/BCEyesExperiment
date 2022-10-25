import cv2
import threading
import time
from playsound import playsound
from datetime import datetime
import numpy as np
import json
from pylsl import StreamInlet, resolve_stream
#import multiprocessing as mp
from threading import Thread
from time import sleep
class Experiment:
    def __init__(self, numTrials, trialLength):
        self.numTrials = numTrials
        self.trialLength = trialLength
        self.dataStreamProcess = None
        self.experimentProcess = None
        self.EEGdata = []
        self.experimentMetaData = []  
        self.streamReady = False
        self.experimentComplete = False

    def beginExperimentAndCollectMetaData(self):
        while(not self.streamReady):
            sleep(1)
        experimentStartTime = datetime.now()
        eyesClosed = cv2.imread(".\\Resources\\eyesClosed.jpg")
        eyesOpen = cv2.imread(".\\Resources\\eyesOpen.jpg")
        for i in range(self.numTrials):
            trialStartTime = datetime.now()
            dp = {}
            dp["trialStartTime"] = trialStartTime.strftime("%d-%b-%Y %H:%M:%S.%f")
            if i % 2 == 0:
                dp["trialType"] = "eyesOpen"
                t = threading.Thread(target=playsound, args=[".\\Resources\\doorbellTone.wav"]).start()
                cv2.imshow("window", eyesOpen)
                cv2.waitKey(self.trialLength * 1000)
                cv2.destroyAllWindows()
            else:
                dp["trialType"] = "eyesClosed"
                cv2.imshow("window", eyesClosed)
                cv2.waitKey(self.trialLength * 1000)
                cv2.destroyAllWindows()
            trialEndTime = datetime.now()
            dp["trialEndTime"] = trialEndTime.strftime("%d-%b-%Y %H:%M:%S.%f")
            self.experimentMetaData.append(dp)
        self.experimentComplete = True

    def collectData(self):
        print("looking for an EEG stream...")
        #streams = resolve_stream('type', 'EEG')
        #if not streams or streams[0]:
        #    return
        #inlet = StreamInlet(streams[0])
        self.streamReady = True
        while not self.experimentComplete:
            # get a new sample (you can also omit the timestamp part if you're not
            # interested in it)
            now = datetime.now()
            #sample, timestamp = inlet.pull_sample()
            sample = ["0","0","0"]
            timestamp = now.strftime("%d-%b-%Y %H:%M:%S.%f")
            # add time correction to get system local time, and append timestamp to data
            #timestamp += inlet.time_correction()
            if sample and timestamp:
                sample.append(timestamp)
            self.EEGdata.append(sample)
    
    def runExperiment(self):
        self.dataStreamProcess = Thread(target=self.collectData)
        self.experimentProcess = Thread(target=self.beginExperimentAndCollectMetaData)
        self.dataStreamProcess.start()
        self.experimentProcess.start()
        self.experimentProcess.join()
        self.dataStreamProcess.join()
            

    
