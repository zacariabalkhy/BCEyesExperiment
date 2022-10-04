from asyncore import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import time
import math

#Sample Rate = 250 Hz
sampleRate = 250
def readDataIntoDF():
    dataframes = []
    datadir = "../../Data/AlphaWaveExperimentData/"
    dataFiles = os.listdir(datadir)

    for filePath in dataFiles:
        if (".txt" in filePath):
            data = []
            file = open(datadir + filePath, 'r')
            lines = file.readlines()[4:]
            for line in lines:
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                datapoints = line.split(",")
                data.append(datapoints)
            df = pd.DataFrame(data[1:], columns=data[0])
            df.drop(['AccelChannel0', 'AccelChannel1', 'AccelChannel2', 'Other',
        'Other', 'Other', 'Other', 'Other', 'Other', 'Other',
        'AnalogChannel0', 'AnalogChannel1', 'AnalogChannel2', 'Other', 'Timestamp(Formatted)'], axis=1, inplace=True)
            df = df.apply(pd.to_numeric, errors='coerce')
            dataframes.append(df)
    return dataframes

def readTrialData():
    trialData = []
    trialdataFileName = "../../Data/AlphaWaveExperimentData/alphaWaveTrialData.json"
    with open(trialdataFileName) as file:
        trialData = json.load(file)
    return trialData

def splitRawDataIntoTrials(df, trialData):
    trialSplitData = []
    rawDataStartTime = df.loc[0]["Timestamp"]

    for i in range(len(trialData)):
        trialData[i]["trialNumber"] = i
        format = "%d-%b-%Y %H:%M:%S.%f"

        startTimeStr = trialData[i]["trialStartTime"]
        startTime = datetime.strptime(startTimeStr, format)
        unixStartTime = time.mktime(startTime.timetuple())

        endTimeStr = trialData[i]["trialEndTime"]
        endTime = datetime.strptime(endTimeStr, format)
        unixEndTime = time.mktime(endTime.timetuple())

        timeStartDiff = unixStartTime - rawDataStartTime
        calcSampleStartIndex = math.ceil(timeStartDiff * sampleRate)
        
        timeEndDiff = unixEndTime - rawDataStartTime
        calcSampleEndIndex = math.ceil(timeEndDiff * sampleRate)
        
        trialDf = df.iloc[calcSampleStartIndex:calcSampleEndIndex, :]
        trialData[i]["data"] = trialDf
        trialSplitData.append(trialData[i])

    return trialSplitData

def createTimeSlices(df, startIndex, length):
    return df[startIndex:startIndex+length][:]


#def extractSingleTrial(trialStartTime, trialEndTime):

def plotsample(df):

    freqAmplitudes = np.abs(np.fft.fft(df[:]["EXGChannel7"]))
    freqs = np.fft.fftfreq(n=df[:]["EXGChannel7"].size, d=1/sampleRate)

    plt.figure()

    plt.subplot(121)
    plt.stem(freqs[1:], freqAmplitudes[1:], 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 50)
    plt.ylim(0,10000)

    plt.subplot(122)
    plt.plot(pd.to_numeric(df[:]['Timestamp']), pd.to_numeric(df[:]["EXGChannel0"]), "b")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trialData = readTrialData()
    
    dataframes = readDataIntoDF()
    for dataframe in dataframes:
        print(dataframe.columns)
        print(dataframe.head)
        trials = splitRawDataIntoTrials(dataframe, trialData)
        for i in range(len(trials)):
            print(trials[i]["trialType"])
            plotsample(trials[i]["data"])