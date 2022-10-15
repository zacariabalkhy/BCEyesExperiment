from asyncore import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import time
import math
from sklearn.linear_model import LogisticRegression

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

def plotsample(df):
    freqAmplitudes = np.abs(np.fft.fft(df[:]["EXGChannel7"]))
    freqs = np.fft.fftfreq(n=df[:]["EXGChannel7"].size, d=1/sampleRate)
    
    freqs = freqs[:math.floor(len(freqs)/2)]
    freqAmplitutdes = freqAmplitudes[:math.floor(len(freqAmplitudes)/2)]

    plt.figure()

    plt.subplot(121)
    plt.stem(freqs, freqAmplitutdes, 'b')
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

def normalize(data):
    minimum = np.amin(data)
    maximum = np.amax(data)
    minMaxDiff = maximum - minimum
    for i in range(len(data)):
        data[i] = (data[i] - minimum) / minMaxDiff
    return data
        
def getFreqsAveragesForChannel(df, channel):
    # note trials are 5 seconds long sampled at 250Hz. Only taking the last 3 sec
    twoSecondIndex = sampleRate*2
    freqAmplitudes = np.abs(np.fft.fft(df[twoSecondIndex:][channel]))
    freqs = np.fft.fftfreq(n=df[twoSecondIndex:][channel].size, d=1/sampleRate)
    
    #cut out imaginary part of the fft
    freqs = freqs[:math.floor(len(freqs)/2)]
    freqAmplitudes = freqAmplitudes[:math.floor(len(freqAmplitudes)/2)]
    
    #subset freqs and amplitudes to the frequenencies we care about
    startIndex = np.where(freqs==6)[0][0]
    endIndex = np.where(freqs==14)[0][0] + 1
    
    freqs = freqs[startIndex:endIndex]
    freqAmplitudes = freqAmplitudes[startIndex:endIndex]

    #normalize frequencyAmplitudes
    freqAmplitudes = normalize(freqAmplitudes)

    #average frequency powers outside of alpha range and frequencies inside alpha range
    alphaAverage = 0
    aroundAlphaAverage = 0
    numFreqsAroundAlpha = 0
    numFreqsInAlpha = 0
    for i in range(len(freqAmplitudes)):
        if (freqs[i] >= 6 and freqs[i] < 8 or (freqs[i] > 12 and freqs[i] <= 14)):
            aroundAlphaAverage += freqAmplitudes[i]
            numFreqsAroundAlpha += 1
        if freqs[i] >= 8 and freqs[i] <= 12:
            alphaAverage += freqAmplitudes[i]
            numFreqsInAlpha += 1
    
    alphaAverage = alphaAverage / numFreqsInAlpha
    aroundAlphaAverage = aroundAlphaAverage / numFreqsAroundAlpha
    return alphaAverage, aroundAlphaAverage

def getAlphaAndNonalphaFreqAvgsPerChannel(df):
    channels = list(df.columns.values)
    alphaAvgPerChannel = []
    aroundAlphaAvgPerChannel = []
    for channel in list(channels):
        if "EXG" not in channel:
            channels.remove(channel)
        else:
            alphaAverage, aroundAlphaAverage = getFreqsAveragesForChannel(df, channel)
            alphaAvgPerChannel.append(alphaAverage)
            aroundAlphaAvgPerChannel.append(aroundAlphaAverage)
    return np.array(alphaAvgPerChannel), np.array(aroundAlphaAvgPerChannel)

def plotAveragePowerPerChannel(df, trialType):
    alphaAvgPerChannel, aroundAlphaAvgPerChannel = getAlphaAndNonalphaFreqAvgsPerChannel(df)
    
    channels = list(df.columns.values)
    X_axis = np.arange(len(channels))

    plt.figure(figsize=(12,6))
    plt.bar(X_axis - 0.2, alphaAvgPerChannel, 0.4, label = "alphaAvg")
    plt.bar(X_axis + 0.2, aroundAlphaAvgPerChannel, 0.4, label= "aroundAlphaAvg")
    plt.xticks(X_axis, channels)
    plt.xlabel("Channels")
    plt.ylabel("Avg power per channel")
    plt.title("Avg power for " + trialType + " trial")
    plt.legend()
    plt.show()

def fitModel(samples,labels):
    # we fit a logistic regression model to the averaged alpha band power per channel
    clf = LogisticRegression(random_state=0).fit(samples, labels)
    return clf

def predictSample(df, model):
    alphaAvgPerChannel, avgNonAlphaPerChannel = getAlphaAndNonalphaFreqAvgsPerChannel(df)
    prediction = model.predict(np.append(alphaAvgPerChannel, avgNonAlphaPerChannel).reshape(1, -1))
    return prediction

if __name__ == "__main__":
    trialData = readTrialData()
    dataframes = readDataIntoDF()

    for dataframe in dataframes:
        print(dataframe.columns)
        print(dataframe.head)
        trials = splitRawDataIntoTrials(dataframe, trialData)
        trainingData = []
        y_values = []
        for i in range(len(trials)):
            alphaAvgPerChannel, avgNonAlphaPerChannel = getAlphaAndNonalphaFreqAvgsPerChannel(trials[i]["data"])
            trainingData.append(np.append(alphaAvgPerChannel, avgNonAlphaPerChannel))
            y_values.append(trials[i]["trialType"])
        print(y_values)
        print(np.array(trainingData).shape)

        model = fitModel(trainingData, y_values)
        for i in range(len(trials)):
            print(trials[i]["trialType"])
            print(predictSample(trials[i]["data"], model))
            print ("=======================")
            #plotAveragePowerPerChannel(trials[i]["data"], trials[i]["trialType"])
        