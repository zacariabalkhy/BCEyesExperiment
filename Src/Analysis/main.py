from asyncore import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

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
        'AnalogChannel0', 'AnalogChannel1', 'AnalogChannel2', 'Other'], axis=1, inplace=True)
            dataframes.append(df)
    return dataframes

def readTrialData():
    trialData = []
    trialdataFileName = "../../Data/AlphaWaveExperimentData/alphaWaveTrialData.json"
    with open(trialdataFileName) as file:
        trialData = json.load(file)
    for i in range(len(trialData)):
        print (trialData[i])

def createTimeSlices(df, startIndex, length):
    return df[startIndex:startIndex+length][:]


#def extractSingleTrial(trialStartTime, trialEndTime):

def plotsample(df):

    freqAmplitudes = np.abs(np.fft.fft(pd.to_numeric(df[:]["EXGChannel0"])))
    freqs = np.fft.fftfreq(n=pd.to_numeric(df[:]["EXGChannel0"]).size, d=1/sampleRate)

    plt.figure()

    plt.subplot(121)
    plt.stem(freqs[1:], freqAmplitudes[1:], 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 250)
    #plt.ylim(0,100000)

    plt.subplot(122)
    plt.plot(pd.to_numeric(df[:]['Timestamp']), pd.to_numeric(df[:]["EXGChannel0"]), "b")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    readTrialData()
    
    dataframes = readDataIntoDF()
    for dataframe in dataframes:
        print(dataframe.columns)
        print(dataframe.head)
        plotsample(createTimeSlices(dataframe, 4000, 10000))