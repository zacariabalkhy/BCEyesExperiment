from os import times
from time import time
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression

class Classifier:

    def __init__(self, model):
        self.model = model
    
    def getFreqsAveragesForChannel(self, df, channel, sampleRate):
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
        freqAmplitudes = self.normalize(freqAmplitudes)

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

    def normalize(self, data):
        minimum = np.amin(data)
        maximum = np.amax(data)
        minMaxDiff = maximum - minimum
        for i in range(len(data)):
            data[i] = (data[i] - minimum) / minMaxDiff
        return data

    def getAlphaAndNonalphaFreqAvgsPerChannel(self, df, sampleRate):
        channels = list(df.columns.values)
        alphaAvgPerChannel = []
        aroundAlphaAvgPerChannel = []
        for channel in list(channels):
            if "EXG" not in channel:
                channels.remove(channel)
            else:
                alphaAverage, aroundAlphaAverage = self.getFreqsAveragesForChannel(df, channel, sampleRate)
                alphaAvgPerChannel.append(alphaAverage)
                aroundAlphaAvgPerChannel.append(aroundAlphaAverage)
        return np.array(alphaAvgPerChannel), np.array(aroundAlphaAvgPerChannel)

    def predictSample(self, df, sampleRate):
        if (not self.model):
            return 0
        alphaAvgPerChannel, avgNonAlphaPerChannel = self.getAlphaAndNonalphaFreqAvgsPerChannel(df, sampleRate)
        prediction = self.model.predict(np.append(alphaAvgPerChannel, avgNonAlphaPerChannel).reshape(1, -1))
        return prediction

    def retrainModel(self, samples, labels):
        # samples is a list of np arrays containing fft data, labels is a list of binary values 
        # we fit a logistic regression model to the averaged alpha band power per channel
        clf = LogisticRegression(random_state=0).fit(samples, labels)
        self.model = clf
        return clf

