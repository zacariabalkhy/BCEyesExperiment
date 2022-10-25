"""Example program to show how to read a multi-channel time series from LSL."""

from os import times
from pylsl import StreamInlet, resolve_stream
import datetime
from StreamReader import StreamReader
from Classifier import Classifier
import pandas as pd
from Experiment import Experiment

model = None
def main():
    streamReader = None
    try:
        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        if not streams or streams[0]:
            return
        streamReader = StreamReader(1, streams[0])
        classifier = Classifier(model)
        sampleRate = streamReader.sampleRate
        print("begin reading data.......")
        while True:
            df = getLastSecondOfData(streamReader=streamReader)
            
            if df.empty:
                continue
            print(df.shape)
            prediction = classifier.predictSample(df, sampleRate)
            print(prediction)
    except KeyboardInterrupt:
        if streamReader:
            streamReader.shutDownStreamer()

def runExperiment(numTrials, trialLength):
    experiment = Experiment(numTrials, trialLength)
    experiment.runExperiment()
    return experiment.EEGdata, experiment.experimentMetaData



def getLastSecondOfData(streamReader):
    data = streamReader.getSecondWorthOfData()
    df = pd.DataFrame(data)
    return df
    
if __name__ == "__main__":
    print("Welcome to the streamer. Type train to train a new model. Type test to test the current model\n")
    val = input("Type train to train a new model. Type test to test the current model: ")
    if (val == "train"):
        print("Starting training module")
        numTrials = int(input("Please enter your desired number of trials: "))
        trialLength = int(input("Please enter desired trial length in seconds: "))
        rawData, experimentMetaData = runExperiment(numTrials, trialLength)

        print(len(rawData))
        print(rawData[len(rawData) - 1])
        for i in range(15):
            print(rawData[i])

        for i in range(len(experimentMetaData)):
            print(experimentMetaData[i])

    elif val == "test":
        print("starting test")
    
    #main()