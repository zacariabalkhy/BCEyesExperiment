from os import times
from pylsl import StreamInlet, resolve_stream
from datetime import datetime
from StreamReader import StreamReader
from Classifier import Classifier
import pandas as pd
from Experiment import Experiment
import time

model = None
def readStream(classifier):
    streamReader = None
    try:
        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        if not streams or not streams[0]:
            return
        streamReader = StreamReader(1, streams[0])
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
    columns = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8", "timestamp"]
    data = streamReader.getSecondWorthOfData()
    df = pd.DataFrame(data, columns=columns)
    return df
    
if __name__ == "__main__":
    format = "%d-%b-%Y %H:%M:%S.%f"
    classifier = Classifier()
    experiment = None
    print("Welcome to the streamer. Type train to train a new model. Type test to test the current model\n")
    val = input("Type train to train a new model. Type test to test the current model: ")
    if (val == "train"):
        print("Starting training module")
        numTrials = int(input("Please enter your desired number of trials: "))
        trialLength = int(input("Please enter desired trial length in seconds: "))
        experiment = Experiment(numTrials, trialLength)
        experiment.runExperiment()
        experiment.cleanData()
        classifier.processDataAndTrainModel(experiment.trialSeparatedData, experiment.sampleRate)
        print("training done, now reading stream and making real time predictions")

        readStream(classifier)
    elif val == "test":
        print("starting test")
    