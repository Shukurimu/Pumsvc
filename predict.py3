import datetime
import csv
import numpy as np
from collections import defaultdict
from copy import deepcopy
from keras.models import load_model

def parse_csv(filePath):
    temDict = defaultdict( lambda:defaultdict(lambda:list))
    with open(filePath, newline='',  encoding="big5-hkscs") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader) #delete first row
        for row in spamreader:
            if len(row) > 8:
                print("error")
            id = int(row[0])
            date = datetime.datetime.strptime(row[1], "%Y%m%d")
            feature = list(map(lambda a: a.replace(",",""), row[3:]))
            feature = list(map(float, feature))
            temDict[id][date] = feature
    return temDict

def preprocess( stockDict):
    tmpStock = deepcopy(stockDict) 
    tmpCoef = {}
    for id, dateData in stockDict.items():
        feature = np.array( list(dateData.values()))
        means = np.mean( feature, axis = 0)
        stds = np.std( feature, axis = 0)
        for date, feature in dateData.items():
            tmpStock[id][date] = (feature - means) / stds
        tmpCoef[id] = (means, stds)
    return tmpStock, tmpCoef

def restore_preprocess( predDict, adjustCoefs):
    for id, predData in predDict.items():
        predDict[id] = predData * adjustCoefs[id][1] + adjustCoefs[id][0]
    return predDict

def produce_pair( stockDict, days = 30):
    tail = 10
    feature, label = [], []
    for id, dateData in stockDict.items():
        data = np.vstack(dateData.values())
        for i in range( data.shape[0]-days-tail-1):
            feature.append( data[i:i+days])
            label.append( data[i+days])
    return np.array(feature), np.array(label)

if __name__ == '__main__':
    model = load_model("inin.h5")
    fundAfter = parse_csv("TBrain_Round2_DataSet_20180331/taetfp.csv")
    adjustStock, adjustCoefs = preprocess(fundAfter)
    result = dict()
    for id, dateDate in adjustStock.items():
        data = list(dateDate.values())[-30:]
        for i in range(5):
            tmpResult = model.predict(np.array([data[-30:]]), batch_size=1)
            data.append(tmpResult[0])
        result[id] = data[-5:]
    predDict = restore_preprocess( result, adjustCoefs)
    print(predDict)