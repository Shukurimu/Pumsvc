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
        predDict[id] = np.around(predDict[id], decimals=2)
    return predDict

def produce_pair( stockDict, days = 30):
    tail = 10
    feature, label = [], []
    for id, dateData in stockDict.items():
        data = np.vstack(dateData.values())
        for i in range( data.shape[0]-days-tail-1):
            feature.append( data[i:i+days])
            label.append( data[i+days][3])
    tmp = zip(feature, lebel)

    return np.array(feature), np.array(label)

def calculateUpDown(l1, l2):
    tmp = l1 - l2
    return np.where(tmp == 0, 0, np.where(tmp > 0, 1, -1))

def score( realDict, predDict):
    score = 0.0
    for id, predPrice in predDict.items():
        data = np.array(list(realDict[id].values()))
        realPrice = data[-5:,-2]
        predPrice = predPrice[:,-2]
        realUpDown = calculateUpDown(data[-5:,-2], data[-6:-1,-2])
        upDown = calculateUpDown(predPrice, data[-6:-1,-2])
           
        diff = np.abs(realPrice - predPrice)
        upDownDiff = (realUpDown == upDown)
        tmpScore = (realPrice - diff) / realPrice * 0.5 + upDownDiff * 0.5
        score += np.sum(tmpScore * [0.10, 0.15, 0.20, 0.25, 0.30])
    print(score)


if __name__ == '__main__':
    model = []
    for i in range(5):
        model.append(load_model("inin%d.h5"%(i)))

    fundAfter = parse_csv("TBrain_Round2_DataSet_20180331/taetfp.csv")
    adjustStock, adjustCoefs = preprocess(fundAfter)

    result = defaultdict(list)
    for id, dateData in adjustStock.items():
        data = list(dateData.values())[-35:-5]
        data = np.array(data, ndmin=3)
        for i in range(5):
            result[id].append(model[i].predict(data, batch_size=1)[0])
    predDict = restore_preprocess( result, adjustCoefs)
    score(fundAfter, predDict)
    #"ETFid,	Mon_ud,	Mon_cprice,	Tue_ud,	Tue_cprice,	Wed_ud,	Wed_cprice,	Thu_ud,	Thu_cprice,	Fri_ud,	Fri_cprice"
