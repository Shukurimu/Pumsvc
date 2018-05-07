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

def write_csv(filePath, predDict):
    with open(filePath, "w", newline='',  encoding="big5-hkscs") as csvfile:
        headerRowString = "ETFid Mon_ud Mon_cprice Tue_ud Tue_cprice Wed_ud Wed_cprice Thu_ud Thu_cprice Fri_ud Fri_cprice"
        spamWriter = csv.writer(csvfile, delimiter=',')
        spamWriter.writerow(headerRowString.split(" "))
        for id, predData in predDict.items():
            predPrice, predUpDown = calculatePriceUpDown(predData)
            print(predPrice)
            tmp = [id]
            for pP, pUD in zip(predPrice.tolist(), predUpDown.tolist()):
                tmp.append(pUD)
                tmp.append(round(pP, 2))
            print(tmp)
            spamWriter.writerow(tmp)

def preprocess( stockDict):
    tmpStock = deepcopy(stockDict) 
    tmpCoef = {}
    for id, dateData in stockDict.items():
        feature = np.array(list(dateData.values()), dtype=np.float32)
        means = np.mean(feature, axis = 0)
        stds = np.std(feature, axis = 0)
        for date, feature in dateData.items():
            tmpStock[id][date] = (feature - means) / stds
        tmpCoef[id] = (means, stds)
    return tmpStock, tmpCoef

def restore_preprocess( predDict, adjustCoefs):
    for id, predData in predDict.items():
        predDict[id] = predData * adjustCoefs[id][1] + adjustCoefs[id][0]
        predDict[id] = np.around(predDict[id], decimals=2)
    return predDict

def calculatePriceUpDown(npArray):
    price = npArray[-5:,-2]
    upDown = np.around(npArray[-5:,-2] * 100) - np.around(npArray[-6:-1,-2] * 100)
    return price, np.where(upDown == 0, 0, np.where(upDown > 0, 1, -1))

def score(realDict, predDict):
    score = 0.0
    for id, predData in predDict.items():
        realPrice, realUpDown = calculatePriceUpDown(np.array(list(realDict[id].values())))
        predPrice, predUpDown = calculatePriceUpDown(predData)
           
        diff = np.abs(realPrice - predPrice)
        upDownDiff = (realUpDown == predUpDown)
        tmpScore = ((realPrice - diff) / realPrice + upDownDiff) * 0.5
        score += np.sum(tmpScore * [0.10, 0.15, 0.20, 0.25, 0.30])
    print(score)



if __name__ == '__main__':
    model = []
    for i in range(5):
        model.append(load_model("ininGG%d.h5"%(i)))

    fundAfter = parse_csv("TBrain_Round2_DataSet_20180504/taetfp.csv")
    adjustStock, adjustCoefs = preprocess(fundAfter)

    result = defaultdict(list)
    for id, dateData in adjustStock.items():
        data = list(dateData.values())[-30:]
        data = np.array(data, dtype=np.float32, ndmin=3)
        result[id].append(data[:,-1,-2])
        for i in range(5):
            result[id].append(model[i].predict(data, batch_size=1)[0])

    predDict = restore_preprocess(result, adjustCoefs)
    score(fundAfter, predDict)
    write_csv("result.csv", predDict)
    
