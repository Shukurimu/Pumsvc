import datetime
import csv
import numpy as np
from collections import defaultdict
from copy import deepcopy
from keras.models import load_model

class GaussianNormalize(object):
    def normalize(self, origin):
        tmp = deepcopy(origin)
        self.mean, self.std = {}, {}
        for id, dateData in origin.items():
            feature = np.array(list(dateData.values()), dtype=np.float32)
            self.mean[id] = np.mean(feature, axis = 0)
            self.std[id] = np.std(feature, axis = 0)
            for date, feature in dateData.items():
                tmp[id][date] = (feature - self.mean[id]) / self.std[id]
        return tmp

    def denormalize(self, adjusted, dim=slice(5)):
        if not self.mean or not self.std:
            return adjusted
        tmp = {}
        for id, data in adjusted.items():
            tmp[id] = data * self.std[id][dim] + self.mean[id][dim]
        return tmp

class MaxNormalize(object):
    def normalize(self, origin):
        tmp = deepcopy(origin)
        self.max = {}
        for id, dateData in origin.items():
            feature = np.array(list(dateData.values()), dtype=np.float32)
            self.max[id] = np.max(feature, axis = 0)
            for date, feature in dateData.items():
                tmp[id][date] = feature / self.max[id]
        return tmp

    def denormalize(self, adjusted, dim=slice(5)):
        if not self.max:
            return adjusted
        tmp = {}
        for id, data in adjusted.items():
            tmp[id] = data * self.max[id][dim]
        return tmp

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

def calculatePriceUpDown(npArray):
    price = npArray[-6:,-2] if npArray.ndim == 2 else npArray[-6:]
    upDown = np.around(price[-5:] * 100) - np.around(price[-6:-1] * 100)
    return price[-5:], np.where(upDown == 0, 0, np.where(upDown > 0, 1, -1))

def score(realDict, predDict):
    score = 0.0
    for id, predData in predDict.items():
        realPrice, realUpDown = calculatePriceUpDown(np.array(list(realDict[id].values())))
        predPrice, predUpDown = calculatePriceUpDown(predData)
           
        diff = np.abs(realPrice - predPrice)
        upDownDiff = (realUpDown == predUpDown)
        tmpScore = ((realPrice - diff) / realPrice + upDownDiff) * 0.5
        score += np.sum(tmpScore * [0.10, 0.15, 0.20, 0.25, 0.30])
        print(upDownDiff*1)
        print((realPrice - diff) / realPrice)
    print(score)

def produce_pair( stockDict, postpone, days = 30):
    tail = 30
    feature, label = [], []
    for id, dateData in stockDict.items():
        data = np.vstack(dateData.values())
        for i in range( data.shape[0]-days-tail-1):
            feature.append( data[i:i+days])
            label.append( data[i+days+postpone][3])
    return np.array(feature), np.array(label)

def new_produce_pair( stockDict, days = 30):
    tail = 30
    feature, label = [], []
    for id, dateData in stockDict.items():
        data = np.vstack(dateData.values())
        for i in range( data.shape[0]-days-tail-1):
            feature.append(data[i:i+days])
            label.append(list(map(lambda a: a[-2], data[i+days+1:i+days+6])))
    return np.array(feature), np.array(label)