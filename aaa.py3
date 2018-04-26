import tensorflow as tf
import numpy as np
import datetime
import csv
from collections import defaultdict
#from tf.keras import Sequential, Dense

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
    tmpDict = {}
    for id, dateData in stockDict.items():
        feature = np.array( list(dateData.values()))
        means = np.mean( feature, axis = 0)
        stds = np.std( feature, axis = 0)
        for date, feature in dateData.items():
            stockDict[id][date] = (feature - means) / stds
        tmpDict[id] = (means, stds)
    return tmpDict

def restore_process( predDict, adjustCoefs):
    for id, predData in predDict.items():
        predDict[id] = predData * adjustCoefs[id][1] + adjustCoefs[id][0]
    return predDict


#fundBefore = parse_csv("TBrain_Round2_DataSet_20180331/tetfp.csv")
fundAfter = parse_csv("TBrain_Round2_DataSet_20180331/taetfp.csv")
adjustCoefs = preprocess(fundAfter)
print(fundAfter)
#stockBefore = parse_csv("TBrain_Round2_DataSet_20180331/tsharep.csv")
#stockAfter = parse_csv("TBrain_Round2_DataSet_20180331/tasharep.csv")


#for date in dates:
#    print(date)
