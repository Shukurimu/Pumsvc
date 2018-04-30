#import tensorflow as tf
import numpy as np
import datetime
import csv
from collections import defaultdict
from copy import deepcopy
from keras.models import Model
from keras.layers import GRU, Dense, Activation, Input
from keras import optimizers
from keras import losses
'''
class RMSProp():
    def __init__():
        self.b
        self.a
    def fun(x):
        pass

class Sigmoid():
    def fun(x):
        expNegX = np.exp(-x)
        return 1/(1+np.exp(-x))
    def derFun(x):
        funValue = fun(x)
        return funValue * (1 - funValue)

class Tanh():
    def fun(x):
        expX = np.exp(x)
        expNegX = np.exp(-x)
        return (expX - expNegX) / (expX + expNegX)
    def derFun(x):
        funValue = fun(x)
        return 1 - funValue ** 2

class GRU():
    #variable as same as https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    def __init__( inputDim, outputDim, activeFun = Sigmoid(), activeFun2 = Tanh(), updateFun = RMSProp()):
        low, high = -0.05, 0.05
        totalDim = inputDim + outputDim
        self.weightR = np.random.uniform(low, high, (outputDim, totalDim))
        self.weightZ = np.random.uniform(low, high, (outputDim, totalDim))
        self.weight = np.random.uniform(low, high, (outputDim, totalDim))
        self.h = np.zeros((outputDim))
        self.activeFun = activeFun
        self.activeFun2 = activeFun2
        self.updateFun = RMSProp()

    def forward( x):
        self.a = np.hstack( self.h, x)
        r = self.activeFun.fun(self.weightR.dot(a))
        z = self.activeFun.fun(self.weightZ.dot(a))
        hLoss = self.activeFun2.fun(self.weight.dot(a))
        self.hNew = (1-z) * self.h + z * hLoss
        return self.hNew
    
    def backprogation( hTrue):
        if self.hNew is None:
            perror("error: Not Forward")
        diff = hTrue - self.hNew
        self.h = self.hNew
        del self.hNew 
'''

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

def restore_process( predDict, adjustCoefs):
    for id, predData in predDict.items():
        predDict[id] = predData * adjustCoefs[id][1] + adjustCoefs[id][0]
    return predDict

def produce_pair( stockDict, days = 15):
    tail = 10
    feature, label = [], []
    for id, dateData in stockDict.items():
        data = np.vstack(dateData.values())
        for i in range( data.shape[0]-days-tail-1):
            feature.append( data[i:i+days])
            label.append( data[i+days])
    return np.array(feature), np.array(label)

#fundBefore = parse_csv("TBrain_Round2_DataSet_20180331/tetfp.csv")
#stockBefore = parse_csv("TBrain_Round2_DataSet_20180331/tsharep.csv")
#stockAfter = parse_csv("TBrain_Round2_DataSet_20180331/tasharep.csv")

if __name__ == '__main__':
    trainDim = 20
    fundAfter = parse_csv("TBrain_Round2_DataSet_20180331/taetfp.csv")
    adjustStock, adjustCoefs = preprocess(fundAfter)
    feature, label = produce_pair(adjustStock)
    print(len(feature), " ", feature[0].shape)
    inputs = Input(shape=(15, 5))
    x = GRU(64, activation='relu')(inputs)
    predictions = Dense(5, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])
    model.fit( feature, label, epochs=10, batch_size=30)
