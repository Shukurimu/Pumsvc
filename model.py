from functional import *

from optparse import OptionParser
from keras.models import Model, load_model
from keras.layers import GRU, Dense, Activation, Input
from keras import optimizers, losses


#fundBefore = parse_csv("TBrain_Round2_DataSet_20180331/tetfp.csv")
#stockBefore = parse_csv("TBrain_Round2_DataSet_20180331/tsharep.csv")
#stockAfter = parse_csv("TBrain_Round2_DataSet_20180331/tasharep.csv")

if __name__ == '__main__':
    optParser = OptionParser()
    optParser.add_option("-t", "--train", action= "store_true", help="train")
    optParser.add_option("-v", "--validate", action= "store_true", help="validate")
    optParser.add_option("-p", "--predict", action= "store_true", help="predict")
    (options, args) = optParser.parse_args()

    fundAfter = parse_csv("TBrain_Round2_DataSet_20180525/taetfp.csv")
    adjust = GaussianNormalize()
    adjustStock = adjust.normalize(fundAfter)

    if options.train:
        for i in range(5):
            feature, label = produce_pair(adjustStock, i)
            #build model
            inputs = Input(shape=(30, 5))
            x = GRU(128, activation='relu')(inputs)
            predictions = Dense(1, activation='linear')(x)

            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer='rmsprop',
                    loss='mean_absolute_error')
            model.fit( feature, label, epochs=300, batch_size=32)
            model.save("ininII%d.h5"%(i))
    
    if options.validate or options.predict:
        model = []    
        result = defaultdict(list)
        for i in range(5):
            model.append(load_model("ininII%d.h5"%(i)))

        if options.validate:
            for id, dateData in adjustStock.items():
                data = list(dateData.values())[-35:-5]
                data = np.array(data, dtype=np.float32, ndmin=3)
                result[id].append(data[:,-1,-2])
                for i in range(5):
                    result[id].append(model[i].predict(data, batch_size=1)[0])
            predDict = adjust.denormalize(result)
            score(fundAfter, predDict)
        
        else:
            for id, dateData in adjustStock.items():
                data = list(dateData.values())[-30:]
                data = np.array(data, dtype=np.float32, ndmin=3)
                result[id].append(data[:,-1,-2])
                for i in range(5):
                    result[id].append(model[i].predict(data, batch_size=1)[0])
            predDict = adjust.denormalize(result)
            write_csv("result.csv", predDict)