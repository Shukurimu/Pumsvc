from functional import *
from optparse import OptionParser
from keras.models import Model, load_model
from keras.layers import GRU, Dense, Activation, Input, Lambda, Concatenate, Dropout
from keras import optimizers, losses

#fundBefore = parse_csv("TBrain_Round2_DataSet_20180511/tetfp.csv")
#stockBefore = parse_csv("TBrain_Round2_DataSet_20180511/tsharep.csv")
#stockAfter = parse_csv("TBrain_Round2_DataSet_20180511/tasharep.csv")
def create_model_arch(gruDims = 32, days = 30):
    
    inputs = Input(shape=(days, 5))
    
    gruTensor = GRU(5*gruDims)(inputs)
    middle = []
    for i in range(1, 6):
        tmp = Lambda(lambda x: x[:,:gruDims*i], output_shape=(gruDims*i,))(gruTensor)
        middle.append(Dense(1, activation='linear')(tmp))

    outputs = Concatenate()(middle)
    return Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    optParser = OptionParser()
    optParser.add_option("-t", "--train", action= "store_true", default=False, help="train")
    optParser.add_option("-v", "--validate", action= "store_true", default=False, help="validate")
    optParser.add_option("-p", "--predict", action= "store_true", default=False, help="predict")
    (options, args) = optParser.parse_args()

    fundAfter = parse_csv("TBrain_Round2_DataSet_20180525/taetfp.csv")
    adjust = GaussianNormalize()
    adjustFund = adjust.normalize(fundAfter)

    stockAfter = parse_csv("TBrain_Round2_DataSet_20180525/tasharep.csv")
    adjust2 = GaussianNormalize()
    adjustStock = adjust2.normalize(stockAfter)
    del stockAfter

    days, gruDims, epochs = 30, 16, 100
    model = create_model_arch(gruDims=gruDims ,days=days)
    modelWeightName = "cm_%d_%d.h5"%(gruDims, epochs)

    if options.train:
        #featureS, labelS = new_produce_pair(adjustFund, days=days)
        #featureF, labelF = new_produce_pair(adjustStock, days=days)
        #del adjustStock
        #feature = np.vstack((featureS, featureF))
        #label = np.vstack((labelS, labelF))
        feature, label = new_produce_pair(adjustFund, days=days)
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit( feature, label, epochs=epochs, batch_size=32)
        model.save_weights(modelWeightName)

    if options.validate or options.predict:
        result = {}
        model.load_weights(modelWeightName)

        for val in [options.validate*1, options.predict*2]:
            if val <= 0:
                continue
            for id, dateData in adjustFund.items():
                data = list(dateData.values())
                data = data[-days-5:-5] if val < 2 else data[-days:]
                data = np.array(data, dtype=np.float32, ndmin=3)
                result[id] = np.hstack((data[:,-1,-2], model.predict(data, batch_size=1)[0]))
            predDict = adjust.denormalize(result, dim=-2)
            if val < 2:
                score(fundAfter, predDict)
            else:
                write_csv("result.csv", predDict)

            