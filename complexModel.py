from functional import *
from optparse import OptionParser
from keras.models import Model, load_model
from keras.layers import GRU, Dense, Activation, Input, Lambda, Concatenate, Dropout
from keras import optimizers, losses

#fundBefore = parse_csv("TBrain_Round2_DataSet_20180511/tetfp.csv")
#stockBefore = parse_csv("TBrain_Round2_DataSet_20180511/tsharep.csv")
#stockAfter = parse_csv("TBrain_Round2_DataSet_20180511/tasharep.csv")
def create_model_arch():
    gruDim = 1
    inputs = Input(shape=(30, 5))
    
    gruTensor = GRU(5*gruDim)(inputs)
    middle = []
    for i in range(1, 6):
        tmp = Lambda(lambda x: x[:,:gruDim*i], output_shape=(gruDim*i,))(gruTensor)
        tmp = Dropout(0.5)(tmp) 
        middle.append(Dense(1, activation='linear')(tmp))

    outputs = Concatenate()(middle)
    return Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    optParser = OptionParser()
    optParser.add_option("-t", "--train", action= "store_true", default=False, help="train")
    optParser.add_option("-v", "--validate", action= "store_true", default=False, help="validate")
    optParser.add_option("-p", "--predict", action= "store_true", default=False, help="predict")
    (options, args) = optParser.parse_args()

    fundAfter = parse_csv("TBrain_Round2_DataSet_20180511/taetfp.csv")
    adjust = GaussianNormalize()
    adjustFund = adjust.normalize(fundAfter)

    stockAfter = parse_csv("TBrain_Round2_DataSet_20180511/tasharep.csv")
    adjust2 = GaussianNormalize()
    adjustStock = adjust2.normalize(stockAfter)

    model = create_model_arch()
    modelWeightName = "complexGRUGG.h5"

    if options.train:
        featureS, labelS = new_produce_pair(adjustFund)
        featureF, labelF = new_produce_pair(adjustStock)
        feature = np.vstack((featureS, featureF))
        label = np.vstack((labelS, labelF))
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit( feature, label, epochs=1, batch_size=32)
        model.save_weights(modelWeightName)

    if options.validate or options.predict:
        result = {}
        model.load_weights(modelWeightName)

        for val in [options.validate*1, options.predict*2]:
            if val <= 0:
                continue
            for id, dateData in adjustFund.items():
                data = list(dateData.values())
                data = data[-35:-5] if val < 2 else data[-30:]
                data = np.array(data, dtype=np.float32, ndmin=3)
                result[id] = np.hstack((data[:,-1,-2], model.predict(data, batch_size=1)[0]))
            predDict = adjust.denormalize(result, dim=-2)
            if val < 2:
                score(fundAfter, predDict)
            else:
                write_csv("result.csv", predDict)

            