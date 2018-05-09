from functional import *
from keras.models import Model
from keras.layers import GRU, Dense, Activation, Input
from keras import optimizers, losses

#fundBefore = parse_csv("TBrain_Round2_DataSet_20180331/tetfp.csv")
#stockBefore = parse_csv("TBrain_Round2_DataSet_20180331/tsharep.csv")
#stockAfter = parse_csv("TBrain_Round2_DataSet_20180331/tasharep.csv")

if __name__ == '__main__':
    fundAfter = parse_csv("TBrain_Round2_DataSet_20180331/taetfp.csv")
    adjustStock, adjustCoefs = preprocess(fundAfter)

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