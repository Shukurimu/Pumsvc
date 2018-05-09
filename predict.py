from functional import *
from collections import defaultdict
from keras.models import load_model

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
    
