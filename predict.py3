from . train import preprocess, restore_preprocess, parse_csv
from keras.models import load_model

def produce_pair( stockDict, days = 30):
    tail = 10
    feature, label = [], []
    for id, dateData in stockDict.items():
        data = np.vstack(dateData.values())
        for i in range( data.shape[0]-days-tail-1):
            feature.append( data[i:i+days])
            label.append( data[i+days])
    return np.array(feature), np.array(label)

if __name__ is "__main__":
    fundAfter = parse_csv("TBrain_Round2_DataSet_20180331/taetfp.csv")
    adjustStock, adjustCoefs = preprocess(fundAfter)
    for id, dateDate in fundAfter.items():
        fundAfter[id] = dateDate.values[-30:]
        print(fundAfter[id])