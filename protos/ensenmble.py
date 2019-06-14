# coding:utf-8
"""
複数の分類器によるアンサンブル
"""

import numpy as np
import keras
from keras.models import load_model
import load
import pandas as pd


def ensembling(models_path, X):
    preds_sum = None
    for model_path in models_path:
        print("=== load model === : ", model_path)
        model = load_model(model_path, compile=False)
        if preds_sum is None:
            preds_sum = model.predict(X)
        else:
            preds_sum += model.predict(X)
        
    probs = preds_sum / len(models_path)
    return np.argmax(probs, axis=1)


def main(models_path):
    """ dataset load """
    kmnist_dl = load.KMNISTDataLoader(validation_size=0.2, img_resize=38)
    train_x, train_y, valid_x, valid_y = kmnist_dl.load('../input/')
    train_x, train_y, valid_x, valid_y = load.Preprocessor().transform(train_x, train_y, valid_x, valid_y)
    print('train_x:', train_x.shape)
    """ testdata load """
    test_x = load.test_load(imgsize=38)

    """ calc train_x score """
    pred_test = ensembling(models_path, test_x)
    # pred_train_acc = accuracy_score(np.ravel(train_y), pred_train)
    # print("train_x Ensemble Accuracy: ", pred_train_acc)

    print('predicts.shape: ', pred_test.shape) # hope shape(10000, )
    print('predicts: ', pred_test)
    submit = pd.DataFrame(data={"ImageId": [], "Label": []})
    submit.ImageId = list(range(1, pred_test.shape[0]+1))
    submit.Label = pred_test

    submit.to_csv("../output/ensemble3.csv", index=False)

if __name__ == "__main__":
    models_path = [
        '../train_log/wrn_imgsize38_batchsize128_adabound_SEmodule_Truealltrain/model.h5',
        '../train_log/wrn-4-8-SE-True-NeXt-True_imgsize38_batchsize128_adabound_fullTrain_custom_expandConv/model.h5',
        '../train_log/wrn-2-10-SE-True-NeXt-False_imgsize38_batchsize128_adabound_full-Train_custom_expandConv-witoutSWA/model.h5'
    ]


    main(models_path)
