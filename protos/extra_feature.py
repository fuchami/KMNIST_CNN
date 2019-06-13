# coding-utf8
"""
特徴抽出をしてnpzに保存する
"""

import os,sys
import numpy as np
import load
import keras
from keras.models import Model

def main():

    """ dataset load """
    kmnist_dl = load.KMNISTDataLoader(validation_size=0.0, img_resize=38)
    train_x, train_y, valid_x, valid_y = kmnist_dl.load('../input/')
    train_x, train_y, valid_x, valid_y = load.Preprocessor().transform(train_x, train_y, valid_x, valid_y)
    print('train_x:', train_x.shape)

    """ load model"""
    model_path = "../train_log/wrn-4-8-SE-True-NeXt-True_imgsize38_batchsize128_adabound_fullTrain_custom_expandConv/model.h5"
    # model = keras.models.load_model(model_path)
    # model.summary()
    # feature_extra_model = Model(inputs=model.input,
        # outputs=model.get_layer("").output)

    feature_x = []

    for x in train_x:
        x = expand(x[:,:,0])
        # y = feature_extra_model.predict(x)
        print("x.shape:", x.shape)

def expand(x):
    return np.expand_dims(np.expand_dims(x, axis=0), axis=3)
if __name__ == "__main__":
    main()