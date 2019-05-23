# coding:utf-8
"""
いろんなモデルを試してみる
"""

import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, BatchNormalization, Dropout, Input
from keras.layers import AveragePooling2D, Input
from keras.models import Sequential, Model
from keras import regularizers
from keras.regularizers import l2
from keras import backend as K

from funcy import concat, identity, juxt, partial, rcompose, repeat, repeatedly, take
import wrn


def wrn_net(imgsize):
    model = Model(*juxt(identity, wrn.computational_graph(10))(Input(shape=(imgsize,imgsize,1))))

    return model

def prot1():
    """ with dropout """
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def prot2():
    """
    BatchNorm
    https://www.kaggle.com/kentaroyoshioka47/cnn-with-batchnormalization-in-keras-94#L71
    """
    model = Sequential()
    #convolution 1st layer
    model.add(Conv2D(64, (3,3), padding="same",
                    activation='relu',
                    input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))#3
    #model.add(MaxPooling2D())

    #convolution 2nd layer
    model.add(Conv2D(64, (3,3), activation='relu',border_mode="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    #convolution 3rd layer
    model.add(Conv2D(64, (3,3), activation='relu',border_mode="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    #Fully connected 1st layer
    model.add(Flatten()) 
    model.add(Dense(500,use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    #Fully connected final layer
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def prot3(args):
    """
    https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
    """
    weight_decay = 1e-4 #記事通りなら1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(args.imgsize,args.imgsize,1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(GlobalAveragePooling2D())

    model.add(Dense(10, activation='softmax'))

    return model
