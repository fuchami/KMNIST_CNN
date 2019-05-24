# coding:utf-8
"""
いろんなモデルを試してみる
"""

import numpy as np
import tensorflow as tf
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
from wideresnet import se_block

def prot3_SE(args):
    """
    https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
    これにSEmoduleを追加した版
    """

    weight_decay = 1e-4 
    inputs = Input(shape=(args.imgsize, args.imgsize, 1))

    x = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(inputs)
    x = Activation('elu')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    if args.se : x = se_block(x, 32)

    x = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
    x = Activation('elu')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)
    if args.se : x = se_block(x, 32)

    x = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
    x = Activation('elu')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    if args.se : x = se_block(x, 64)

    x = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
    x = Activation('elu')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)
    if args.se : x = se_block(x, 64)

    x = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
    x = Activation('elu')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    if args.se : x = se_block(x, 128)

    x = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
    x = Activation('elu')(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)
    if args.se : x = se_block(x, 128)

    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs, x)
    return model
