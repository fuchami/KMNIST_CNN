# coding:utf-8
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Add, Multiply, Lambda, concatenate
from keras.regularizers import l2
from keras import backend as K


"""
Res block: BN - Conv(3x3) - BN - ReLU - (Dropout?) - Conv(3x3) - BN

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4

N: depth. Compute N = (n - 4) / 6
k: width

Dropout 0.3~0.4
"""

def se_block(in_block, ch, ratio=8):
    z = GlobalAveragePooling2D()(in_block)
    x = Dense(ch//ratio, activation='relu')(z)
    x = Dense(ch, activation='sigmoid')(x)
    return Multiply()([in_block, x])

def expand_conv(init, base, k, strides=(1,1), weight_decay=0.0005, se_module=False):
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    x = Activation('relu')(x)

    x = Conv2D(base * k, (3, 3), padding='same',strides=strides,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay), use_bias=False)(x)

    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Conv2D(base * k, (3, 3), padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay), use_bias=False)(x)
    
    if se_module : x = se_block(x, base*k)

    skip = Conv2D(base * k, (1, 1), padding='same', strides=strides,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay), use_bias=False)(init)
    
    m = Add()([x, skip])
    return m

def resblock(input, k=1, k_filter=16, dropout=0.4, weight_decay=0.0005, se_module=False, NeXt=False, cardinality=16):
    init = input

    if NeXt == False:
        print('Wide ResNet')
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
        x = Conv2D(k_filter * k, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False)(x)

        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        if dropout > 0.0: x = Dropout(dropout)(x)

        x = Conv2D(k_filter * k, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False)(x)
        
        if se_module : x = se_block(x, k_filter*k)

        m = Add()([init, x])
        return m
    
    # NeXt
    group_list = []
    grouped_channels = int((k * k_filter)/ cardinality)

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)
       
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Conv2D(grouped_channels, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False)(x)

        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        if dropout > 0.0: x = Dropout(dropout)(x)

        x = Conv2D(grouped_channels, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False)(x)
        
        if se_module : x = se_block(x, grouped_channels)
        group_list.append(x)
    
    group_merge = concatenate(group_list)
    m = Add()([init, group_merge])
    return m

def create_wide_residual_network(input_dim, nb_classes=10, N=4, k=10, dropout=0.4, 
                                weight_decay=0.0005, se_module=False, NeXt=False, cardinality=16):
    inputs = Input(shape=input_dim)

    # initial_conv
    x = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay), use_bias=False)(inputs)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 16, k, se_module=se_module)

    # conv1
    for i in range(N-1):
        x = resblock(x, k, k_filter=16, dropout=dropout, se_module=se_module, NeXt=NeXt, cardinality=cardinality)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    
    x = expand_conv(x, 32, k, strides=(2,2), se_module=se_module)

    # conv2
    for i in range(N-1):
        x = resblock(x, k, k_filter=32, dropout=dropout, se_module=se_module, NeXt=NeXt, cardinality=cardinality)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, strides=(2,2), se_module=se_module)

    # conv3
    for i in range(N-1):
        x = resblock(x, k, k_filter=64, dropout=dropout, se_module=se_module, NeXt=NeXt, cardinality=cardinality)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, kernel_regularizer=l2(weight_decay), activation='softmax')(x)
    model = Model(inputs, x)

    return model