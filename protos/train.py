#coding:utf-8

#%%
import numpy as np
import matplotlib.pyplot as plt
import keras
import load
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, BatchNormalization, Dropout, Input
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.utils import to_categorical
from keras import backend as K
#%matplotlib inline

#%% load kmnist dataset

kmnist_dl = load.KMNISTDataLoader()
datapath = '../input'
train_x, train_y, valid_x, valid_y = kmnist_dl.load(datapath)

#%% plot kminst dataset
class RandomPlotter(object):
    def __init__(self):
        self.label_char = ["お(o)", "き(ki)", "す(su)", "つ(tsu)",\
                           "な(na)", "は(ha)", "ま(ma)", "や(ya)",\
                           "れ(re)", "を(wo)"]
        # plt.rcParams['font.family'] = 'IPAPGothic'
    
    def _get_unique_labels(self, labels: np.ndarray) -> np.ndarray:
        label_unique = np.sort(np.unique(labels))
        return label_unique
    
    def _get_random_idx_list(self, labels: np.ndarray) -> list:
        label_unique = self._get_unique_labels(labels)

        random_idx_list = []
        for label in label_unique:
            label_indices = np.where(labels == label)[0]
            random_idx = np.random.choice(label_indices)
            random_idx_list.append(random_idx)
        
        return random_idx_list
    
    def plot(self, images: np.ndarray, labels: np.ndarray) -> None:
        random_idx_list = self._get_random_idx_list(labels)

        fig = plt.figure()
        for i, idx in enumerate(random_idx_list):
            ax = fig.add_subplot(2, 5, i+1)
            ax.tick_params(labelbottom=False, bottom=False)
            ax.tick_params(labelleft=False, left=False)
            img = images[idx]
            ax.imshow(img, cmap="gray")
            ax.set_title(self.label_char[i])
        
        fig.show()

RandomPlotter().plot(train_x, train_y)
#%% [markdown]
# ## Data Preprocessing
# - 数値データの型をfloat32へ変更
# - 画像のndarrayのshapeを(N, 28, 28)から(N, 28, 28, 1)に変更（Nは画像の枚数）
# - 値を[0, 255]から[0, 1]に標準化

class Preprocessor(object):
    def transform(self, train_x, train_y, valid_x, valid_y):
        """ convert images """
        train_x = train_x[:, :, :, np.newaxis].astype('float32') / 255.0
        valid_x = valid_x[:, :, :, np.newaxis].astype('float32') / 255.0

        """ convert labels """
        label_num = len(np.unique(train_y))
        train_y = to_categorical(train_y, label_num)
        valid_y = to_categorical(valid_y, label_num)

        return train_x, train_y, valid_x, valid_y

train_x, train_y, valid_x, valid_y = Preprocessor().transform(train_x, train_y, valid_x, valid_y)

print('train_x.shape :', train_x.shape)
print('valid_x.shape :', valid_x.shape)
print('train_y.shape :', train_y.shape)
print('valid_y.shape :', valid_y.shape)

#%% Let's train!
batch_size =128
label_num = 10
epochs = 1
base_lr = 0.001
lr_decay_rate = 1 / 3
lr_steps = 4
reduce_lr = LearningRateScheduler(lambda ep: float(base_lr * lr_decay_rate ** (ep * lr_steps // epochs)), verbose=1)
input_shape = (28, 28, 1)

""" build model """
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.25))
model.add(Dense(label_num, activation='softmax'))

model.summary()

loss = keras.losses.categorical_crossentropy
optimizer = SGD(lr=base_lr, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

""" add ImageDataGenerator """
from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rotation_range=20,
                                width_shift_range=0.08,
                                shear_range=0.2,
                                height_shift_range=0.08,
                                zoom_range=0.08)
test_gen = ImageDataGenerator()

trainig_set = train_gen.flow(train_x, train_y, batch_size=batch_size)
test_set = train_gen.flow(valid_x, valid_y, batch_size=batch_size)

""" model train """
model.fit_generator(trainig_set,
                    steps_per_epoch = 48000//batch_size,
                    validation_data= test_set,
                    validation_steps=12000//batch_size,
                    epochs=epochs)

train_score = model.evaluate(train_x, train_y)
validation_score = model.evaluate(valid_x, valid_y)

print('Train loss :', train_score[0])
print('Train accuracy :', train_score[1])
print('validation loss :', validation_score[0])
print('validation accuracy :', validation_score[1])

#%% submit file 
import pandas as pd
test_x = np.load('../input/kmnist-test-imgs.npz')['arr_0']
""" convert images """
test_x = test_x[:, :, :, np.newaxis].astype('float32') / 255.0
print(test_x.shape)

predicts = np.argmax(model.predict(test_x), axis=1)

submit = pd.DataFrame(data={"ImageId": [], "Label": []})

submit.ImageId = list(range(1, predicts.shape[0]+1))
submit.Label = predicts

submit.to_csv("../output/submit.csv", index=False)