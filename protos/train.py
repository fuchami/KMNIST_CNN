#coding:utf-8

#%%
import os, sys, argparse, csv
import numpy as np
import matplotlib.pyplot as plt
import keras
import load, model
from keras.optimizers import SGD, Adam, rmsprop
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K
from adabound import AdaBound
from random_eraser import get_random_eraser

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

def main(args):

    """ log params """
    para_str = 'model_{}_Batchsize{}_opt{}_augmode{}_zscore{}'.format(
        args.model, args.batchsize, args.opt, args.aug_mode, args.zscore)
    print("start this params CNN train: ", para_str)
    para_path = '../train_log/' + para_str
    if not os.path.exists( para_path + '/'):
        os.makedirs( para_path + '/')
    if not os.path.exists('../train_log/log.csv'):
        with open('../train_log/log.csv', 'w')as f:
            writer = csv.writer(f)
            header = ['params', 'train accuracy', 'train loss', 'validation accuracy', 'validation loss']
            writer.writerow(header)

    kmnist_dl = load.KMNISTDataLoader()
    datapath = '../input'
    train_x, train_y, valid_x, valid_y = kmnist_dl.load(datapath)

    #%% plot kminst dataset
    # RandomPlotter().plot(train_x, train_y)

    train_x, train_y, valid_x, valid_y = Preprocessor().transform(train_x, train_y, valid_x, valid_y)

    print('train_x.shape :', train_x.shape)
    print('valid_x.shape :', valid_x.shape)
    print('train_y.shape :', train_y.shape)
    print('valid_y.shape :', valid_y.shape)

    """ z-score """
    if args.zscore == "True":
        print('--- z-score True ---')
        mean = np.mean(train_x, axis=(0,1,2,3))
        std = np.std(train_x, axis=(0,1,2,3))
        train_x = (train_x-mean)/(std+1e-7)
        valid_x = (valid_x-mean)/(std+1e-7)

    #%% Let's train!
    batch_size = args.batchsize
    label_num = 10
    epochs = args.epochs
    base_lr = 0.001
    lr_decay_rate = 1 / 3
    lr_steps = 4
    callbacks = []
    def lr_schedule(epoch):
        lrate = 0.001
        if epoch > 75:
            lrate = 0.0005
        if epoch > 100:
            lrate = 0.0003
        if epoch > 120:
            lrate = 0.0001
        return lrate

    """ build model """
    if args.model == 'prot3':
        select_model = model.prot3()
    elif args.model == 'prot2':
        select_model = model.prot2()
    elif args.model == 'resnet':
        select_model = model.resnet()
    elif args.model == 'wrn_net':
        select_model = model.wrn_net()
    else:
        raise SyntaxError("please select model: prot3 resnet wrn")

    select_model.summary()

    """ select optimizer """
    if args.opt == 'sgd':
        print('--- optimizer: SGD ---')
        opt = SGD(lr=base_lr, momentum=0.9, decay=1e-6, nesterov=True)
        callbacks.append(LearningRateScheduler(lr_schedule))
    elif args.opt == 'rms':
        print('--- optimizer: rmsprop ---')
        opt = rmsprop(lr=0.001, decay=1e-4, rho=0.9, epsilon=1e-08)
        callbacks.append(LearningRateScheduler(lr_schedule))
    elif args.opt == 'adabound':
        print('--- optimizer: adabound ---')
        opt = AdaBound(lr=0.001, final_lr=0.1, gamma=0.001, weight_decay=5e-4, amsbound=False)

    loss = keras.losses.categorical_crossentropy
    select_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    """ add ImageDataGenerator """
    from keras.preprocessing.image import ImageDataGenerator
    train_gen = ImageDataGenerator(rotation_range=15,
                                    width_shift_range=0.08,
                                    height_shift_range=0.08,
                                    preprocessing_function=get_random_eraser(
                                                    p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                                                    v_l=0, v_h=255, pixel_level=False))
    test_gen = ImageDataGenerator()

    trainig_set = train_gen.flow(train_x, train_y, batch_size=batch_size)
    test_set = train_gen.flow(valid_x, valid_y, batch_size=batch_size)

    """ model train """
    select_model.fit_generator(trainig_set,
                        steps_per_epoch = 48000//batch_size,
                        validation_data= test_set,
                        validation_steps=12000//batch_size,
                        epochs=epochs,
                        callbacks=callbacks)

    """ evaluate model """
    train_score = select_model.evaluate(train_x, train_y)
    validation_score = select_model.evaluate(valid_x, valid_y)

    print('Train loss :', train_score[0])
    print('Train accuracy :', train_score[1])
    print('validation loss :', validation_score[0])
    print('validation accuracy :', validation_score[1])

    """ logging score """
    with open('./train_log/log.csv', 'a') as f:
        data = [para_str, train_score[1], train_score[0], validation_score[1], validation_score[0]]
        writer = csv.writer(f)
        writer.writerow(data)

    #%% submit file 
    import pandas as pd
    test_x = np.load('../input/kmnist-test-imgs.npz')['arr_0']
    """ convert images """
    test_x = test_x[:, :, :, np.newaxis].astype('float32') / 255.0

    """ z-score """
    if args.zscore == "True":
        test_x = (test_x-mean)/(std+1e-7)
    print(test_x.shape)

    predicts = np.argmax(select_model.predict(test_x), axis=1)

    submit = pd.DataFrame(data={"ImageId": [], "Label": []})

    submit.ImageId = list(range(1, predicts.shape[0]+1))
    submit.Label = predicts

    submit.to_csv("../output/" + para_str + ".csv", index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train CNN model for classify')
    parser.add_argument('--epochs', '-e', type=int, default=300)
    parser.add_argument('--imgsize', '-s', type=int, default=28)
    parser.add_argument('--batchsize', '-b', type=int, default=64)
    parser.add_argument('--aug_mode', '-a', default='non',
                        help='aug1 aug2 Random erasing')
    parser.add_argument('--model', '-m', default='prot3',
                        help='prot3/resnet/Wide-Res')
    parser.add_argument('--opt', '-o', default='adabound',
                        help='sgd rms adabound')
    parser.add_argument('--zscore', '-z', default='True',
                        help='true false')

    args = parser.parse_args()

    main(args)
