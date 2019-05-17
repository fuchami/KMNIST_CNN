#coding:utf-8

#%%
import os, sys, argparse, csv
import h5py

import numpy as np
import matplotlib.pyplot as plt
import keras
import load, model, tools, myopt
from keras.optimizers import SGD, Adam, rmsprop
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from advanced_optimizers import AdaBound, RMSpropGraves

def main(args):

    """ log params """
    para_str = '{}_batchsize{}_{}_zscore{}'.format(
        args.model, args.batchsize, args.opt, args.zscore)
    print("start this params CNN train: ", para_str)
    para_path = '../train_log/' + para_str
    """ 単一モデル用のログ """
    if not os.path.exists( para_path + '/'):
        os.makedirs( para_path + '/')
    """ 比較用の統括ログ """
    if not os.path.exists('../train_log/log.csv'):
        with open('../train_log/log.csv', 'w')as f:
            writer = csv.writer(f)
            header = ['params', 'train accuracy', 'train loss', 'validation accuracy', 'validation loss']
            writer.writerow(header)

    """ dataset load """
    kmnist_dl = load.KMNISTDataLoader()
    datapath = '../input'
    train_x, train_y, valid_x, valid_y = kmnist_dl.load(datapath)
    train_x, train_y, valid_x, valid_y = load.Preprocessor().transform(train_x, train_y, valid_x, valid_y)

    """ z-score """
    if args.zscore == "True":
        print('--- z-score True ---')
        mean = np.mean(train_x, axis=(0,1,2,3))
        std = np.std(train_x, axis=(0,1,2,3))
        train_x = (train_x-mean)/(std+1e-7)
        valid_x = (valid_x-mean)/(std+1e-7)

    """ define hyper parameters """
    label_num = 10
    base_lr = 0.001
    lr_decay_rate = 1 / 3
    lr_steps = 4
    csv_logger = CSVLogger( para_path + '/log.csv', separator=',')
    callbacks = [ csv_logger]

    def lr_schedule(epoch):
        lrate = 0.001
        if epoch > 75:
            lrate = 0.0005
        if epoch > 100:
            lrate = 0.0003
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
        raise SyntaxError("please select model")
    select_model.summary()


    """ select optimizer """
    if args.opt == 'sgd':
        print('--- optimizer: SGD ---')
        opt = SGD(lr=base_lr, momentum=0.9, decay=1e-6, nesterov=True)
        callbacks.append(LearningRateScheduler(lr_schedule))
    elif args.opt == 'rms':
        print('--- optimizer: RMSpropGraves ---')
        # opt = rmsprop(lr=0.001, decay=1e-6)
        opt = RMSpropGraves(lr=0.001, decay=1e-6)
        callbacks.append(LearningRateScheduler(lr_schedule))
    elif args.opt == 'adabound':
        print('--- optimizer: adabound ---')
        opt = AdaBound(lr=0.001, final_lr=0.1, gamma=0.001, weight_decay=5e-4, amsbound=False)

    loss = keras.losses.categorical_crossentropy
    select_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    train_generator, valid_generator = load.mygenerator(args, train_x, train_y, valid_x, valid_y, label_num)

    """ model train """
    history = select_model.fit_generator(train_generator,
                        steps_per_epoch = 51000//args.batchsize,
                        validation_data= valid_generator,
                        validation_steps=9000//args.batchsize,
                        epochs=args.epochs,
                        callbacks=callbacks)
    """ plot learning history """
    # tools.plot_history(history, para_str, para_path)

    """ evaluate model """
    train_score = select_model.evaluate(train_x, train_y)
    validation_score = select_model.evaluate(valid_x, valid_y)

    print('Train loss :', train_score[0])
    print('Train accuracy :', train_score[1])
    print('validation loss :', validation_score[0])
    print('validation accuracy :', validation_score[1])

    """ logging score """
    with open('../train_log/log.csv', 'a') as f:
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
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--imgsize', '-s', type=int, default=28)
    parser.add_argument('--batchsize', '-b', type=int, default=128)
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
