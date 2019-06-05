#coding:utf-8

import os, sys, argparse, csv
import h5py

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD, Adam, rmsprop
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from advanced_optimizers import AdaBound, RMSpropGraves
from swa import SWA

import load, tools 
from model import prot4_SE
from wideresnet import create_wide_residual_network

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list = "0,1", # specify GPU number
        allow_growth = True
    )
)
set_session(tf.Session(config=config))

def main(args):
    label_num = 10

    """ log params """
    if args.model == 'wrn':
        para_str = '{}-{}-{}-SE-{}-NeXt-{}_imgsize{}_batchsize{}_{}=fullTrain_custom_expandConv'.format(
            args.model, args.wrn_n, args.wrn_k, args.se, args.next,
            args.imgsize, args.batchsize, args.opt)
    else:
        para_str = '{}-SE-{}_imgsize{}_batchsize{}_{}'.format(
            args.model, args.se, args.imgsize, args.batchsize, args.opt)

    print("start this params CNN train: ", para_str)
    para_path = '../train_log/' + para_str
    """ model logging """
    if not os.path.exists( para_path + '/'):
        os.makedirs( para_path + '/')
    """ total model logging for compare """
    if not os.path.exists('../train_log/log.csv'):
        with open('../train_log/log.csv', 'w')as f:
            writer = csv.writer(f)
            header = ['params', 'train accuracy', 'train loss', 'validation accuracy', 'validation loss']
            writer.writerow(header)

    """ dataset load """
    kmnist_dl = load.KMNISTDataLoader(img_resize=args.imgsize)
    train_x, train_y, valid_x, valid_y = kmnist_dl.load('../input/')
    train_x, train_y, valid_x, valid_y = load.Preprocessor().transform(train_x, train_y, valid_x, valid_y)

    train_generator, valid_generator = load.mygenerator(args, train_x, train_y, valid_x, valid_y, label_num)

    """ define hyper parameters """
    base_lr = 0.001
    lr_decay_rate = 1 / 3
    lr_steps = 4
    swa = SWA(para_path+'/swa.h5', args.epochs - 40)
    csv_logger = CSVLogger( para_path + '/log.csv', separator=',')
    callbacks = [ csv_logger, swa]

    def lr_schedule(epoch):
        lrate = base_lr
        if epoch > 75:
            lrate *= 0.2
        if epoch > 100:
            lrate *= 0.2
        if epoch > 150:
            lrate *= 0.2
        return lrate

    """ build model """
    if args.model == 'prot4':
        model = prot4_SE(args)
    elif args.model == 'wrn':
        # select_model = model.wrn_net(args.imgsize)
        input_dim = (args.imgsize, args.imgsize, 1)
        model = create_wide_residual_network(input_dim, N=args.wrn_n, k=args.wrn_k, se_module=args.se, NeXt=args.next)
    else:
        raise SyntaxError("please select model")
    # model = multi_gpu_model(select_model, gpus=gpu_count)
    model.summary()
    # tools.model_plot(model, para_str)

    """ select optimizer """
    if args.opt == 'sgd':
        print('--- optimizer: SGD ---')
        opt = SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True)
        callbacks.append(LearningRateScheduler(lr_schedule))
    elif args.opt == 'rms':
        print('--- optimizer: RMSprop ---')
        opt = rmsprop(lr=0.001, decay=1e-6)
        callbacks.append(LearningRateScheduler(lr_schedule))
    elif args.opt == 'rmsgraves':
        print('--- optimizer: RMSpropGraves ---')
        opt = RMSpropGraves(lr=0.001, decay=1e-6)
        callbacks.append(LearningRateScheduler(lr_schedule))
    elif args.opt == 'adam':
        print('--- optimizer: adam ---')
        opt =  Adam()
    elif args.opt == 'adabound':
        print('--- optimizer: adabound ---')
        opt = AdaBound(lr=1e-3, final_lr=0.1, amsbound=False)

    loss = keras.losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    """ model train """
    history = model.fit_generator(train_generator,
                        steps_per_epoch = 60000//args.batchsize,
                        epochs=args.epochs,
                        callbacks=callbacks)
    """ plot learning history """
    # tools.plot_history(history, para_str, para_path)
    """ save model """
    model.save(para_path + '/model.h5')

    """ evaluate model """
    train_score = model.evaluate(train_x, train_y)
    validation_score = model.evaluate(valid_x, valid_y)

    print('Train loss :', train_score[0])
    print('Train accuracy :', train_score[1])
    # print('validation loss :', validation_score[0])
    # print('validation accuracy :', validation_score[1])

    """ logging score """
    with open('../train_log/log.csv', 'a') as f:
        data = [para_str, train_score[1], train_score[0]]
        writer = csv.writer(f)
        writer.writerow(data)

    #%% submit file 
    import pandas as pd
    test_x = np.load('../input/kmnist-test-imgs.npz')['arr_0']

    def resize(imgs, imgsize):
        resized_imgs = []
        for im in imgs:
            resized_img = cv2.resize(im, (imgsize, imgsize))
            resized_imgs.append(resized_img)
        resized_imgs = np.array(resized_imgs)
        return resized_imgs
    
    test_x = resize(test_x, args.imgsize)
    print('resized test_x shape:', test_x.shape)
    """ convert images """
    test_x = test_x[:, :, :, np.newaxis].astype('float32') / 255.0

    print('--- predict data ---')
    predicts = np.argmax(model.predict(test_x), axis=1)

    print('predicts.shape: ', predicts.shape) # hope shape(10000, )
    print(predicts)
    print('predicts', predicts)
    submit = pd.DataFrame(data={"ImageId": [], "Label": []})
    submit.ImageId = list(range(1, predicts.shape[0]+1))
    submit.Label = predicts

    submit.to_csv("../output/" + para_str + ".csv", index=False)

    print('--- end train... ---')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train CNN model for classify')
    parser.add_argument('--epochs', '-e', type=int, default=300)
    parser.add_argument('--imgsize', '-s', type=int, default=32)
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--model', '-m', default='prot4',
                        help='prot4/resnet/wrn_net')
    parser.add_argument('--opt', '-o', default='rms',
                        help='sgd rms adabound')
    parser.add_argument('--se', '-q', default=False,
                        help='add se_module')
    parser.add_argument('--wrn_n', '-n', type=int, default=4)
    parser.add_argument('--wrn_k', '-k', type=int, default=10)
    parser.add_argument('--next', '-x', default=False)

    args = parser.parse_args()

    main(args)
