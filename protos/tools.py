# coding:utf-8
"""
学習の予備で使ってるいろんな関数

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import load
import tqdm

# test time augmentation
def tta(model, test_x, batch_size):
    tta_steps = 10
    predictions = []

    test_datagen = load.MyImageDataGenerator(shear_range=0.2,
                                                zoom_range=0.1,
                                                width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                rotation_range=15,
                                                random_erasing=True)
    test_generator = test_datagen.flow(test_x, batch_size=batch_size, shuffle=False)

    for i in tqdm(range(tta_steps)):
        preds = model.predict_generator(test_generator, steps=test_x.shape[0]/batch_size)
        predictions.append(preds)

    pred = np.mean(predictions, axis=0)
    return pred


def plot_history(history, parastr, path):

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.xlabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig( path + '/accuracy.png')
    plt.close()

    # lossの履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.xlabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.savefig( path + '/loss.png')
    plt.close()

 # 混同行列のヒートマップをプロット
def print_cmx(y_true, y_pred, parastr):
    cmx_data = confusion_matrix(y_true, y_pred, labels=classes)

    df_cmx = pd.DataFrame(cxm_data, index=labels, columns=classes)

    plt.figure(figsize = (10, 7))
    sn.heatmap(df_cmx, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("predict classes")
    plt.ylabel("true classes")
    plt.savefig('./train_log/' + parastr + '/confution_mx.png')
    plt.close()
    