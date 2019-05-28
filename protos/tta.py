# coding:utf-8
import numpy as np
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2
from wideresnet import create_wide_residual_network
import load

class TTA_predict():

    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path + '/model.h5', compile=False)
        self.model.summary()

        self.imgsize = 42
        self.tta_steps = 15
        self.test_path = '../input/kmnist-test-imgs.npz'

        self.para_path = model_path.lstrip("../train_log/")
        print(self.para_path)

        self.create_submit()
    
    def test_load(self):
        test_x = np.load(self.test_path)['arr_0']
        resized_imgs = []
        for im in test_x:
            resized_img = cv2.resize(im, (self.imgsize, self.imgsize))
            resized_imgs.append(resized_img)
        resized_imgs = np.array(resized_imgs)
        resized_imgs = resized_imgs[:, :, :, np.newaxis].astype('float32') / 255.0
        return resized_imgs
    
    def valid_load(self):
        kmnist_dl = load.KMNISTDataLoader(img_resize=self.imgsize)
        _, _, valid_x, valid_y = kmnist_dl.load('../input/')
        _, _, valid_x, valid_y = load.Preprocessor().transform(train_x, train_y, valid_x, valid_y)

        """ evaluate score """
        validation_score = self.model.evaluate(valid_x, valid_y)
        print('validation loss :', validation_score[0])
        print('validation accuracy :', validation_score[1])

    
    def random_eraser(self, original_img):
        image = np.copy(original_img)
        p = 0.5
        s = (0.02, 0.4)
        r = (0.3, 3)

        mask_value = np.random.random()
        h, w, _ = image.shape
        mask_area = np.random.randint(h * w * s[0], h * w * s[1])
        mask_aspect_ratio = np.random.rand() * r[1] + r[0]
        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))

        if mask_height > h-1:
            mask_height = h-1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w-1:
            mask_width = w-1
        
        top = np.random.randint(0, h-mask_height)
        left = np.random.randint(0, w-mask_width)
        bottom = top+mask_height
        right = left+mask_width

        image[top:bottom, left:right, :].fill(mask_value)

        return image
    def predict_wDataGenerator(self):
        X = self.test_load()
        print(X.shape)

        test_datagen =  ImageDataGenerator(
                            shear_range=0.1,
                            zoom_range = 0.1,
                            rotation_range = 10,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1)
        predictions = []

        for i in range(self.tta_steps):
            preds = self.model.predict_generator(
                            test_datagen.flow(X, batch_size=128, shuffle=False),
                            steps = len(X)/128)
            print(preds.shape)
            predictions.append(preds)
        
        pred = np.mean(predictions, axis=0)
        print(pred.shape)
        predicts = np.argmax(pred, axis=1)
        print(predicts)
        print('preditcs.shpae: ', predicts.shape) # shape(1000,)
        return predicts
    
    def predict(self):
        X = self.test_load()
        print(X.shape)

        pred = []
        for x_i in X:
            """ random erasing1 """
            x_p1 = self.random_eraser(x_i)
            p1 = self.model.predict(self._expand(x_p1[:,:,0]))
            # print('p1 predict: ', np.argmax(p1, axis=1))

            """ random erasing2 """
            x_p2 = self.random_eraser(x_i)
            p2 = self.model.predict(self._expand(x_p2[:,:,0]))

            """ original """
            x_i = self._expand(x_i[:,:,0])
            p0 = self.model.predict(x_i)

            p = (p0 * 0.6) + (p1 * 0.3) + (p2 * 0.3)
            print('final predict: ', np.argmax(p, axis=1))
            pred.append(p[0])

        pred = np.array(pred)
        return pred
    
    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)

    def create_submit(self):
        # predicts = np.argmax(self.predict(), axis=1)
        predicts = self.predict_wDataGenerator()

        submit = pd.DataFrame(data={"ImageId": [], "Label": []})
        submit.ImageId = list(range(1, predicts.shape[0]+1))
        submit.Label = predicts

        print('../output/wTTA_' + self.para_path + ".csv")
        submit.to_csv("../output/wTTA_" + self.para_path + ".csv", index=False)

if __name__ == "__main__":
    model_path = '../train_log/wrn_imgsize42_batchsize128_adabound_SEmodule_True'
    tta = TTA_predict(model_path)
