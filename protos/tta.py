# coding:utf-8
import numpy as np
import keras
import cv2
import pandas as pd
from wideresnet import create_wide_residual_network

class TTA_predict():

    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.summary()

        self.imgsize = 32
        self.test_path = '../input/kmnist-test-imgs.npz'

        self.para_path = model_path.lstrip("./").rstrip(".h5")
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
    
    def predict(self):
        X = self.test_load()
        print(X.shape)

        pred = []
        cnt = 0
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
            cnt +=1
            if cnt > 10: break

        pred = np.array(pred)
        return pred
    
    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)

    def create_submit(self):
        predicts = np.argmax(self.predict(), axis=1)
        print(predicts)
        print('preditcs.shpae: ', predicts.shape) # shape(1000,)

        submit = pd.DataFrame(data={"ImageId": [], "Label": []})
        submit.ImageId = list(range(1, predicts.shape[0]+1))
        submit.Label = predicts

        print("../output/" + self.para_path + "_wTTA.csv")
        submit.to_csv("../output/" + self.para_path + "_wTTA.csv", index=False)

if __name__ == "__main__":
    model_path = './wrn_imgsize32_batchsize128_sgd_SEmodule_Truemodel.h5'
    tta = TTA_predict(model_path)
