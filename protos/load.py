# coding:utf-8
""" load kuzushi-ji data """

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

class KMNISTDataLoader(object):
    """
    Example
    >>> kmnist_dl = KMINSTDataloader()
    >>> datapath = "./input/"
    >>> train_imgs, train_lbls, validation_imgs, train_lbls = kmist_dl.load(datapath)
    """
    def __init__(self, validation_size: float=0.15):
        self._basename_list = [ 'kmnist-train-imgs.npz', 'kmnist-train-labels.npz']
        self.validation_size = validation_size
    
    def load(self, datapath: str, random_seed: int=13) -> np.ndarray:
        filename_list = self._make_filenames(datapath)

        data_list = [np.load(filename)['arr_0'] for filename in filename_list]
        all_imgs, all_lbls = data_list

        # shuffle data
        np.random.seed(random_seed)
        perm_idx = np.random.permutation(len(all_imgs))
        all_imgs = all_imgs[perm_idx]
        all_lbls = all_lbls[perm_idx]

        # split train & validation
        validation_num = int(len(all_lbls)*self.validation_size)

        val_x = all_imgs[:validation_num]
        val_y = all_lbls[:validation_num]

        train_x = all_imgs[validation_num:]
        train_y = all_lbls[validation_num:]

        return train_x, train_y, val_x, val_y
    
    def _make_filenames(self, datapath: str) -> list:
        filename_list = [os.path.join(datapath, basename) for basename in self._basename_list]
        return filename_list

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
class MyImageDataGenerator(ImageDataGenerator):
    def __init__(self, featurewise_center=False, samplewise_center=False,
                featurewise_std_normalization=False, sampleWise_std_normalization=False,
                zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0,
                height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
                channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
                vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, 
                validation_split=0.0, random_crop=None, mix_up_alpha=0.0, random_erasing=False):

        # 親クラスのコンストラクタ
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization,
                        sampleWise_std_normalization, zca_whitening, zca_epsilon, rotation_range,
                        width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range,
                        channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale,
                        preprocessing_function, data_format, validation_split)
        # 拡張処理のパラメータ
        assert mix_up_alpha >= 0.0
        self.mix_up_alpha = mix_up_alpha
        assert random_crop == None or len(random_crop) == 2
        self.random_crop_size = random_crop

        self.random_erasing = random_erasing

    """ Random Crop """
    def random_crop(self, original_img):
        # Note: image_data_format is 'channel_last'
        assert original_img.shape[2] == 3
        if original_img.shape[0] < self.random_crop_size[0] or original_img.shape[1] < self.random_crop_size[1]:
            raise ValueError(" はい" )
        
        height, width = original_img.shape[0], original_img.shape[1]
        dy, dx = self.random_crop_size
        x = np.random.randint(0, width -dx + 1)
        y = np.random.randint(0, height -dy + 1)
        return original_img[y:(y+dy), x:(x+dx), :]

    """ Mix up """
    def mix_up(self,x1, y1, x2, y2):
        # assert x1.shape[0] == y1.shape ==[0] == x2.shape[0] == y2.shape[0]
        batch_size = x1.shape[0]
        l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
        x_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)
        X = x1 * x_l + x2 * (1- x_l)
        Y = y1 * y_l + y2 * (1- y_l)
        return X, Y
    
    """ Random Erasing 
    https://www.kumilog.net/entry/numpy-data-augmentation
    """
    def random_eraser(self, original_img):
        image = np.copy(original_img)
        p=0.5
        s=(0.02, 0.4)
        r=(0.3, 3)

        # マスクするかしないか
        if np.random.rand() > p:
            return image

        # マスクする画素値をランダムで決める
        mask_value = np.random.random()

        h, w, _ = image.shape
        # マスクサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
        mask_area = np.random.randint(h * w * s[0], h* w * s[1])

        # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
        mask_aspect_ratio = np.random.rand() * r[1] + r[0]
        
        # マスクのサイズとアスペクト比からマスクの高さと幅を決める
        # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正
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
    
    def flow(self,
            x,
            y=None,
            batch_size=32,
            shuffle=True,
            sample_weight=None,
            seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            subset=None):
        # 親クラスのflow
        batches = super().flow(x,y,batch_size,shuffle,sample_weight,seed,save_to_dir,save_prefix,save_format,subset)

        # 拡張処理
        while True:
            batch_x, batch_y = next(batches)

            """ mix up """
            if self.mix_up_alpha > 0:
                while True:
                    batch_x_2, batch_y_2 = next(batches)
                    m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                    if m1 < m2:
                        batch_x_2 = batch_x_2[:m1]
                        batch_y_2 = batch_y_2[:m1]
                        break
                    elif m1 == m2:
                        break
                batch_x, batch_y = self.mix_up(batch_x, batch_y, batch_x_2, batch_y_2)
            
            """ random erasing """
            if self.random_erasing == True:
                x = np.zeros((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], 1))
                for i in range(batch_x.shape[0]):
                    x[i] = self.random_eraser(batch_x[i])
                batch_x = x
            
            yield(batch_x, batch_y)

def mygenerator(args, train_x, train_y, valid_x, valid_y, label_num):

    print('train_x.shape :', train_x.shape)
    print('valid_x.shape :', valid_x.shape)
    print('train_y.shape :', train_y.shape)
    print('valid_y.shape :', valid_y.shape)

    train_datagen = MyImageDataGenerator(shear_range=0.1,
                                            zoom_range=0.1,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            rotation_range=15,
                                            # mix_up_alpha=0.2,
                                            random_erasing=True)
    
    valid_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(train_x, train_y, batch_size=args.batchsize)
    valid_generator = valid_datagen.flow(valid_x, valid_y, batch_size=args.batchsize)

    return train_generator, valid_generator
if __name__ == '__main__':
    kmnist_dl = KMNISTDataLoader()
    datapath = "./input"
    # train_imgs, train_lbls, validation_imgs, train_lbls = kmist_dl.load(datapath)
    kmnist_dl.load(datapath)
    