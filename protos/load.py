# coding:utf-8
""" load kuzushi-ji data """

import os
import numpy as np

class KMNISTDataLoader(object):
    """
    Example
    >>> kmnist_dl = KMINSTDataloader()
    >>> datapath = "./input/"
    >>> train_imgs, train_lbls, validation_imgs, train_lbls = kmist_dl.load(datapath)
    """
    def __init__(self, validation_size: float=0.2):
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


if __name__ == '__main__':
    kmnist_dl = KMNISTDataLoader()
    datapath = "./input"
    # train_imgs, train_lbls, validation_imgs, train_lbls = kmist_dl.load(datapath)
    kmnist_dl.load(datapath)
    