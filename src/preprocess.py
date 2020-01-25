import numpy as np
import cv2
from sklearn.decomposition import PCA


class ImagePreprocessor:
    def __init__(self, normalize, hog_tf, pca_tf, win_size=None, block_size=None,
                 block_stride=None, cell_size=None, nbins=None):
        self.normalize = normalize
        self.hog_tf = hog_tf
        self.pca_tf = pca_tf
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins

    def scale(self, x_train, x_test):
        # normalize
        if self.normalize:
            x_train = x_train / 255
            x_test = x_test / 255

        # hog
        if self.hog_tf:
            # reshape as 28 x 28
            pix = self.win_size[0]
            x_train = x_train.reshape((x_train.shape[0], pix, pix, -1))
            x_test = x_test.reshape((x_test.shape[0], pix, pix, -1))

            x_train = self.hog_transform(x_train)
            x_test = self.hog_transform(x_test)

            # flatten
            x_train = x_train.reshape((x_train.shape[0], -1))
            x_test = x_test.reshape((x_test.shape[0], -1))

        # pca
        if self.pca_tf:
            x_train = self.pca_transform(x_train)
            x_test = self.pca_transform(x_test)

        return x_train, x_test

    def hog_transform(self, imgs):
        hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride,
                                self.cell_size, self.nbins)
        features = []
        for img in imgs:
            features.append(hog.compute(img))
        return np.array(features)

    def pca_transform(self, imgs):
        pca = PCA(n_components=200)
        imgs = pca.fit_transform(imgs)

        return imgs


class TextPreprocessor:
    def __init__(self, d2v_tf):
        self.d2v_tf = d2v_tf


class Cifar10Preprocessor:
    def __init__(self, normalize):
        self.normalize = normalize
