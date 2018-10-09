import numpy as np
import os, sys
import json
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2grey

DATADIR = '../data/'
IMG_SIZE = 128  # unet uses 512x512

def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(True)
    X = (X - X.mean(0)) / (X.std(0)+1e-8)
    return X, y

def load_wine():
    from sklearn.datasets import load_wine
    X, y = load_wine(True)
    y[y == 2] = 1
    X = (X - X.mean(0)) / (X.std(0)+1e-8)
    return X, y

def _load_images(dirname):
    img_file = os.path.join(dirname, 'X.npy')
    if os.path.exists(img_file):
        X = np.load(img_file)
    else:
        imgs_dirname = os.path.join(dirname, 'imgs', 'seg')
        imgs_files = sorted(os.listdir(imgs_dirname))
        imgs = []
        for i, filename in enumerate(imgs_files):
            sys.stdout.write('\r%s/imgs: %4.1f%%' % (dirname, 100*i/len(imgs_files)))
            img = imread(os.path.join(imgs_dirname, filename))
            img = resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', anti_aliasing=True)
            imgs.append(img)
        X = np.array(imgs, np.float32)
        np.save(img_file, X)
    mask_file = os.path.join(dirname, 'y.npy')
    if os.path.exists(mask_file):
        y = np.load(mask_file)
    else:
        masks_dirname = os.path.join(dirname, 'masks', 'seg')
        masks_files = sorted(os.listdir(masks_dirname))
        masks = []
        for i, filename in enumerate(masks_files):
            sys.stdout.write('\r%s/masks: %4.1f%%' % (dirname, 100*i/len(masks_files)))
            mask = imread(os.path.join(masks_dirname, filename), True)
            mask = resize(mask, (IMG_SIZE, IMG_SIZE), mode='constant', anti_aliasing=True)
            masks.append(mask)
        y = np.array(masks)
        y = np.round(y[:, :, :, np.newaxis]).astype(np.int32)
        np.save(mask_file, y)
    sys.stdout.write('\r                                                              \r')
    return X, y

def load_images(dataset):
    dirname = os.path.join(DATADIR, dataset)
    with open(dirname + '.json') as f:
        aug = json.load(f)
    return _load_images(os.path.join(dirname, 'train')), \
        _load_images(os.path.join(dirname, 'validation')), \
        _load_images(os.path.join(dirname, 'test')), \
        aug

def load_csv(filename):
    data = np.loadtxt('../data/%s' % dataset, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    X = (X - X.mean(0)) / (X.std(0)+1e-8)
    return X, y
