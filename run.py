import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--burnepochs', default=0, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=32, type=int)
parser.add_argument('--posbatchsize', default=64, type=int)
parser.add_argument('--rho', default=0.05, type=float)
parser.add_argument('--term', choices=['squared', 'relu_squared', 'entropy', 'old'])
parser.add_argument('--pingpong', action='store_true')
parser.add_argument('--sigma', default=0.5, type=float)
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--pinball', action='store_true')
parser.add_argument('--output', required=True)
parser.add_argument('--output-model')
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--activation', default='relu')
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--no-validation', action='store_true')
parser.add_argument('--disable-dropout', action='store_true')
parser.add_argument('--disable-augmentation', action='store_true')
args = parser.parse_args()

if not args.output.endswith('.csv'):
    sys.exit('output must be a .csv file')

# random seed
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

# allow memory grow
from keras import backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
backend.set_session(tf.Session(config=config))

# dataset

import mydatasets
if args.dataset.endswith('.csv'):
    X, y = mydatasets.load_csv(args.dataset)
elif hasattr(mydatasets, 'load_%s' % args.dataset):
    X, y = getattr(mydatasets, 'load_%s' % args.dataset)()
else:
    (Xtr, ytr), (Xval, yval), (Xts, yts), augfile = mydatasets.load_images(args.dataset)
    if args.no_validation:
        Xtr = np.r_[Xtr, Xval]
        ytr = np.r_[ytr, yval]
        Xval = Xtr
        yval = ytr

if 'X' in locals():  # split it ourselves
    if len(X.shape) == 2:  # if matrix
        from sklearn.model_selection import train_test_split
        y = y[:, np.newaxis]
        Xtr, Xhalf, ytr, yhalf = train_test_split(X, y, test_size=0.50, stratify=y)
        Xval, Xts, yval, yts = train_test_split(Xhalf, yhalf, test_size=0.50, stratify=yhalf)
    else:  # if image
        def train_test_split(X, y, test_size):
            ix = np.arange(len(X))
            np.random.shuffle(ix)
            i = int(len(X)*test_size)
            return X[ix[i:]], X[ix[:i]], y[ix[i:]], y[ix[:i]]
        if args.no_validation:
            Xtr, Xts, ytr, yts = train_test_split(X, y, 0.50)
            Xval = Xtr
            yval = ytr
        else:
            Xtr, Xhalf, ytr, yhalf = train_test_split(X, y, 0.50)
            Xval, Xts, yval, yts = train_test_split(Xhalf, yhalf, 0.50)

print('Dataset shapes:', Xtr.shape, ytr.shape, Xval.shape, yval.shape, Xts.shape, yts.shape)
if len(ytr.shape) == 2:
    print('== Distributions ==')
    print(np.bincount(ytr), np.bincount(yval), np.bincount(yts))

# model

from keras import callbacks, optimizers, preprocessing
from skimage.exposure import adjust_sigmoid
import mymodels
import mylosses
import mymetrics

BATCHES = len(Xtr) // args.batchsize

model = getattr(mymodels, 'create_%s' % args.model)(Xtr.shape[1:], args.depth, activation=args.activation, dropout=not args.disable_dropout)
model.summary()

# Since we have so few positives, I was forced to concatenate train and validation (not
# just use validation) when estimating false-negatives

contrast_stretching_perc = augfile['contrast_stretching_perc']

def preprocessing_function(x):
    if contrast_stretching_perc:
        contrast_p = np.random.uniform(
            1-contrast_stretching_perc, 1+contrast_stretching_perc)
        x = adjust_sigmoid(x, cutoff=0.5, gain=5*contrast_p, inv=False)
    return x

class MyImageDataGenerator(preprocessing.image.ImageDataGenerator):
    def __init__(self, ismask, contrast_stretching_perc=0, **args):
        if ismask:
            super().__init__(**args)
        else:
            super().__init__(**args, preprocessing_function=preprocessing_function)

if 'augfile' in locals() and not args.disable_augmentation:
    # tr
    Xtr_aug = MyImageDataGenerator(False, **augfile, fill_mode='constant')
    Xtr_aug = Xtr_aug.flow(Xtr, seed=123, batch_size=args.batchsize)
    ytr_aug = MyImageDataGenerator(True, **augfile, fill_mode='constant')
    ytr_aug = ytr_aug.flow(ytr, seed=123, batch_size=args.batchsize)
    tr_aug = zip(Xtr_aug, ytr_aug)
    # val
    Xval_aug = MyImageDataGenerator(False, **augfile, fill_mode='constant')
    Xval_aug = Xval_aug.flow(Xval, seed=124, batch_size=args.posbatchsize)
    yval_aug = MyImageDataGenerator(True, **augfile, fill_mode='constant')
    yval_aug = yval_aug.flow(yval, seed=124, batch_size=args.posbatchsize)
    val_aug = zip(Xval_aug, yval_aug)
    augmentation = True
else:
    augmentation = False

def entropy_gen():
    while True:
        if augmentation:
            xtr, ytr = next(tr_aug)
            xval, yval = next(val_aug)
            # Keras truncates sampling to available samples, but we always want to
            # force the same batchsize
            while len(xtr) < args.batchsize:
                xtr2, ytr2 = next(tr_aug)
                xtr = np.r_[xtr, xtr2][:args.batchsize]
                ytr = np.r_[ytr, ytr2][:args.batchsize]
            while len(xval) < args.posbatchsize:
                xval2, yval2 = next(val_aug)
                xval = np.r_[xval, xval2][:args.posbatchsize]
                yval = np.r_[yval, yval2][:args.posbatchsize]
            yield np.r_[xtr, xval], np.r_[ytr, yval]
        else:
            ix = np.random.choice(len(Xtr), args.batchsize, False)
            jx = np.random.choice(len(Xval), args.posbatchsize, False)
            yield np.r_[Xtr[ix], Xval[jx]], np.r_[Xtr[jx], Xval[jx]]

class MetricsCallback(callbacks.Callback):
    def __init__(self, metrics, tr, val, ts):
        self.metrics = metrics
        self.tr = tr
        self.val = val
        self.ts = ts

    def on_epoch_end(self, epoch, logs={}):
        for name, (X, y) in zip(('', 'val_', 'test_'), (self.tr, self.val, self.ts)):
            yhat = self.model.predict(X)
            for metric_name, metric in self.metrics:
                score = metric(y, yhat)
                logs[name + metric_name] = score

class FnrCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        yhat = self.model.predict(Xval)
        FN = mymetrics.fnr(yval, yhat)
        print('false negatives:', FN)
        if FN <= args.rho*(1-args.sigma):
            self.model.stop_training = True

class PingPongCallback(callbacks.Callback):
    def __init__(self, alpha):
        self.controlled = True
        self.alpha = alpha

    def on_batch_end(self, batch, logs=None):
        yhat = self.model.predict(Xval)
        rho_hat = mymetrics.fnr(yval, yhat)
        old_controlled = self.controlled
        self.controlled = \
            (self.controlled and rho_hat <= args.rho) or \
            (not self.controlled and rho_hat < args.rho*(1-args.sigma))
        print('batch %2d - FNR: %.3f - controlled? %d' % (batch, rho_hat, self.controlled))
        if old_controlled != self.controlled:
            backend.set_value(self.alpha, float(self.controlled))

metrics = [
    ('tnr', mymetrics.tnr),
    ('fnr', mymetrics.fnr),
    ('crossentropy', mymetrics.crossentropy),
    ('entropy_averse_term', mymetrics.entropy_averse_term(args.rho)),
]
mcb = MetricsCallback(metrics, (Xtr, ytr), (Xval, yval), (Xts, yts))

optimizer = getattr(optimizers, args.optimizer.title())()

def model_fit(*options1, **options2):
    if augmentation:
        return model.fit_generator(tr_aug, BATCHES, *options1, **options2, verbose=2,
            use_multiprocessing=True)
    else:
        return model.fit(Xtr, ytr, args.batchsize, *options1, **options2, verbose=2)

if args.burnepochs:
    model.compile(optimizer, mylosses.crossentropy)
    burnH = model_fit(args.burnepochs, callbacks=[mcb])

if args.warmup:
    model.compile(optimizer, mylosses.crossentropy, [mylosses.fnr])
    h1 = model_fit(args.epochs, callbacks=[mcb])
    model.compile(optimizers.SGD(1e-3), mylosses.crossentropy_pos, [mylosses.fnr])
    h2 = model_fit(
        args.epochs*2, callbacks=[mcb, FnrCallback()], initial_epoch=args.epochs)
    H = {key: h1.history[key] + h2.history[key] for key in h1.history.keys()}
elif args.pingpong:
    loss, alpha = mylosses.crossentropy_pingpong()
    model.compile(optimizer, loss, [mylosses.crossentropy_pos])
    h = model_fit(
        args.epochs, callbacks=[mcb, PingPongCallback(alpha)],
        initial_epoch=args.burnepochs)
    H = h.history
elif args.term:
    term = getattr(mylosses, '%s_averse_term' % args.term)(args.rho)
    loss = mylosses.crossentropy_slice(args.alpha, term, args.batchsize, args.posbatchsize)
    model.compile(optimizer, loss, [mylosses.crossentropy, term])
    h = model.fit_generator(
        entropy_gen(), BATCHES, args.epochs, verbose=2, callbacks=[mcb],
        initial_epoch=args.burnepochs)
    H = h.history
elif args.pinball:
    model.compile(optimizer, mylosses.crossentropy_pinball(args.rho))
    h = model_fit(args.epochs, callbacks=[mcb], initial_epoch=args.burnepochs)
    H = h.history
else:
    model.compile(optimizer, mylosses.crossentropy)
    h = model_fit(args.epochs, callbacks=[mcb], initial_epoch=args.burnepochs)
    H = h.history

if args.burnepochs:
    H = {key: burnH.history[key] + H[key] for key in burnH.history.keys()}

if args.output_model:
    model.save(args.output_model)

from sklearn.metrics import confusion_matrix
import pandas as pd

yp = model.predict(Xts)
yp = np.round(yp).astype(np.int32)
M = confusion_matrix(yts.flatten(), yp.flatten())
M = M.astype(float) / M.sum(1)
print('final confusion matrix:', M)
pd.DataFrame(H).to_csv(args.output, index_label='epoch')
