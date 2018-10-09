import keras.backend as K
import tensorflow as tf

def tnr(y, yhat): # yhat=0 | y=0
    return K.sum((1-y) * (1-K.round(yhat))) / K.sum(1-y)

def fnr(y, yhat): # yhat=0 | y=1
    return K.sum(y * (1-K.round(yhat))) / K.sum(y)

def tpr(y, yhat): # yhat=1 | y=1
    return K.sum(y * K.round(yhat)) / K.sum(y)

def fpr(y, yhat):  # yhat=1 | y=0
    return K.sum((1-y) * K.round(yhat)) / K.sum(1-y)

########## APPROXIMATIONS FOR DERIVATIVES ###########

def fnr_approx(y, yhat): # yhat=0 | y=1
    return K.sum(y * (1-yhat)) / K.sum(y)

###############################

def crossentropy(y, yhat):
    yhat = K.clip(yhat, K.epsilon(), 1-K.epsilon())
    return -K.mean(y*K.log(yhat) + (1-y)*K.log(1-yhat))

def crossentropy_pos(y, yhat):  # used by ping-pong
    yhat = K.clip(yhat, K.epsilon(), 1-K.epsilon())
    return -K.sum(y*K.log(yhat)) / (K.sum(y)+K.epsilon())

def crossentropy_pingpong():
    alpha = K.ones([], 'float32')
    def fn(y, yhat):
        yhat = K.clip(yhat, K.epsilon(), 1-K.epsilon())
        loss1 = crossentropy(y, yhat)
        loss2 = crossentropy_pos(y, yhat)
        return alpha*loss1 + (1-alpha)*loss2
    return fn, alpha

def crossentropy_pinball(rho):  # this is like using a cost matrix
    def fn(y, yhat):
        yhat = K.clip(yhat, K.epsilon(), 1-K.epsilon())
        return -K.mean((1-rho)*y*K.log(yhat) + rho*(1-y)*K.log(1-yhat))
    return fn

###############################

def slice_averse(y, yhat, batchsize, posbatchsize):
    if len(yhat.shape) == 2:
        start_normal = [0, 0]
        size_normal = [batchsize, 1]
        start_pos = [batchsize, 0]
        size_pos = [posbatchsize, 1]
    else:
        start_normal = [0, 0, 0, 0]
        yhat_shape = yhat.shape.as_list()[1:]
        size_normal = [batchsize, *yhat_shape]
        start_pos = [batchsize, 0, 0, 0]
        size_pos = [posbatchsize, *yhat_shape]
    y_normal = K.slice(y, start_normal, size_normal)
    yhat_normal = K.slice(yhat, start_normal, size_normal)
    y_pos = K.slice(y, start_pos, size_pos)
    yhat_pos = K.slice(yhat, start_pos, size_pos)
    return (y_normal, yhat_normal), (y_pos, yhat_pos)

def squared_averse_term(rho):
    def fn(y, yhat):
        rho_hat = fnr_approx(y, yhat)
        return (rho_hat - rho) ** 2
    return fn

def relu_squared_averse_term(rho):
    def fn(y, yhat):
        rho_hat = fnr_approx(y, yhat)
        return K.relu(rho_hat - rho) ** 2
    return fn

def entropy_averse_term(rho):
    def fn(y, yhat):
        rho_hat = fnr_approx(y, yhat)
        return K.relu(-K.log(1-rho_hat+rho))
    return fn

def old_averse_term(rho):
    def fn(y, yhat):
        rho_hat = fnr_approx(y, yhat)
        return K.exp(rho_hat - rho) - 1
    return fn

def crossentropy_slice(alpha, term, batchsize, posbatchsize):
    def fn(y, yhat):
        (y, yhat), (y_pos, yhat_pos) = slice_averse(y, yhat, batchsize, posbatchsize)
        if term:
            return crossentropy(y, yhat) + alpha*term(y_pos, yhat_pos)
        return crossentropy(y, yhat)
    return fn

if __name__ == '__main__':  # TEST!
    data = [
        ('all correct', [0, 0, 1, 1], [0, 0, 1, 1]),
        ('all different', [1, 1, 0, 0], [0, 0, 1, 1]),
        ('mixed', [0, 0, 1, 1], [0.1, 0.4, 0.8, 0.9]),
        ('one-fn', [0, 0, 1, 1], [0, 0, 0, 1]),
    ]

    print('\n* sklearn')
    from sklearn.metrics import log_loss
    for name, y, yhat in data:
        print(name, log_loss(y, yhat))

    print('\n* keras')
    from keras import losses
    for name, y, yhat in data:
        print(name, K.eval(losses.binary_crossentropy(K.constant(y), K.constant(yhat))))

    print('\n* my functions')
    for fn in [crossentropy, tnr, fnr, tpr, fpr]:
        print('*', fn.__name__)
        for name, y, yhat in data:
            print(name, K.eval(fn(K.constant(y), K.constant(yhat))))

    print('\n* my averse term')
    for name, y, yhat in data:
        print(name, K.eval(averse_term(K.constant(y), K.constant(yhat))))

    print('\n* my averse term relu')
    for name, y, yhat in data:
        print(name, K.eval(averse_term_relu(K.constant(y), K.constant(yhat))))
