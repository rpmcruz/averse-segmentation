from keras import models, layers, utils, regularizers


def create_simple(X, depth):
    input_layer = x = layers.Input(X.shape[1:])
    for _ in range(depth):
        x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(input_layer, x)

def create_unet(input_shape, depth, rep_layers=2, initial_filters=32, kernel_size=3, activation='relu', dropout=True):
    if activation == 'leakyrelu':
        act = lambda: layers.LeakyReLU(0.1)
    else:
        act = lambda: layers.Activation(activation)

    # the unet original paper used initial_filters=64
    # filters are increased by a power of two
    x = input_layer = layers.Input(input_shape)

    # encoding layers
    l = []
    for i in range(depth):
        filters = initial_filters * (2 ** i)
        for j in range(rep_layers):
            x = layers.Conv2D(filters, kernel_size, kernel_initializer='he_normal', padding='same')(x)
            x = act()(x)
            if j == 0 and dropout:
                x = layers.Dropout(0.1*(i+1))(x)
        l.append(x)
        x = layers.MaxPooling2D()(x)

    # middle layers
    filters = initial_filters * (2**depth)
    for j in range(rep_layers):
        x = layers.Conv2D(filters//(2**j), kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = act()(x)
        if j == 0 and dropout:
            x = layers.Dropout(0.1*depth)(x)

    # decoding layers (featuring skip-layers)
    for i in range(depth):
        filters = initial_filters * (2 ** (depth-i-1))
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([l[-i-1], x])
        for j in range(rep_layers):
            x = layers.Conv2D(filters//(2**j), kernel_size, kernel_initializer='he_normal', padding='same')(x)
            x = act()(x)
            if j == 0 and dropout:
                x = layers.Dropout(0.1*(depth-i))(x)

    x = layers.Conv2D(1, 1, activation='sigmoid')(x)
    return models.Model(input_layer, x)
