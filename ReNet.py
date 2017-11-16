"""
This file dedicates to configure models (Keras)
1. create model
2. load parameters
3. prune filters and fc nodes
4. save pruned model
"""
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import adam
import numpy as np
from VGG_conv_settings import VGG_conv_03p, VGG_conv_05p, VGG_conv_10p, VGG_conv_12p, VGG_conv_15p, VGG_conv_20p, \
    VGG_conv_25p, \
    VGG_conv_30p, VGG_conv_40p, VGG_conv_50p, VGG_conv_75p, VGG_conv_100

idx_conv_layer_VGG = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17]  # indices conv layers VGG
idx_fc_layer_VGG = [20, 21]  # indices fc layers VGG
model_path = '/home/biyi/PycharmProjects/MSDNet/models/'  # path to model zoo


class VGG_full(object):
    """
    VGG 100% filter-fc model
    [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        # create model
        self.model = create_model(VGG_conv_100, weights)
        # load weights (or None)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_75p(object):
    """
    VGG 75% model, pruned filters and fc
    [48, 64, 128, 128, 256, 256, 256, 384, 384, 384, 384, 384, 384]
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        # create model
        self.model = create_model(VGG_conv_75p, weights)
        # load weights (or None)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_50p(object):
    """
    VGG 50% model, pruned filters and fc.
    [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        # create model
        self.model = create_model(VGG_conv_50p, weights)
        # load weights (or None)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_40p(object):
    """
    VGG 40% model, pruned filters and fc.
    [32, 48, 96, 112, 224, 224, 224, 224, 224, 224, 224, 224, 224]
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_40p, weights, lr=0.000035)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_30p(object):
    """
    VGG 30% model, pruned filters and fc.
    [32, 32, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_30p, weights, lr=0.00035)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_25p(object):
    """
    VGG 25% model, pruned filters and fc.
    [32, 32, 64, 64, 96, 96, 96, 96, 96, 96, 96, 96, 96]
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_25p, weights, lr=0.00035)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_20p(object):
    """
    VGG 20% model, pruned filters and fc.
    [24, 32, 64, 64, 96, 96, 80, 80, 80, 80, 80, 80, 80]
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_20p, weights, lr=0.0003)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_15p(object):
    """
    VGG 25% model, pruned filters and fc.
    [16, 32, 64, 64, 80, 80, 80, 72, 72, 72, 72, 72, 72]
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_15p, weights, lr=0.0003)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_12p(object):
    """
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_12p, weights, lr=0.00028)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_10p(object):
    """
    """

    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_10p, weights, lr=0.00028)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_05p(object):
    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_05p, weights, lr=0.0002)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


class VGG_03p(object):
    def __init__(self, weights=None):
        """
        weights can be in multiple forms: None (NoneType), model.h5 (str), numpy array list (list)
        :param weights:
        """
        self.model = create_model(VGG_conv_03p, weights, lr=0.0002)

    def get_model(self):
        return self.model

    def find_filter_prune(self, trim_setting):
        return trim_helper(self.model, trim_setting)


def create_model(parameter_setting, weights, lr=0.00005):
    # model initialization
    input_shape = (224, 224, 3)
    img_input = Input(shape=input_shape)
    x = Conv2D(parameter_setting[0], (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input)
    x = Conv2D(parameter_setting[1], (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(parameter_setting[2], (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(parameter_setting[3], (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(parameter_setting[4], (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(parameter_setting[5], (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(parameter_setting[6], (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(parameter_setting[7], (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(parameter_setting[8], (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(parameter_setting[9], (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(parameter_setting[10], (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(parameter_setting[11], (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(parameter_setting[12], (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(10, activation='softmax', name='predictions')(x)
    model = Model(inputs=img_input, outputs=x)

    opt = adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if type(weights) == str:
        model.load_weights(model_path + weights)
    elif type(weights) == list:
        k = 0
        for layer in [model.layers[i] for i in idx_conv_layer_VGG + idx_fc_layer_VGG]:
            layer.set_weights(weights[k])
            k += 1

    print('Model Load Successful: ' + (weights if type(weights) == str else 'From Numpy List') + '\n')
    return model


def trim_helper(model, trim_setting):
    """
    find the indices of filter that have least l1 norm
    the number of pruned filters of each layer is determined by trim_setting
    :param trim_setting:
    :return: the trimmed weights as lists [[w0, b0], [w1, b1], ...]
    """
    conv_layers = [model.layers[i] for i in idx_conv_layer_VGG]
    w = [conv_layers[i].get_weights()[0] for i in range(len(idx_conv_layer_VGG))]
    b = [conv_layers[i].get_weights()[1] for i in range(len(idx_conv_layer_VGG))]

    w_fc1 = model.layers[idx_fc_layer_VGG[0]].get_weights()[0]
    num_out_flatten = w_fc1.shape[0]
    w_fc1 = np.reshape(w_fc1.T, (512, 7, 7, num_out_flatten // 49))
    # first transpose [18816, 512] to [512, 18816]
    # unflatten to find the slice to prune [512, 7, 7, 384]
    # 7 x 7 are hard-coded (determined by pooling strategy of the network)
    b_fc1 = model.layers[idx_fc_layer_VGG[0]].get_weights()[1]
    w_fc2 = model.layers[idx_fc_layer_VGG[1]].get_weights()[0]
    b_fc2 = model.layers[idx_fc_layer_VGG[1]].get_weights()[1]

    for i, layer in enumerate(conv_layers):
        """
        Take care of filters-bias each layer. Also take care the next layer
        """
        num_filter = w[i].shape[3]  # original filter number, e.g., 48
        num_trim_filter = trim_setting[i]  # filter number after trim, e.g., 32
        norml1 = [np.sum(np.fabs(w[i][:, :, :, idx])) for idx in
                  range(num_filter)]  # calculate the filter rank using l1 norm

        # iteratively find the minimum and delete wrt norml1, w and b.
        # do it (num_filter - num_trim_filter) times
        for j in range(num_filter - num_trim_filter):  # e.g., 64 - 48 = 16 filters to be trimmed
            pos = np.argmin(norml1)
            norml1 = np.delete(norml1, pos, axis=0)
            w[i] = np.delete(w[i], pos, axis=3)
            b[i] = np.delete(b[i], pos, axis=0)

            # fix the weight next layer
            if i < len(conv_layers) - 1:  # all conv layers
                w[i + 1] = np.delete(w[i + 1], pos, axis=2)
            else:  # first fc layer, the rearranged
                w_fc1 = np.delete(w_fc1, pos, axis=3)

    w_fc1 = np.reshape(w_fc1, (512, -1)).T
    w.append(w_fc1)
    w.append(w_fc2)
    b.append(b_fc1)
    b.append(b_fc2)
    print('\nTrimmed Successful: ')
    print('Expected Out (only print out filter numbers): ')
    print(trim_setting)
    print('We Get: ')
    print('\n'.join(str(x) for x in [w[i].shape for i in range(len(w))]))

    return [(w[i], b[i]) for i in range(len(w))]


def main_100to75():
    vgg_full_model = VGG_full(weights='VGG16-E100-9336-1fc512-11epoch.h5')
    w = vgg_full_model.find_filter_prune(trim_setting=VGG_conv_75p)
    keras.backend.clear_session()
    vgg_75p_model = VGG_75p(weights=w)
    # vgg_75p_model.get_model().save('VGG16-S75p-1fc512-0epoch.h5')


def main_75to50():
    vgg_75p_model = VGG_75p(weights='VGG16-E75p-9291-1fc512-3epoch.h5')
    w = vgg_75p_model.find_filter_prune(trim_setting=VGG_conv_50p)
    keras.backend.clear_session()
    vgg_50p_model = VGG_50p(weights=w)
    # vgg_50p_model.get_model().save('VGG16-S50p-1fc512-0epoch.h5')


def main_50to40():
    vgg_50p_model = VGG_50p(weights='VGG16-E50p-9244-1fc512-10epoch.h5')
    w = vgg_50p_model.find_filter_prune(trim_setting=VGG_conv_40p)
    keras.backend.clear_session()
    vgg_40p_model = VGG_40p(weights=w)
    vgg_40p_model.get_model().save('VGG16-S40p-1fc512-0epoch.h5')


def main_40to30():
    vgg_40p_model = VGG_40p(weights='VGG16-E40p-9364-1fc512-10epoch.h5')
    w = vgg_40p_model.find_filter_prune(trim_setting=VGG_conv_30p)
    keras.backend.clear_session()
    vgg_30p_model = VGG_30p(weights=w)
    vgg_30p_model.get_model().save('VGG16-S30p-1fc512-0epoch.h5')


def main_30to25():
    vgg_30p_model = VGG_30p(weights='VGG16-E30p-8902-1fc512-10epoch.h5')
    w = vgg_30p_model.find_filter_prune(trim_setting=VGG_conv_25p)
    keras.backend.clear_session()
    vgg_25p_model = VGG_25p(weights=w)
    vgg_25p_model.get_model().save('VGG16-S25p-1fc512-0epoch.h5')


def main_25to20():
    vgg_25p_model = VGG_25p(weights='VGG16-E25p-8779-1fc512-20epoch.h5')
    w = vgg_25p_model.find_filter_prune(trim_setting=VGG_conv_20p)
    keras.backend.clear_session()
    vgg_20p_model = VGG_20p(weights=w)
    vgg_20p_model.get_model().save('models/VGG16-S20p-1fc512-0epoch.h5')


def main_20to15():
    vgg_20p_model = VGG_20p(weights='VGG16-E20p-8794-1fc512-20epoch.h5')
    w = vgg_20p_model.find_filter_prune(trim_setting=VGG_conv_15p)
    keras.backend.clear_session()
    vgg_15p_model = VGG_15p(weights=w)
    vgg_15p_model.get_model().save('models/VGG16-S15p-1fc512-0epoch.h5')


def main_15to12():
    vgg_15p_model = VGG_15p(weights='VGG16-E15p-8770-1fc512-12epoch.h5')
    w = vgg_15p_model.find_filter_prune(trim_setting=VGG_conv_12p)
    keras.backend.clear_session()
    vgg_12p_model = VGG_12p(weights=w)
    vgg_12p_model.get_model().save('models/VGG16-S12p-1fc512-0epoch.h5')


def main_12to10():
    vgg_12p_model = VGG_12p(weights='VGG16-E12p-8814-1fc512-20epoch.h5')
    w = vgg_12p_model.find_filter_prune(trim_setting=VGG_conv_10p)
    keras.backend.clear_session()
    vgg_10p_model = VGG_10p(weights=w)
    vgg_10p_model.get_model().save('models/VGG16-S10p-1fc512-0epoch.h5')


def main_10to05():
    vgg_10p_model = VGG_10p(weights='VGG16-E10p-8708-1fc512-20epoch.h5')
    w = vgg_10p_model.find_filter_prune(trim_setting=VGG_conv_05p)
    keras.backend.clear_session()
    vgg_05p_model = VGG_05p(weights=w)
    vgg_05p_model.get_model().save('models/VGG16-S05p-1fc512-0epoch.h5')


def main_05to03():
    vgg_05p_model = VGG_05p(weights='VGG16-E05p-7487-1fc512-20epoch.h5')
    w = vgg_05p_model.find_filter_prune(trim_setting=VGG_conv_03p)
    keras.backend.clear_session()
    vgg_03p_model = VGG_03p(weights=w)
    vgg_03p_model.get_model().save('models/VGG16-S03p-1fc512-0epoch.h5')


def flop_cal(trim_setting, origin_setting=VGG_conv_100):
    """
    calculate the remaining FLOP of each layer a trim_setting network to an origin_setting
    :param origin_setting: original model conv layer setting, e.g., VGG_conv_100
    :param trim_setting: trimmed model conv layer setting, e.g., VGG_conv_75p
    :return: layer by layer percentage 0 ~ 1
    """
    origin_setting = [1] + origin_setting
    trim_setting = [1] + trim_setting
    results = [(trim_setting[i] / origin_setting[i] * trim_setting[i - 1] / origin_setting[i - 1]) for i in
               range(1, 14)]
    print(results)


if __name__ == '__main__':
    # VGG_05p().model.summary()
    # main_10to05()
    # flop_cal(VGG_conv_05p)
    # a = VGG_05p(weights='VGG16-E05p-8358-1fc512-90epoch.h5').find_filter_prune(trim_setting=VGG_conv_05p)
    # print('ok')
    # main_05to03()
    VGG_03p().model.summary()
