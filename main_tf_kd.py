import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import ReNet_tf_model_kd
import ReNet
import keras
import scipy
from sklearn.externals import joblib
from VGG_conv_settings import VGG_conv_05p, VGG_conv_10p, VGG_conv_12p, VGG_conv_15p, VGG_conv_20p, VGG_conv_25p, \
    VGG_conv_30p, VGG_conv_40p, VGG_conv_75p, VGG_conv_100

batch_size = 40
num_classes = 10
epochs = 100
data_augmentation = True


def prepro_fn(img):
    """
    VGG style preprocessing. there is NO normalization involved
    :param img:
    :return:
    """
    img = np.array(img).astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    # img = scipy.misc.imresize(img, (224, 224))
    return img


def resize(gen):
    """
    resize image to 224 x 224
    """
    while True:
        g = gen.next()
        img = np.array([scipy.misc.imresize(g[0][i, ...], (224, 224)) for i in range(batch_size)])
        yield (img, g[1])


def main():
    # The data, shuffled and split between train and test sets:
    x_train_num = 50000
    x_test_num = 10000
    w_05p = ReNet.VGG_05p(weights='VGG16-E05p-8358-1fc512-90epoch.h5').find_filter_prune(trim_setting=VGG_conv_05p)
    keras.backend.clear_session()
    model = ReNet_tf_model_kd.VGG(conv_setting=VGG_conv_05p, recover_setting=VGG_conv_10p, weights=w_05p)

    x_train, y_train, y_train_soft = joblib.load('cifar10_train.pkl')
    y_train = np.concatenate((y_train, y_train_soft), axis=1)  # 50000 x 513 (1 + 512)

    x_test, y_test, y_test_soft = joblib.load('cifar10_test.pkl')
    y_test = np.concatenate((y_test, np.squeeze(y_test_soft)), axis=1)  # 10000 x 513 (1 + 512)

    print('Using real-time data augmentation.')

    train_datagen = ImageDataGenerator(
        preprocessing_function=prepro_fn,
        shear_range=0.1,
        horizontal_flip=True,
        rotation_range=30.,
        width_shift_range=0.1,
        height_shift_range=0.1)

    train_generator = resize(train_datagen.flow(x_train, y_train,
                                                batch_size=batch_size))

    test_datagen = ImageDataGenerator(preprocessing_function=prepro_fn)

    validation_generator = resize(test_datagen.flow(x_test, y_test,
                                                    batch_size=batch_size))

    # model.validate(validation_generator,
    #                validation_steps=x_test_num // batch_size)

    model.train(train_generator,
                validation_generator,
                epochs=epochs,
                train_steps=x_train_num // batch_size,
                validation_steps=x_test_num // batch_size)


if __name__ == '__main__':
    main()
