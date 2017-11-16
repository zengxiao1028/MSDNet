"""
Keras main file. Preprosessing and data generator + training entrance.
Biyi Fang
"""
from __future__ import print_function
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time
import numpy as np
import ReNet
import scipy
from numpy.random import shuffle
from sklearn.externals import joblib

batch_size = 40
num_classes = 10
epochs = 20
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'VGG16-E30p-1fc512-epoch_{}.h5'.format(time.time())


# def data_generator(data_gen):
#     while True:
#         x, y = data_gen.next()
#         yield x, [y] * num_classifiers


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

    return img


def resize(gen):
    """
    resize image to 224 x 224
    change to one-hot
    """
    while True:
        g = gen.next()
        img = np.array([scipy.misc.imresize(g[0][i, ...], (224, 224)) for i in range(batch_size)])
        y = np.zeros((batch_size, num_classes))
        y[np.arange(batch_size), np.squeeze(g[1])] = 1
        yield (img, y)


def main():
    # The data, shuffled and split between train and test sets:
    x_train_num = 50000
    x_test_num = 10000

    # model = ReNet.VGG_full(weights='VGG16-100p-9336-1fc512-11epoch.h5').get_model()
    # model = ReNet.VGG_50p(weights='VGG16-S75p-1fc512-0epoch.h5').get_model()
    model = ReNet.VGG_03p(weights='VGG16-S03p-1fc512-0epoch.h5').get_model()
    auto_save_callback = keras.callbacks.ModelCheckpoint(os.path.join('./models', model_name), monitor='val_acc',
                                                         mode='max', save_best_only=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

    print('Using real-time data augmentation.')

    x_train, y_train, _ = joblib.load('cifar10_train.pkl')
    x_test, y_test, _ = joblib.load('cifar10_test.pkl')

    train_datagen = ImageDataGenerator(
        preprocessing_function=prepro_fn,
        shear_range=0.1,
        horizontal_flip=True,
        rotation_range=30.,
        width_shift_range=0.1,
        height_shift_range=0.1)

    # train_generator = train_datagen.flow_from_directory(
    #     './dataset/cifar10/train/',
    #     target_size=(224, 224),
    #     batch_size=batch_size,
    #     class_mode='categorical')
    #
    # validation_generator = test_datagen.flow_from_directory(
    #     './dataset/cifar10/test/',
    #     target_size=(224, 224),
    #     batch_size=batch_size,
    #     class_mode='categorical')

    train_generator = resize(train_datagen.flow(x_train, y_train,
                                                batch_size=batch_size))

    test_datagen = ImageDataGenerator(preprocessing_function=prepro_fn)

    validation_generator = resize(test_datagen.flow(x_test, y_test,
                                                    batch_size=batch_size))

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=x_train_num // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=x_test_num // batch_size,
                        callbacks=[auto_save_callback, tensorboard],
                        max_queue_size=50)

    # Evaluate model with test data set and share sample prediction results
    evaluation = model.evaluate_generator(validation_generator,
                                          steps=x_test_num // batch_size)

    print('Model Accuracy = %.4f' % (evaluation[1]))


if __name__ == '__main__':
    main()
