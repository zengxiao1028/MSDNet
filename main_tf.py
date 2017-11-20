from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import ReNet_tf_model
import ReNet
import keras
from VGG_conv_settings import VGG_conv_05p, VGG_conv_10p, VGG_conv_12p, VGG_conv_15p, VGG_conv_20p, VGG_conv_25p, \
    VGG_conv_30p, VGG_conv_40p, VGG_conv_75p, VGG_conv_100

batch_size = 40
num_classes = 10
epochs = 1000
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
    # img = (img / 255. - 0.5) * 2.0
    return img

from keras import backend as K
from keras.engine.topology import Layer
class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def main():
    # The data, shuffled and split between train and test sets:
    x_train_num = 50000
    x_test_num = 10000
    w_05p = ReNet.VGG_05p(weights='VGG16-E05p-8358-1fc512-90epoch.h5').find_filter_prune(trim_setting=VGG_conv_05p)
    keras.backend.clear_session()
    model = ReNet_tf_model.VGG(conv_setting=VGG_conv_05p, recover_setting=VGG_conv_10p, weights=w_05p)

    print('Using real-time data augmentation.')

    train_datagen = ImageDataGenerator(
        preprocessing_function=prepro_fn,
        shear_range=0.1,
        horizontal_flip=True,
        rotation_range=30.,
        width_shift_range=0.1,
        height_shift_range=0.1)

    test_datagen = ImageDataGenerator(preprocessing_function=prepro_fn)

    train_generator = train_datagen.flow_from_directory(
        './dataset/cifar10/train/',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        './dataset/cifar10/test/',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    # model.validate(validation_generator,
    #                validation_steps=x_test_num // batch_size)

    model.train(train_generator,
                validation_generator,
                epochs=epochs,
                train_steps=x_train_num // batch_size,
                validation_steps=x_test_num // batch_size)


if __name__ == '__main__':
    main()
