import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import layers
import tensorflow as tf
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.datasets import cifar10
import scipy, keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings, os
from keras.optimizers import adam
import keras.backend as K
import json, shutil, glob
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from collections import defaultdict
from keras.utils.layer_utils import count_params
from keras.applications.imagenet_utils import _obtain_input_shape, preprocess_input, decode_predictions
from keras.engine.topology import get_source_inputs

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

from keras.engine.topology import InputSpec
from keras.callbacks import Callback
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Reshape
from keras.models import load_model
from dataset.imagenet import imagenet_generator
from ResNet50 import ResNet50
from keras.layers import SeparableConv2D

def multile_output_generator(gen):
    while True:
        x, y = next(gen)
        yield x, [y]* 17

class EE_ResNet50(ResNet50):

    ee_names = ['max_pooling2d_1', 'activation_4','ee2b','ee2c','activation_11','ee3b','ee3c',
                'ee3d','activation_20','ee4b','ee4c','ee4d','ee4e', 'ee4f','activation_33','ee5b']

    def generate_ee_model(self,freeze=True, add_depthwise=False):

        def relu6(x):
            return K.relu(x, max_value=6)

        def _depthwise_conv_block(inputs, pointwise_conv_filters,
                                  depth_multiplier=1, strides=(1, 1), block_id=1):
            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

            x = inputs

            # x = Conv2D(64, (1, 1),
            #            padding='same',
            #            use_bias=False,
            #            strides=(1, 1))(x)
            # x = BatchNormalization(axis=channel_axis, name='conv_pre_pw_%d_bn' % block_id)(x)
            # x = Activation(relu6, name='conv_pre_pw_%d_relu' % block_id)(x)

            x = DepthwiseConv2D((3, 3),
                                padding='same',
                                depth_multiplier=depth_multiplier,
                                strides=strides,
                                use_bias=False)(x)
            x = BatchNormalization(axis=channel_axis)(x)
            x = Activation(relu6)(x)

            x = Conv2D(pointwise_conv_filters, (1, 1),
                       padding='same',
                       use_bias=False,
                       strides=(1, 1))(x)
            x = BatchNormalization(axis=channel_axis)(x)
            return Activation(relu6)(x)

        def _create_ouput(x, ee_name):

            if add_depthwise:
                x = SeparableConv2D(64,(3,3),activation='relu')(x)

            x = GlobalAveragePooling2D()(x)
            x = Reshape((1, 1, -1), )(x)
            x = Conv2D(self.config['model']['classes'], (1, 1),
                       padding='same')(x)
            x = Activation('softmax')(x)
            x = Reshape((self.config['model']['classes'],), name=ee_name + '_output')(x)
            return x

        if freeze:
            layers = self.model.layers
            for layer in layers:
                layer.trainable = False

        print([layer.name for layer in layers])
        ee_layers = [ layer for layer in self.model.layers if layer.name in self.ee_names]
        print([layer.name for layer in ee_layers])

        ee_outputs = []
        for idx,ee_layer in enumerate(ee_layers):
            x = _create_ouput(
                ee_layer.output,'ee'+str(idx+1))
            ee_outputs.append(x)

        model = Model(inputs=self.model.inputs, outputs=ee_outputs + [self.model.output] )

        self.model = model



    def train_hongkong_car(self, training_save_dir='./resnet/hongkong_car/ee_results', epochs=None):
        epochs = epochs if epochs is not None else self.config['train']['epochs']

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=False,
            rotation_range=15.,
            width_shift_range=0.1,
            height_shift_range=0.1)

        train_generator = train_datagen.flow_from_directory(
            './dataset/hongkong_car/train/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/hongkong_car/test/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        #### comppile model ########
        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        #### prepare training ########
        name = self.model.name
        os.makedirs(training_save_dir, exist_ok=True)
        result_counter = len(
            [log for log in os.listdir(training_save_dir) if name == '_'.join(log.split('_')[:-1])]) + 1
        saved_dir = os.path.join(training_save_dir, name + '_' + str(result_counter))
        os.makedirs(saved_dir, exist_ok=True)
        shutil.copyfile(self.config_path, os.path.join(saved_dir, self.config_path.split('/')[-1]))
        best_checkpoint = ModelCheckpoint(os.path.join(saved_dir, self.model.name + '_best.h5'),
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='max',
                                          period=1)

        checkpoint = ModelCheckpoint(os.path.join(saved_dir, self.model.name + '.h5'),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=False,
                                     mode='max',
                                     period=1)

        tensorboard = TensorBoard(log_dir=saved_dir,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)

        self.model.fit_generator(generator=multile_output_generator(train_generator),
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=multile_output_generator(validation_generator),
                                 validation_steps=validation_generator.samples // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)


    def compute_flops(self):
        sum = 0
        for layer in self.model.layers:
            if type(layer) is Conv2D:
                p1 = layer.input.get_shape().as_list()[1:]
                p2 = layer.output.get_shape().as_list()[-1:]
                p3 = list(layer.kernel_size)
                sum += np.product(p1 + p2 + p3)
            elif type(layer) is Dense:
                sum += np.product(layer.kernel.get_shape().as_list())

        return sum

    def eval_hongkong_car(self, steps=None):
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/hongkong_car/test/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = validation_generator.samples // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(multile_output_generator(validation_generator),
                                                   steps=steps)
        print(evaluation)
        return evaluation

    def train_imagenet(self, training_save_dir='./resnet/imagenet/results', epochs=None):

        epochs = epochs if epochs is not None else self.config['train']['epochs']
        train_generator, validation_generator = imagenet_generator.data_generator('./dataset/imagenet/train/',
                                                                                  './dataset/imagenet/val/',
                                                                                  self.config['train']['batch_size'],
                                                                                  top_n_classes=self.config['model'][
                                                                                      'classes'])

        #### comppile model ########
        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        #### prepare training ########
        name = self.model.name
        os.makedirs(training_save_dir, exist_ok=True)
        result_counter = len(
            [log for log in os.listdir(training_save_dir) if name == '_'.join(log.split('_')[:-1])]) + 1
        saved_dir = os.path.join(training_save_dir, name + '_' + str(result_counter))
        os.makedirs(saved_dir, exist_ok=True)
        shutil.copyfile(self.config_path, os.path.join(saved_dir, self.config_path.split('/')[-1]))
        best_checkpoint = ModelCheckpoint(os.path.join(saved_dir, self.model.name + '_best.h5'),
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='max',
                                          period=1)

        checkpoint = ModelCheckpoint(os.path.join(saved_dir, self.model.name + '.h5'),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=False,
                                     mode='max',
                                     period=1)

        tensorboard = TensorBoard(log_dir=saved_dir,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)

        self.model.fit_generator(generator=multile_output_generator(train_generator),
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=multile_output_generator(validation_generator),
                                 validation_steps=(validation_generator.samples // self.config['train'][
                                     'batch_size']) * 0.5,
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)

    def eval_imagenet(self, steps=None):

        _, validation_generator = imagenet_generator.data_generator('./dataset/imagenet/train/',
                                                                    './dataset/imagenet/val/',
                                                                    self.config['train']['batch_size'],
                                                                    top_n_classes=self.config['model']['classes'])

        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = validation_generator.samples // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(multile_output_generator(validation_generator),
                                                   steps=steps)
        print(evaluation)
        return evaluation



    def train_scene(self, training_save_dir='./resnet/scene/results', epochs=None):
        classes = ['bookstore', 'computer_room', 'kitchen', 'airplane_cabin', 'swimming_pool-indoor', 'art_gallery',
                   'beach',
                   'mountain', 'rainforest', 'highway', 'crosswalk', 'campus', 'baseball_field', 'landfill',
                   'vegetable_garden',
                   'street', 'cafeteria', 'office', 'staircase', 'subway_station-platform', 'gymnasium-indoor',
                   'movie_theater-indoor',
                   'waterfall', 'desert-sand', 'golf_course', 'playground', 'parking_lot', 'tower', 'water_park', 'dam',
                   'balcony-exterior',
                   'phone_booth']

        epochs = epochs if epochs is not None else self.config['train']['epochs']

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=False,
            rotation_range=15.,
            width_shift_range=0.1,
            height_shift_range=0.1)

        train_generator = train_datagen.flow_from_directory(
            './dataset/scene/train/',
            classes=classes,
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/scene/test/',
            classes=classes,
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        #### comppile model ########
        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # self.model.summary()
        #### prepare training ########
        name = self.model.name
        os.makedirs(training_save_dir, exist_ok=True)
        result_counter = len(
            [log for log in os.listdir(training_save_dir) if name == '_'.join(log.split('_')[:-1])]) + 1
        saved_dir = os.path.join(training_save_dir, name + '_' + str(result_counter))
        os.makedirs(saved_dir, exist_ok=True)
        shutil.copyfile(self.config_path, os.path.join(saved_dir, self.config_path.split('/')[-1]))
        best_checkpoint = ModelCheckpoint(os.path.join(saved_dir, self.model.name + '_best.h5'),
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='max',
                                          period=1)

        checkpoint = ModelCheckpoint(os.path.join(saved_dir, self.model.name + '.h5'),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=False,
                                     mode='max',
                                     period=1)

        tensorboard = TensorBoard(log_dir=saved_dir,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)


        self.model.fit_generator(generator=multile_output_generator(train_generator),
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=multile_output_generator(validation_generator),
                                 validation_steps=validation_generator.samples // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)

    def eval_scene(self, steps=None):
        classes = ['bookstore', 'computer_room', 'kitchen', 'airplane_cabin', 'swimming_pool-indoor', 'art_gallery',
                   'beach',
                   'mountain', 'rainforest', 'highway', 'crosswalk', 'campus', 'baseball_field', 'landfill',
                   'vegetable_garden',
                   'street', 'cafeteria', 'office', 'staircase', 'subway_station-platform', 'gymnasium-indoor',
                   'movie_theater-indoor',
                   'waterfall', 'desert-sand', 'golf_course', 'playground', 'parking_lot', 'tower', 'water_park', 'dam',
                   'balcony-exterior',
                   'phone_booth']
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/scene/test/',
            target_size=(224, 224),
            classes=classes,
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = validation_generator.samples // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(multile_output_generator(validation_generator),
                                                   steps=steps)
        print(evaluation)
        return evaluation

def main_car():
    ee_resnet = EE_ResNet50('./resnet/hongkong_car/configs/80.json', weights_path='/home/xiao/projects/MSDNet/resnet'
                                                                              '/hongkong_car/results/80_1/80_best.h5')
    ee_resnet.generate_ee_model(freeze=True, add_depthwise=False)
    ee_resnet.eval_hongkong_car()
    ee_resnet.train_hongkong_car(training_save_dir='./resnet/hongkong_car/ee_results', epochs=200)

def main_imagenet():
    ee_resnet = EE_ResNet50('./resnet/imagenet/configs/100.json', weights_path='/home/xiao/projects/MSDNet/resnet'
                                                                              '/imagenet/results/100_1/100_best.h5')
    ee_resnet.generate_ee_model(freeze=True, add_depthwise=True)

    ee_resnet.eval_imagenet()
    ee_resnet.train_imagenet(training_save_dir='./resnet/imagenet/ee_results', epochs=200)

def main_scene():
    ee_resnet = EE_ResNet50('./resnet/scene/configs/90.json', weights_path='/home/xiao/projects/MSDNet/resnet'
                                                                                  '/scene/results/90_1/90_best.h5')
    ee_resnet.generate_ee_model(freeze=True, add_depthwise=True)

    ee_resnet.eval_scene()
    ee_resnet.train_scene(training_save_dir='./resnet/scene/ee_results', epochs=200)


if __name__ == '__main__':
    # main_GTSRB()
    # main_flops()
    # main_imagenet()
    # main_gender()
    main_car()
    # main_vgg_flops()
    # save_models_for_android()
    # main_imagenet100_from_scratch()
    # main_imagenet50_from_scratch()

    # main_dog()

    #main_imagenet()
    #main_scene()

