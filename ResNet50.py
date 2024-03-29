import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
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
from keras.layers import SeparableConv2D
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

from keras.engine.topology import InputSpec
from keras.callbacks import Callback
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Reshape
from keras.models import load_model
from dataset.imagenet import imagenet_generator


class ResNet50(object):

    def __init__(self, config_path, weights_path=None):

        assert os.path.exists(config_path)

        self.config_path = config_path
        with open(config_path) as config_buffer:
            self.config = json.load(config_buffer)

        if weights_path is not None:
            self.model = self._resNet50(self.config['model']['filters'],
                                        include_top=self.config['model']['include_top'],
                                        weights=None,
                                        classes=self.config['model']['classes'],
                                        model_name=self.config['model']['name'])
            self.model.load_weights(weights_path)
        elif self.config['model']['weights'] in {'imagenet', None}:
            self.model = self._resNet50(self.config['model']['filters'],
                                        include_top=self.config['model']['include_top'],
                                        weights=self.config['model']['weights'],
                                        classes=self.config['model']['classes'],
                                        model_name=self.config['model']['name'])
        else:
            self.model = self._resNet50(self.config['model']['filters'],
                                        include_top=self.config['model']['include_top'],
                                        weights=None,
                                        classes=self.config['model']['classes'],
                                        model_name=self.config['model']['name'])
            if self.config['model']['weights'] != "":
                self.model.load_weights(self.config['model']['weights'])

    @classmethod
    def init_from_folder(cls, folder_path, best_only=True):
        original_config_paths = glob.glob(os.path.join(folder_path, '*.json'))
        assert len(original_config_paths) == 1

        original_config_path = original_config_paths[0]
        with open(original_config_path) as config_buffer:
            config = json.load(config_buffer)

        suffix = '_best' if best_only else ''
        resnet = cls(original_config_path,
                     os.path.join(folder_path, config['model']['name'] + '%s.h5' % suffix))
        return resnet

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu', name='ee' + str(stage) + block)(x)
        return x

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """A block that has a conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def _resNet50(self, filters_config, include_top=True, weights='imagenet',
                  input_tensor=None, input_shape=None,
                  pooling=None,
                  classes=1000, model_name='resnet50'):
        """Instantiates the ResNet50 architecture.
        Optionally loads weights pre-trained
        on ImageNet. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The data format
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization)
                or 'imagenet' (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 197.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')

        # if weights == 'imagenet' and include_top and classes != 1000:
        #    raise ValueError('If using `weights` as imagenet with `include_top`'
        #                     ' as true, `classes` should be 1000')

        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=197,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top,
                                          weights=weights)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(
            filters_config[0][0], (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        ee1 = x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        ee2 = x = self.conv_block(x, 3, filters_config[1], stage=2, block='a', strides=(1, 1))
        ee3 = x = self.identity_block(x, 3, filters_config[2], stage=2, block='b')
        ee4 = x = self.identity_block(x, 3, filters_config[3], stage=2, block='c')

        ee5 = x = self.conv_block(x, 3, filters_config[4], stage=3, block='a')
        ee6 = x = self.identity_block(x, 3, filters_config[5], stage=3, block='b')
        ee7 = x = self.identity_block(x, 3, filters_config[6], stage=3, block='c')
        ee8 = x = self.identity_block(x, 3, filters_config[7], stage=3, block='d')

        ee9 =  x = self.conv_block(x, 3, filters_config[8], stage=4, block='a')
        ee10 = x = self.identity_block(x, 3, filters_config[9], stage=4, block='b')
        ee11 = x = self.identity_block(x, 3, filters_config[10], stage=4, block='c')
        ee12 = x = self.identity_block(x, 3, filters_config[11], stage=4, block='d')
        ee13 = x = self.identity_block(x, 3, filters_config[12], stage=4, block='e')
        ee14 = x = self.identity_block(x, 3, filters_config[13], stage=4, block='f')

        ee15 = x = self.conv_block(x, 3, filters_config[14], stage=5, block='a')
        ee16 = x = self.identity_block(x, 3, filters_config[15], stage=5, block='b')
        x = self.identity_block(x, 3, filters_config[16], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        if include_top:
            x = Flatten()(x)
            x = Dense(classes, activation='softmax', name='fc1000')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        # Create model.
        model = Model(inputs, x, name=model_name)

        # load weights
        if weights == 'imagenet':
            if include_top and classes == 1000:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path, by_name=True)
            if K.backend() == 'theano':
                layer_utils.convert_all_kernels_in_model(model)
                if include_top:
                    maxpool = model.get_layer(name='avg_pool')
                    shape = maxpool.output_shape[1:]
                    dense = model.get_layer(name='fc1000')
                    layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        return model

    def trim(self, filters_config, name, random_trim):

        trimmed_model = self._resNet50(filters_config,  # use new filters setting
                                       include_top=self.config['model']['include_top'],  # use original setting
                                       weights=None,  # no need to preload
                                       classes=self.config['model']['classes'],  # no change
                                       model_name=name)
        w, b = self.model.get_layer('conv1').get_weights()
        gamma, beta, mean, var = self.model.get_layer('bn_conv1').get_weights()

        # find out which filters to be cut
        num_filter = w.shape[3]  # original filter number, e.g., 48
        num_trim_filter = filters_config[0][0]  # filter number after trim, e.g., 32
        assert num_filter >= num_trim_filter
        norml1 = [np.sum(np.fabs(w[:, :, :, idx])) for idx in
                  range(num_filter)]  # calculate the filter rank using l1 norm

        if random_trim:
            pos = np.array([i for i in range(num_filter - num_trim_filter)])
        else:
            pos = np.argsort(norml1)[:num_filter - num_trim_filter]

        # cut
        w = np.delete(w, pos, axis=3)
        b = np.delete(b, pos, axis=0)
        gamma = np.delete(gamma, pos, axis=0)
        beta = np.delete(beta, pos, axis=0)
        mean = np.delete(mean, pos, axis=0)
        var = np.delete(var, pos, axis=0)

        # set conv and conv

        trimmed_model.get_layer('conv1').set_weights([w, b])
        trimmed_model.get_layer('bn_conv1').set_weights([gamma, beta, mean, var])

        pos = self.trim_conv_block(trimmed_model, filters_config[1], stage=2, block='a', last_pos=pos,
                                   random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[2], stage=2, block='b', last_pos=pos,
                                 random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[3], stage=2, block='c', last_pos=pos,
                                 random_trim=random_trim)

        pos = self.trim_conv_block(trimmed_model, filters_config[4], stage=3, block='a', last_pos=pos,
                                   random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[5], stage=3, block='b', last_pos=pos,
                                 random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[6], stage=3, block='c', last_pos=pos,
                                 random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[7], stage=3, block='d', last_pos=pos,
                                 random_trim=random_trim)

        pos = self.trim_conv_block(trimmed_model, filters_config[8], stage=4, block='a', last_pos=pos,
                                   random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[9], stage=4, block='b', last_pos=pos,
                                 random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[10], stage=4, block='c', last_pos=pos,
                                 random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[11], stage=4, block='d', last_pos=pos,
                                 random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[12], stage=4, block='e', last_pos=pos,
                                 random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[13], stage=4, block='f', last_pos=pos,
                                 random_trim=random_trim)

        pos = self.trim_conv_block(trimmed_model, filters_config[14], stage=5, block='a', last_pos=pos,
                                   random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[15], stage=5, block='b', last_pos=pos,
                                 random_trim=random_trim)
        pos = self.trim_id_block(trimmed_model, filters_config[16], stage=5, block='c', last_pos=pos,
                                 random_trim=random_trim)
        #
        # trim last layer
        w, b = self.model.get_layer('fc1000').get_weights()
        ## test ##
        # w = np.random.normal(0, 0.01, w.shape)
        ## test ##
        w = np.delete(w, pos, axis=0)
        trimmed_model.get_layer('fc1000').set_weights([w, b])
        return trimmed_model

    def trim_id_block(self, trim_model, filters, stage, block, last_pos=None, random_trim=False):

        names = ['2a', '2b', '2c']
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        for idx, name in enumerate(names):
            w, b = self.model.get_layer(conv_name_base + name).get_weights()
            gamma, beta, mean, var = self.model.get_layer(bn_name_base + name).get_weights()

            # cut current layer
            num_filter = w.shape[3]  # original filter number, e.g., 48
            num_trim_filter = filters[idx]  # filter number after trim, e.g., 32
            assert num_filter >= num_trim_filter
            norml1 = [np.sum(np.fabs(w[:, :, :, idx])) for idx in
                      range(num_filter)]  # calculate the filter rank using l1 norm
            if random_trim:
                pos = np.array([i for i in range(num_filter - num_trim_filter)])
            else:
                pos = np.argsort(norml1)[:num_filter - num_trim_filter]

            w = np.delete(w, pos, axis=3)
            if last_pos is not None and len(last_pos) > 0:
                w = np.delete(w, last_pos, axis=2)
                last_pos = None
            b = np.delete(b, pos, axis=0)
            gamma = np.delete(gamma, pos, axis=0)
            beta = np.delete(beta, pos, axis=0)
            mean = np.delete(mean, pos, axis=0)
            var = np.delete(var, pos, axis=0)

            # set conv
            trim_model.get_layer(conv_name_base + name).set_weights([w, b])
            # set bn
            trim_model.get_layer(bn_name_base + name).set_weights([gamma, beta, mean, var])

            # save last_pos to cut next layer
            last_pos = pos

        return last_pos

    def trim_conv_block(self, trim_model, filters, stage, block, last_pos=None, random_trim=False):

        name = '1'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        w, b = self.model.get_layer(conv_name_base + name).get_weights()
        gamma, beta, mean, var = self.model.get_layer(bn_name_base + name).get_weights()

        # cut current layer
        num_filter = w.shape[3]  # original filter number, e.g., 48
        num_trim_filter = filters[2]  # filter number after trim, e.g., 32
        assert num_filter >= num_trim_filter
        norml1 = [np.sum(np.fabs(w[:, :, :, idx])) for idx in
                  range(num_filter)]  # calculate the filter rank using l1 norm
        if random_trim:
            pos = np.array([i for i in range(num_filter - num_trim_filter)])
        else:
            pos = np.argsort(norml1)[:num_filter - num_trim_filter]

        w = np.delete(w, pos, axis=3)
        if last_pos is not None and len(last_pos) > 0:
            w = np.delete(w, last_pos, axis=2)
        b = np.delete(b, pos, axis=0)
        gamma = np.delete(gamma, pos, axis=0)
        beta = np.delete(beta, pos, axis=0)
        mean = np.delete(mean, pos, axis=0)
        var = np.delete(var, pos, axis=0)

        # set conv
        trim_model.get_layer(conv_name_base + name).set_weights([w, b])
        # set bn
        trim_model.get_layer(bn_name_base + name).set_weights([gamma, beta, mean, var])

        last_pos = self.trim_id_block(trim_model, filters, stage, block, last_pos)

        return last_pos

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

    def train_cifar10(self, training_save_dir='./resnet/results', epochs=None):
        epochs = epochs if epochs is not None else self.config['train']['epochs']

        def resize(gen):
            """
            resize image to 224 x 224
            change to one-hot
            """
            while True:
                imgs, y = gen.next()
                img = np.array([scipy.misc.imresize(imgs[i, ...], (224, 224)) for i in range(imgs.shape[0])])

                yield (img, y)

        ### prepare dataset #####
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.1,
            horizontal_flip=True,
            rotation_range=30.,
            width_shift_range=0.1,
            height_shift_range=0.1)

        train_generator = resize(train_datagen.flow(x_train, y_train,
                                                    batch_size=self.config['train']['batch_size']))

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        validation_generator = resize(test_datagen.flow(x_test, y_test,
                                                        batch_size=self.config['train']['batch_size']))

        #### comppile model ########
        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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

        periodic_saver = PeriodicSaver(self.model,
                                       os.path.join(saved_dir, self.model.name + '_%03d.h5'), N=5)

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=x_train.shape[0] // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
                                 validation_steps=x_test.shape[0] // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard, periodic_saver],
                                 max_queue_size=64)

    def eval_cifar10(self, steps=None):
        def resize(gen):
            """
            resize image to 224 x 224
            change to one-hot
            """
            while True:
                imgs, y = gen.next()
                img = np.array([scipy.misc.imresize(imgs[i, ...], (224, 224)) for i in range(imgs.shape[0])])

                yield (img, y)

        ### prepare dataset #####
        (_, _), (x_test, y_test) = cifar10.load_data()

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        validation_generator = resize(test_datagen.flow(x_test, y_test,
                                                        batch_size=self.config['train']['batch_size']))
        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = x_test.shape[0] // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(validation_generator,
                                                   steps=steps)
        print(evaluation)
        return evaluation

    def build_early_exit_model(self, pointwise_conv_filters):

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
            x = GlobalAveragePooling2D()(x)
            x = Reshape((1, 1, -1), )(x)
            x = Conv2D(self.config['model']['classes'], (1, 1),
                       padding='same')(x)
            x = Activation('softmax')(x)
            x = Reshape((self.config['model']['classes'],), name=ee_name + '_output')(x)
            return x

        print('freezing original model...')
        layers = self.model.layers
        for layer in layers:
            layer.trainable = False

        print('creating early exit...')
        ee1 = _create_ouput(
            _depthwise_conv_block(self.model.get_layer('ee2c').output, pointwise_conv_filters=pointwise_conv_filters),
            'ee2c')
        ee2 = _create_ouput(
            _depthwise_conv_block(self.model.get_layer('ee3d').output, pointwise_conv_filters=pointwise_conv_filters),
            'ee3d')
        ee3 = _create_ouput(
            _depthwise_conv_block(self.model.get_layer('ee4f').output, pointwise_conv_filters=pointwise_conv_filters),
            'ee4f')

        ee_model = Model(self.model.inputs, [ee1, ee2, ee3] + self.model.outputs, name=self.config['model']['name'])
        self.model = ee_model

    def train_cifar10_early_exit(self, training_save_dir='./resnet/ee_results', epochs=None,
                                 pointwise_conv_filters=128):
        epochs = epochs if epochs is not None else self.config['train']['epochs']

        def resize(gen):
            """
            resize image to 224 x 224
            change to one-hot
            """
            while True:
                imgs, y = gen.next()
                img = np.array([scipy.misc.imresize(imgs[i, ...], (224, 224)) for i in range(imgs.shape[0])])

                yield (img, [y] * 4)

        ### prepare dataset #####
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.1,
            horizontal_flip=True,
            rotation_range=30.,
            width_shift_range=0.1,
            height_shift_range=0.1)

        train_generator = resize(train_datagen.flow(x_train, y_train,
                                                    batch_size=self.config['train']['batch_size']))

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        validation_generator = resize(test_datagen.flow(x_test, y_test,
                                                        batch_size=self.config['train']['batch_size']))

        #### comppile model ########
        opt = adam(lr=1e-4)
        self.build_early_exit_model(pointwise_conv_filters)
        self.model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=x_train.shape[0] // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
                                 validation_steps=x_test.shape[0] // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)

    def save_to_pb(self, folder):
        K.set_learning_phase(0)

        num_output = len(self.model.outputs)

        pred = [None] * num_output
        pred_node_names = [None] * num_output
        for i in range(num_output):
            pred_node_names[i] = 'outnode' + str(i)
            pred[i] = tf.identity(self.model.outputs[i], name=pred_node_names[i])
        print('output nodes names are: ', pred_node_names)

        sess = K.get_session()

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
        graph_io.write_graph(constant_graph, folder, self.model.name + '.pb', as_text=False)
        print('saved the freezed graph (ready for inference) at: ', os.path.join(folder, self.model.name + '.pb'))

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

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
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
        evaluation = self.model.evaluate_generator(validation_generator,
                                                   steps=steps)
        print(evaluation)
        return evaluation

    def train_GTSRB(self, training_save_dir='./resnet/GTSRB/results', epochs=None):

        epochs = epochs if epochs is not None else self.config['train']['epochs']

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=False,
            rotation_range=10.,
            width_shift_range=0.1,
            height_shift_range=0.1)

        train_generator = train_datagen.flow_from_directory(
            './dataset/GTSRB/train224/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/GTSRB/test224/',
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

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
                                 validation_steps=validation_generator.samples // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)

    def train_age(self, training_save_dir='./resnet/age/results', epochs=None):

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
            './dataset/age/faces2/train/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/age/faces2/test/',
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

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
                                 validation_steps=validation_generator.samples // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)

    def eval_age(self, steps=None):

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/age/faces2/test/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = validation_generator.samples // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(validation_generator,
                                                   steps=steps)
        print(evaluation)
        return evaluation

    def train_gender(self, training_save_dir='./resnet/gender/results', epochs=None):

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
            './dataset/gender/train/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/gender/test/',
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

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
                                 validation_steps=validation_generator.samples // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)

    def eval_gender(self, steps=None):

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/gender/test/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = validation_generator.samples // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(validation_generator,
                                                   steps=steps)
        print(evaluation)
        return evaluation

    def train_dog(self, training_save_dir='./resnet/dog/results', epochs=None):

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
            './dataset/dog/train/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/dog/test/',
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

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
                                 validation_steps=validation_generator.samples // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)

    def eval_dog(self, steps=None):

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/dog/test/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = validation_generator.samples // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(validation_generator,
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


        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
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
        evaluation = self.model.evaluate_generator(validation_generator,
                                                   steps=steps)
        print(evaluation)
        return evaluation

    def train_car(self, training_save_dir='./resnet/car/results', epochs=None):

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
            './dataset/car/train/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical'
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/car/test/',
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

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples // self.config['train']['batch_size'],
                                 epochs=epochs,
                                 validation_data=validation_generator,
                                 validation_steps=validation_generator.samples // self.config['train']['batch_size'],
                                 callbacks=[best_checkpoint, checkpoint, tensorboard],
                                 max_queue_size=64)

    def eval_car(self, steps=None):

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/car/test/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = validation_generator.samples // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(validation_generator,
                                                   steps=steps)
        print(evaluation)
        return evaluation

    def eval_GTSRB(self, steps=None):

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        validation_generator = test_datagen.flow_from_directory(
            './dataset/GTSRB/test224/',
            target_size=(224, 224),
            batch_size=self.config['train']['batch_size'],
            class_mode='categorical')

        opt = adam(lr=1e-4)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        if steps is None:
            steps = validation_generator.samples // self.config['train']['batch_size']
        evaluation = self.model.evaluate_generator(validation_generator,
                                                   steps=steps)
        print(evaluation)
        return evaluation


class Trimmer(object):

    def __init__(self, original_model_folder, target_config_path, random_trim=False):
        self.original_model_folder = original_model_folder
        self.target_config_path = target_config_path
        self.random_trim = random_trim

    def trim(self, trim_folder='./resnet/trimmed_models/'):

        with open(self.target_config_path) as config_buffer:
            target_config = json.load(config_buffer)

        trimmed_save_folder = os.path.join(trim_folder, target_config['model']['name'])
        os.makedirs(trimmed_save_folder, exist_ok=True)
        log_file = open(os.path.join(trimmed_save_folder, "log.txt"), 'w')

        log_file.write('loading original model from folder %s \n' % self.original_model_folder)
        resnet = ResNet50.init_from_folder(self.original_model_folder)

        if hasattr(resnet.model, '_collected_trainable_weights'):
            trainable_count = count_params(resnet.model._collected_trainable_weights)
        else:
            trainable_count = count_params(resnet.model.trainable_weights)
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(resnet.model.non_trainable_weights)]))
        log_file.write(
            'Original Model total params: %d, Trainable params %d \n\n' % (trainable_count, non_trainable_count))

        log_file.write('Trimming model, using config : %s \n' % self.target_config_path)

        trimmed_model = resnet.trim(target_config['model']['filters'], target_config['model']['name'], self.random_trim)

        if hasattr(trimmed_model, '_collected_trainable_weights'):
            trainable_count = count_params(trimmed_model._collected_trainable_weights)
        else:
            trainable_count = count_params(trimmed_model.trainable_weights)
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(trimmed_model.non_trainable_weights)]))
        log_file.write(
            'Trimmed Model total params: %d, Trainable params %d \n' % (trainable_count, non_trainable_count))

        trimmed_model.save(os.path.join(trimmed_save_folder, target_config['model']['name'] + '.h5'))

        log_file.write(
            'saving trimmed model to  %s \n' % os.path.join(trimmed_save_folder,
                                                            target_config['model']['name'] + '.h5'))

        shutil.copyfile(self.target_config_path,
                        os.path.join(trimmed_save_folder, self.target_config_path.split('/')[-1]))

        log_file.close()

        keras.backend.clear_session()


class PeriodicSaver(Callback):

    def __init__(self, model, save_path, N=10):
        self.model = model
        self.save_path = save_path  # 'weights%08d.h5'
        self.N = N

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.N == 0:
            name = self.save_path % epoch
            self.model.save(name)


def main_cifar10():
    print('##### Training resnetb20 ##### ')
    trimmer = Trimmer('./resnet/results/b10_1', './resnet/configs/b20.json')
    trimmer.trim()
    resnetb20 = ResNet50.init_from_folder('./resnet/trimmed_models/b20')
    resnetb20.eval_cifar10()
    resnetb20.train_cifar10()


def main_imagenet():
    # resnet100 = ResNet50('./resnet/imagenet/configs/100.json')
    # resnet100.eval_imagenet()
    # resnet100.train_imagenet()

    models = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0', 'b0', 'b10', 'b20']

    for i in range(3, len(models) - 1):
        print('Training resnet%s for imagenet' % models[i + 1])

        trimmer = Trimmer('./resnet/imagenet/results/%s_1' % models[i],
                          './resnet/imagenet/configs/%s.json' % models[i + 1])
        trimmer.trim(trim_folder='./resnet/imagenet/trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/imagenet/trimmed_models/%s' % models[i + 1], best_only=False)
        resnet.train_imagenet()


def main_imagenet50_from_scratch():
    # resnet = ResNet50('./resnet/imagenet50/configs/100.json')
    # resnet.train_imagenet(training_save_dir = './resnet/imagenet50/from_scratch_results')

    models = ['80']
    models = models[::-1]

    for i in range(0, len(models)):
        trimmer = Trimmer('./resnet/imagenet50/from_scratch_results/100_1',
                          './resnet/imagenet50/configs/%s.json' % models[i])
        trimmer.trim(trim_folder='./resnet/imagenet50/scratch_trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/imagenet50/scratch_trimmed_models/%s' % models[i], best_only=False)
        resnet.train_imagenet('./resnet/imagenet50/from_scratch_results')


def main_imagenet100_from_scratch():
    # resnet = ResNet50('./resnet/imagenet/configs/100.json')
    # resnet.train_imagenet(training_save_dir = './resnet/imagenet/from_scratch_results')

    models = ['100', '90', '80']
    models = models[::-1]

    for i in range(0, len(models) - 1):
        trimmer = Trimmer('./resnet/imagenet/from_scratch_results/100_1',
                          './resnet/imagenet/configs/%s.json' % models[i])
        trimmer.trim(trim_folder='./resnet/imagenet/scratch_trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/imagenet/scratch_trimmed_models/%s' % models[i],
                                           best_only=False)
        resnet.train_imagenet('./resnet/imagenet/from_scratch_results')


def main_GTSRB():
    # resnet100 = ResNet50('./resnet/GTSRB/configs/100.json')
    # resnet100.eval_GTSRB()
    # resnet100.train_GTSRB()

    models = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0', 'b0', 'b10', 'b20']

    for i in range(3, len(models) - 1):
        print('Training resnet%s for GTSRB' % models[i + 1])

        trimmer = Trimmer('./resnet/GTSRB/results/%s_1' % models[i], './resnet/GTSRB/configs/%s.json' % models[i + 1])
        trimmer.trim(trim_folder='./resnet/GTSRB/trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/GTSRB/trimmed_models/%s' % models[i + 1], best_only=False)
        resnet.eval_GTSRB()
        resnet.train_GTSRB()


def main_age():
    # resnet100 = ResNet50('./resnet/age/configs/100.json')
    # resnet100.eval_age()
    # resnet100.train_age()

    models = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0', 'b0', 'b10', 'b20']

    for i in range(0, len(models) - 1):
        print('Training resnet%s for age' % models[i + 1])
        trimmer = Trimmer('./resnet/age/results/%s_1' % models[i], './resnet/age/configs/%s.json' % models[i + 1])
        trimmer.trim(trim_folder='./resnet/age/trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/age/trimmed_models/%s' % models[i + 1], best_only=False)
        resnet.eval_age()
        resnet.train_age()


def main_gender():
    # resnet100 = ResNet50('./resnet/gender/configs/100.json')
    # resnet100.eval_gender()
    # resnet100.train_gender()

    models = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0', 'b0', 'b10', 'b20']

    for i in range(0, len(models) - 1):
        print('Training resnet%s for gender' % models[i + 1])

        trimmer = Trimmer('./resnet/gender/results/%s_1' % models[i], './resnet/gender/configs/%s.json' % models[i + 1])
        trimmer.trim(trim_folder='./resnet/gender/trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/gender/trimmed_models/%s' % models[i + 1], best_only=False)
        resnet.eval_gender()
        resnet.train_gender()


def main_dog():
    # resnet100 = ResNet50('./resnet/dog/configs/100.json')
    # resnet100.eval_dog()
    # resnet100.train_dog()

    models = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0', 'b0', 'b10', 'b20']

    for i in range(7, len(models) - 1):
        print('Training resnet%s for dog' % models[i + 1])

        trimmer = Trimmer('./resnet/dog/results/%s_1' % models[i], './resnet/dog/configs/%s.json' % models[i + 1])
        trimmer.trim(trim_folder='./resnet/dog/trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/dog/trimmed_models/%s' % models[i + 1], best_only=False)
        resnet.eval_dog()
        resnet.train_dog()


def main_scene():
    resnet100 = ResNet50('./resnet/scene/configs/100.json')
    resnet100.eval_scene()
    # resnet100.train_scene()

    models = ['100', '90', '70', '50', '40', '10', 'b20']

    for i in range(0, len(models) - 1):
        print('Training resnet%s for scene' % models[i + 1])

        trimmer = Trimmer('./resnet/scene/results/%s_1' % models[i], './resnet/scene/configs/%s.json' % models[i + 1])
        trimmer.trim(trim_folder='./resnet/scene/trimmed_models/')
        K.clear_session()
        resnet = ResNet50.init_from_folder('./resnet/scene/trimmed_models/%s' % models[i + 1], best_only=False)
        resnet.eval_scene()
        resnet.train_scene()


def main_scene_from_scratch():
    config100_path = './resnet/scene/configs/100.json'
    save_folder = './resnet/scene/from_scratch_results/100_1'
    resnet = ResNet50(config100_path)
    os.makedirs(save_folder)
    shutil.copyfile(config100_path, os.path.join(save_folder, config100_path.split('/')[-1]))
    resnet.model.save(os.path.join(save_folder, '100_best.h5'))

    models = ['100', '70', '40', '10', 'b20']
    models = models[::-1]

    for i in range(0, len(models) - 1):
        trimmer = Trimmer(save_folder,
                          './resnet/scene/configs/%s.json' % models[i], random_trim=True)
        trimmer.trim(trim_folder='./resnet/scene/scratch_trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/scene/scratch_trimmed_models/%s' % models[i],
                                           best_only=False)

        # resnet = ResNet50('./resnet/scene/configs/%s.json' % models[i])
        resnet.train_scene('./resnet/scene/from_scratch_results')


def main_scene_from_pretrain_imagenet():
    # config100_path = './resnet/scene/configs/110.json'
    save_folder = './resnet/scene/from_pretrain_imagenet_results/100_1'
    # resnet = ResNet50(config100_path)
    # os.makedirs(save_folder)
    # shutil.copyfile(config100_path, os.path.join(save_folder, config100_path.split('/')[-1]))
    # resnet.train_imagenet('./resnet/scene/from_pretrain_imagenet_results/')

    models = ['100', '70', '40', '10', 'b20']
    models = models[::-1]

    for i in range(0, len(models) - 1):
        #trimmer = Trimmer(save_folder,'./resnet/scene/configs/%s.json' % models[i], random_trim=True)
        #trimmer.trim(trim_folder='./resnet/scene/scratch_trimmed_models/')
        m = keras.models.load_model('./resnet/scene/scratch_trimmed_models/%s/%s.h5' % (models[i],models[i]))
        newout = Dense(32,activation='softmax')(m.layers[-2].output)
        m = Model(m.input ,newout)
        m.save('./resnet/scene/scratch_trimmed_models/%s/%s_best.h5' % (models[i],models[i]))
        resnet = ResNet50.init_from_folder('./resnet/scene/scratch_trimmed_models/%s' % models[i],
                                           best_only=True)

        # resnet = ResNet50('./resnet/scene/configs/%s.json' % models[i])
        resnet.train_scene('./resnet/scene/from_pretrain_imagenet_results')


def main_car():
    resnet100 = ResNet50('./resnet/car/configs/100.json')
    resnet100.eval_car()
    resnet100.train_car()

    models = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0', 'b0', 'b10', 'b20']

    for i in range(0, len(models) - 1):
        print('Training resnet%s for car' % models[i + 1])

        trimmer = Trimmer('./resnet/car/results/%s_1' % models[i], './resnet/car/configs/%s.json' % models[i + 1])
        trimmer.trim(trim_folder='./resnet/car/trimmed_models/')
        resnet = ResNet50.init_from_folder('./resnet/car/trimmed_models/%s' % models[i + 1], best_only=False)
        resnet.eval_car()
        resnet.train_car()


def save_models_for_android():
    model_root_folders = '/home/xiao/projects/MSDNet/resnet/results'

    model_folders = os.listdir(model_root_folders)

    for model_folder in model_folders:
        if os.path.isdir(os.path.join(model_root_folders, model_folder)):
            K.clear_session()
            K.set_learning_phase(0)
            resnet = ResNet50.init_from_folder(os.path.join(model_root_folders, model_folder), best_only=False)
            resnet.model.summary()
            resnet.save_to_pb('/home/xiao/projects/MSDNet/resnet/android_models')


def main_flops():
    models = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0', 'b0', 'b10', 'b20']

    for model in models:
        resnet = ResNet50('./resnet/imagenet/configs/%s.json' % model)
        resnet.model.summary()
        flops = resnet.compute_flops()
        print('{:.2f}'.format(flops / (1000 ** 3)))


def main_vgg_flops():
    folder = '/home/xiao/NestDNN/vgg4096'
    for model_name in os.listdir(folder):
        if model_name.find('.h5') > 0:
            if model_name.find('S10p') > 0:
                K.clear_session()
                model = load_model(os.path.join(folder, model_name))
                model.summary()
                sum = 0
                for layer in model.layers:
                    if type(layer) is Conv2D:
                        p1 = layer.input.get_shape().as_list()[1:]
                        p2 = layer.output.get_shape().as_list()[-1:]
                        p3 = list(layer.kernel_size)
                        sum += np.product(p1 + p2 + p3)
                    elif type(layer) is Dense:
                        sum += np.product(layer.kernel.get_shape().as_list())

                print(model_name + ' GFLOPS:{:.2f}'.format(sum / (1000 ** 3)))


if __name__ == '__main__':
    # main_GTSRB()
    # main_flops()
    # main_imagenet()
    # main_gender()
    # main_car()
    # main_vgg_flops()
    # save_models_for_android()
    # main_imagenet100_from_scratch()
    # main_imagenet50_from_scratch()

    # main_dog()
    main_scene()
    # main_scene_from_pretrain_imagenet()

    # main_vgg_flops()
