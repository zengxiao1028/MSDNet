import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.layers import Conv2D,BatchNormalization,Activation,Input,MaxPooling2D,AveragePooling2D,Flatten,Dense
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras import layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.datasets import cifar10
import scipy,keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings,os
from keras.optimizers import adam
import keras.backend as K
import json,shutil,glob
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from collections import defaultdict
from keras.utils.layer_utils import count_params
from keras.applications.imagenet_utils import _obtain_input_shape,preprocess_input,decode_predictions
from keras.engine.topology import get_source_inputs
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
from keras.legacy import interfaces
from keras.engine.topology import InputSpec
from ResNet50 import ResNet50

class FrozenResNet50(ResNet50):

    def __init__(self, config_path, weights_path = None ):

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
                self.model.load_weights(self.model.load_weights(self.config['model']['weights']))

    @classmethod
    def init_from_folder(cls,folder_path):
        original_config_paths = glob.glob(os.path.join(folder_path, '*.json'))
        assert len(original_config_paths) == 1

        original_config_path = original_config_paths[0]
        with open(original_config_path) as config_buffer:
            config = json.load(config_buffer)

        resnet = cls(original_config_path,
                          os.path.join(folder_path, config['model']['name'] + '.h5'))
        return resnet

    def identity_block(self,input_tensor, kernel_size, filters, stage, block):
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
        x = Activation('relu')(x)
        return x

    def conv_block(self,input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
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

        #if weights == 'imagenet' and include_top and classes != 1000:
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
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, filters_config[1], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, filters_config[2], stage=2, block='b')
        x = self.identity_block(x, 3, filters_config[3], stage=2, block='c')

        x = self.conv_block(x, 3, filters_config[4], stage=3, block='a')
        x = self.identity_block(x, 3, filters_config[5], stage=3, block='b')
        x = self.identity_block(x, 3, filters_config[6], stage=3, block='c')
        x = self.identity_block(x, 3, filters_config[7], stage=3, block='d')

        x = self.conv_block(x, 3, filters_config[8], stage=4, block='a')
        x = self.identity_block(x, 3, filters_config[9], stage=4, block='b')
        x = self.identity_block(x, 3, filters_config[10], stage=4, block='c')
        x = self.identity_block(x, 3, filters_config[11], stage=4, block='d')
        x = self.identity_block(x, 3, filters_config[12], stage=4, block='e')
        x = self.identity_block(x, 3, filters_config[13], stage=4, block='f')

        x = self.conv_block(x, 3, filters_config[14], stage=5, block='a')
        x = self.identity_block(x, 3, filters_config[15], stage=5, block='b')
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
            if include_top and classes==1000:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path,by_name=True)
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



    def recover(self,filters_config, name):
        recovered_model = self._resNet50(filters_config,  # use new filters setting
                                       include_top=self.config['model']['include_top'],  # use original setting
                                       weights=None,  # no need to preload
                                       classes=self.config['model']['classes'],  # no change
                                       model_name=name)
        weights = recovered_model.layers[6].non_trainable_weights
        print(weights)



class FrozenConv2D(Conv2D):

    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 frozen_dim,  ## nb of dim we need to freeze
                 frozen_filters,  ## nb of filters we need to freeze
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)

        ##add by xiao ###

        self.frozen_dim = frozen_dim
        self.frozen_filters = frozen_filters
        ###################

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]

        if self.frozen_filters > self.filters or self.frozen_dim > input_dim:
            raise ValueError('Frozen filter size should be smaller than current filter size')

        kernel_shape = self.kernel_size + (input_dim, self.filters)

        frozen_kernel_shape = self.kernel_size + (self.frozen_dim, self.frozen_filters)
        self.kernel = self.add_weight(shape=frozen_kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=False
                                      )

        self.aug_dim = input_dim - self.frozen_dim
        if self.aug_dim > 0:
            aug_dim_kernel_shape = self.kernel_size + (self.aug_dim, self.frozen_filters)
            self.aug_dim_kernel = self.add_weight(shape=aug_dim_kernel_shape,
                                                  initializer=self.kernel_initializer,
                                                  name='aug_dim_kernel',
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint
                                                  )


        self.aug_filters = self.filters - self.frozen_filters
        if self.aug_filters > 0:
            aug_filter_kernel_shape = self.kernel_size + (input_dim, self.aug_filters)
            self.aug_filter_kernel = self.add_weight(shape=aug_filter_kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='aug_filter_kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint
                                          )


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.frozen_filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=False)
            if self.aug_filters > 0:
                self.aug_bias = self.add_weight(shape=(self.aug_filters,),
                                        initializer=self.bias_initializer,
                                        name='aug_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)


        else:
            self.bias = None


        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        ### concat the frozen and aug filters
        if self.aug_dim > 0:
            self.kernel = K.concatenate([self.kernel, self.aug_dim_kernel], axis=2)
        if self.aug_filters > 0 :
            self.kernel = K.concatenate([self.kernel, self.aug_filter_kernel], axis=3)
            self.bias = K.concatenate([self.bias, self.aug_bias], axis=0)


        return super(FrozenConv2D,self).call(inputs)


class FrozenDense(Dense):

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 frozen_dim, # right now it only supports frozen_dim. Frozen_units will be supported in the future.
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(FrozenDense, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.frozen_dim = frozen_dim


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.frozen_dim > input_dim:
            raise ValueError('Frozen_dim should be smaller than or equal to input dim size')

        self.kernel = self.add_weight(shape=(self.frozen_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=False)

        self.aug_dim = input_dim - self.frozen_dim
        if self.aug_dim > 0 :
            self.aug_dim_kernel = self.add_weight(shape=(self.aug_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='aug_dim_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        if self.aug_dim > 0:
            self.kernel = K.concatenate([self.kernel, self.aug_dim_kernel], axis=0)

        return super(FrozenDense,self).call(inputs)

class FrozenBatchNormalization(BatchNormalization):

    @interfaces.legacy_batchnorm_support
    def __init__(self,
                 frozen_dim,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(FrozenBatchNormalization, self).__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs)
        self.frozen_dim = frozen_dim


    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.frozen_dim > dim:
            raise ValueError('Frozen_dim should be smaller than or equal to input dim size')
        self.aug_dim = dim - self.frozen_dim

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         trainable=False)
            if self.aug_dim > 0:
                self.aug_dim_gamma = self.add_weight(shape=self.aug_dim,
                                         name='aug_dim_gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        trainable=False)
            if self.aug_dim > 0:
                self.aug_dim_beta = self.add_weight(shape=self.aug_dim,
                                         name='aug_dim_beta',
                                         initializer=self.beta_initializer,
                                         regularizer=self.beta_regularizer,
                                         constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):

        return super(FrozenBatchNormalization, self).call(inputs,training)



if __name__ == '__main__':
    pass

