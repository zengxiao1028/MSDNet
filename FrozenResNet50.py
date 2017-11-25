import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.layers import BatchNormalization,Activation,Input,MaxPooling2D,AveragePooling2D,Flatten,Dense,Conv2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras import layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.datasets import cifar10
import scipy,keras,h5py
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
from keras.models import load_model
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Reshape
class FrozenResNet50(ResNet50):


    def __init__(self, config_path, frozen_model_config_path, weights_path = None, frozen_trainbale=False):

        assert os.path.exists(config_path)
        self.conv2d = FrozenConv2D
        self.dense = FrozenDense
        self.config_path = config_path
        self.frozen_trainable = frozen_trainbale
        # obtain frozen configs
        self.frozen_dim_filters = self._obtain_frozen_dim_filters(frozen_model_config_path)

        with open(config_path) as config_buffer:
            self.config = json.load(config_buffer)

        if weights_path is not None:
            self.model = self._resNet50(self.config['model']['filters'],
                                        self.frozen_dim_filters[0], ## frozen dim
                                        self.frozen_dim_filters[1], ## frozen filter
                                        include_top=self.config['model']['include_top'],
                                        weights=None,
                                        classes=self.config['model']['classes'],
                                        model_name=self.config['model']['name'])
            self.model.load_weights(weights_path)

        ## read model from config
        elif self.config['model']['weights'] in {'imagenet'}:
            raise ValueError('imagenet weights do not support frozen layer.')
        else:
            self.model = self._resNet50(self.config['model']['filters'],
                                        self.frozen_dim_filters[0],  ## frozen dim
                                        self.frozen_dim_filters[1],  ## frozen filter
                                        include_top=self.config['model']['include_top'],
                                        weights=None,
                                        classes=self.config['model']['classes'],
                                        model_name=self.config['model']['name'])
            if self.config['model']['weights'] is not None and self.config['model']['weights'] != "" :
                self.model.load_weights(self.config['model']['weights'])

    def load_frozen_aug_weights(self, frozen_model_folder , aug_weights_path = None ):
        original_config_paths = glob.glob(os.path.join(frozen_model_folder, '*.json'))
        assert len(original_config_paths) == 1

        original_config_path = original_config_paths[0]
        with open(original_config_path) as config_buffer:
            config = json.load(config_buffer)

        frozen_model_path = os.path.join(frozen_model_folder, config['model']['name'] + '.h5')
        #frozen_model = load_model(frozen_model_path,custom_objects={'FrozenConv2D':FrozenConv2D, 'FrozenDense':FrozenDense})
        with h5py.File(frozen_model_path, mode='r') as f:
            frozen_model_weights = f['model_weights']


            for idx, layer in enumerate(self.model.layers):

                weights = layer.get_weights()
                if len(weights) == 0:
                    continue

                g = frozen_model_weights[layer.name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                frozen_weights = [g[weight_name] for weight_name in weight_names]
                frozen_weights = [np.asarray(frozen_weight) for frozen_weight in frozen_weights]

                if type(layer) is FrozenConv2D :
                    frozen_weights = self._combine_frozen_weight(frozen_weights,'conv')

                    layer.set_weights( weights[:-2] + frozen_weights )

                elif type(layer) is FrozenDense :

                    frozen_weights = self._combine_frozen_weight(frozen_weights,'fc')
                    layer.set_weights( weights[:-2] + frozen_weights )

                else:
                    layer.set_weights(frozen_weights)



    def _obtain_frozen_dim_filters(self, frozen_model_config_path):
        with open(frozen_model_config_path) as config_buffer:
            frozen_config = json.load(config_buffer)

        frozen_model_dim = np.array(frozen_config['model']['filters'])
        frozen_model_dim[0][2] = frozen_model_dim[0][0]

        frozen_model_dim = np.roll(frozen_model_dim, 1)

        frozen_model_dim = frozen_model_dim[[i for i in range(1,frozen_model_dim.shape[0])] +[0]]
        frozen_model_dim[-1][1] = 0
        return frozen_model_dim, np.array(frozen_config['model']['filters'])

    def _combine_frozen_weight(self, frozen_weights, type):

        if type not in {'conv','fc'}:
            raise ValueError("must be 'conv' or 'fc'")
        if type == 'conv':
            if len(frozen_weights) == 2:#  kernel, bias
                return frozen_weights
            elif len(frozen_weights) == 3:# aug_dim, kernel, bias
                w = np.concatenate((frozen_weights[1], frozen_weights[0]), axis=2)
                b = frozen_weights[2]
                return [w, b]
            elif len(frozen_weights) == 4:  # aug_filter, aug_bias, kernel, bias
                w = np.concatenate((frozen_weights[2], frozen_weights[0]), axis=3)
                b = np.concatenate((frozen_weights[3], frozen_weights[1]), axis=0)
                return [w, b]
            elif len(frozen_weights) == 5:# aug_dim, aug_filter, aug_bias, kernel, bias
                w = np.concatenate( (frozen_weights[3],frozen_weights[0]), axis=2)
                w = np.concatenate((w, frozen_weights[1]), axis=3)
                b = np.concatenate((frozen_weights[4], frozen_weights[2]), axis=0)
                return [w,b]
            else:
                raise ValueError("Unsupported number of weights %d" % len(frozen_weights))

        elif type == 'fc':
            if len(frozen_weights) == 2:#  kernel, bias
                return frozen_weights
            elif len(frozen_weights) == 3:# aug_dim, kernel, bias
                w = np.concatenate( (frozen_weights[1],frozen_weights[0]), axis=0)
                b = frozen_weights[2]
                return [w,b]
            else:
                raise ValueError("Unsupported number of weights %d" % len(frozen_weights))
        else:
            raise NotImplementedError("Unsupport layer %s" % type)
    # @classmethod
    # def init_from_folder(cls,folder_path):
    #     original_config_paths = glob.glob(os.path.join(folder_path, '*.json'))
    #     assert len(original_config_paths) == 1
    #
    #     original_config_path = original_config_paths[0]
    #     with open(original_config_path) as config_buffer:
    #         config = json.load(config_buffer)
    #
    #     resnet = cls(original_config_path,
    #                       os.path.join(folder_path, config['model']['name'] + '.h5'))
    #     return resnet

    def identity_block(self,input_tensor, kernel_size, filters, frozen_dim, frozen_filters, stage, block):

        filters1, filters2, filters3 = filters
        frozen_dim1, frozen_dim2, frozen_dim3 = frozen_dim
        frozen_filers1, frozen_filers2, frozen_filers3 = frozen_filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = self.conv2d(filters1,  (1, 1), frozen_dim= frozen_dim1, frozen_filters=frozen_filers1,
                        frozen_trainable=self.frozen_trainable,
                        name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = self.conv2d(filters2, kernel_size, frozen_dim=frozen_dim2, frozen_filters=frozen_filers2,
                        frozen_trainable=self.frozen_trainable,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = self.conv2d(filters3, (1, 1), frozen_dim=frozen_dim3, frozen_filters=frozen_filers3,
                        frozen_trainable= self.frozen_trainable,
                        name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu',name='ee'+ str(stage) + block)(x)
        return x

    def conv_block(self,input_tensor, kernel_size, filters, frozen_dim, frozen_filters, stage, block, strides=(2, 2)):

        filters1, filters2, filters3 = filters
        frozen_dim1, frozen_dim2, frozen_dim3 = frozen_dim
        frozen_filers1, frozen_filers2, frozen_filers3 = frozen_filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = self.conv2d(filters1,  (1, 1), frozen_dim=frozen_dim1, frozen_filters=frozen_filers1, strides=strides,
                        frozen_trainable=self.frozen_trainable,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = self.conv2d(filters2, kernel_size, frozen_dim = frozen_dim2, frozen_filters=frozen_filers2, padding='same',
                        frozen_trainable=self.frozen_trainable,
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = self.conv2d(filters3, (1, 1), frozen_dim = frozen_dim3, frozen_filters=frozen_filers3,
                        frozen_trainable=self.frozen_trainable,
                        name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = self.conv2d(filters3, (1, 1), frozen_dim = frozen_dim1, frozen_filters=frozen_filers3,
                               frozen_trainable = self.frozen_trainable,
                               strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def _resNet50(self, filters_config, frozen_dim_configs, frozen_filters_config , include_top=True, weights='imagenet',
                  input_tensor=None, input_shape=None,
                  pooling=None,
                  classes=1000, model_name='resnet50'):

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

        x = self.conv2d(
            filters_config[0][0], (7, 7), frozen_dim=3, frozen_filters=frozen_filters_config[0][0],
            frozen_trainable=self.frozen_trainable, strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, filters_config[1], frozen_dim_configs[0], frozen_filters_config[1], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, filters_config[2],frozen_dim_configs[1], frozen_filters_config[2], stage=2, block='b')
        x = self.identity_block(x, 3, filters_config[3],frozen_dim_configs[2], frozen_filters_config[3], stage=2, block='c')

        x = self.conv_block(x, 3, filters_config[4], frozen_dim_configs[3], frozen_filters_config[4], stage=3, block='a')
        x = self.identity_block(x, 3, filters_config[5],frozen_dim_configs[4], frozen_filters_config[5], stage=3, block='b')
        x = self.identity_block(x, 3, filters_config[6],frozen_dim_configs[5], frozen_filters_config[6], stage=3, block='c')
        x = self.identity_block(x, 3, filters_config[7],frozen_dim_configs[6], frozen_filters_config[7], stage=3, block='d')

        x = self.conv_block(x, 3, filters_config[8],frozen_dim_configs[7], frozen_filters_config[8], stage=4, block='a')
        x = self.identity_block(x, 3, filters_config[9],frozen_dim_configs[8], frozen_filters_config[9], stage=4, block='b')
        x = self.identity_block(x, 3, filters_config[10],frozen_dim_configs[9], frozen_filters_config[10], stage=4, block='c')
        x = self.identity_block(x, 3, filters_config[11],frozen_dim_configs[10], frozen_filters_config[11], stage=4, block='d')
        x = self.identity_block(x, 3, filters_config[12],frozen_dim_configs[11], frozen_filters_config[12], stage=4, block='e')
        x = self.identity_block(x, 3, filters_config[13],frozen_dim_configs[12], frozen_filters_config[13], stage=4, block='f')

        x = self.conv_block(x, 3, filters_config[14],frozen_dim_configs[13], frozen_filters_config[14], stage=5, block='a')
        x = self.identity_block(x, 3, filters_config[15],frozen_dim_configs[14], frozen_filters_config[15], stage=5, block='b')
        x = self.identity_block(x, 3, filters_config[16],frozen_dim_configs[15], frozen_filters_config[16], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        if include_top:
            x = Flatten()(x)
            x = self.dense(classes, frozen_dim=frozen_dim_configs[-1][0],
                           frozen_trainable=self.frozen_trainable, activation='softmax', name='fc1000')(x)
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

    def build_early_exit_model(self):

        def relu6(x):
            return K.relu(x, max_value=6)

        def _depthwise_conv_block(inputs, pointwise_conv_filters,
                                  depth_multiplier=1, strides=(1, 1), block_id=1):

            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

            x = inputs

            # x = Conv2D(pointwise_conv_filters, (1, 1),
            #            padding='same',
            #            use_bias=False,
            #            strides=(1, 1),
            #            name='conv_pre_pw_%d' % block_id)(inputs)
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
            x = Reshape((1,1,-1),)(x)
            x = Conv2D(self.config['model']['classes'], (1, 1),
                       padding='same')(x)
            x = Activation('softmax')(x)
            x = Reshape((self.config['model']['classes'],), name=ee_name+'_output')(x)
            return x


        layers = self.model.layers
        for layer in layers:
            layer.trainable = False

        ee1 = _create_ouput(_depthwise_conv_block(self.model.get_layer('ee2c').output,pointwise_conv_filters=512),'ee2c')
        ee2 = _create_ouput(_depthwise_conv_block(self.model.get_layer('ee3d').output,pointwise_conv_filters=512),'ee3d')
        ee3 = _create_ouput(_depthwise_conv_block(self.model.get_layer('ee4f').output,pointwise_conv_filters=512),'ee4f')

        ee_model = Model(self.model.inputs, [ee1,ee2,ee3])
        self.model = ee_model

    def train_cifar10_early_exit(self, training_save_dir='./resnet/ee_results', epochs=None):
        epochs = epochs if epochs is not None else self.config['train']['epochs']

        def resize(gen):
            """
            resize image to 224 x 224
            change to one-hot
            """
            while True:
                imgs, y = gen.next()
                img = np.array([scipy.misc.imresize(imgs[i, ...], (224, 224)) for i in range(imgs.shape[0])])

                yield (img, [y]*3)

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
        self.build_early_exit_model()
        self.model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        #### prepare training ########
        name = self.model.name
        os.makedirs(training_save_dir, exist_ok=True)
        result_counter = len([log for log in os.listdir(training_save_dir) if name == '_'.join(log.split('_')[:-1])]) + 1
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

class FrozenConv2D(Conv2D):

    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 frozen_dim=0,  ## nb of dim we need to freeze
                 frozen_filters=0,  ## nb of filters we need to freeze
                 frozen_trainable=False, # for training baseline
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 #kernel_initializer='zeros',
                 kernel_initializer='truncated_normal',
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
        self.frozen_trainable = frozen_trainable
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
            if self.aug_filters > 0:
                self.aug_bias = self.add_weight(shape=(self.aug_filters,),
                                        initializer=self.bias_initializer,
                                        name='aug_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.kernel = self.add_weight(shape=frozen_kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.frozen_trainable
                                      )
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.frozen_filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=self.frozen_trainable)


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
                 frozen_dim=0, # right now it only supports frozen_dim. Frozen_units will be supported in the future.
                 frozen_trainable=False, # for training baseline
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 #kernel_initializer='zeros',
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
        self.frozen_trainable = frozen_trainable

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.frozen_dim > input_dim:
            raise ValueError('Frozen_dim should be smaller than or equal to input dim size')



        self.aug_dim = input_dim - self.frozen_dim
        if self.aug_dim > 0 :
            self.aug_dim_kernel = self.add_weight(shape=(self.aug_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='aug_dim_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel = self.add_weight(shape=(self.frozen_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.frozen_trainable)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=self.frozen_trainable)
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

def recover_cifar10(frozen_trainable=False):

    model_types = [
        #('b20', 'b10'),
        ('b10', 'b0'),
        ('b0', '50'),
        ('50', '80'),
    ]
    for idx, types in enumerate(model_types):
        K.clear_session()
        frozen_model_type, recover_model_type = types
        resnet = FrozenResNet50(config_path = './resnet/configs/%s.json' % recover_model_type,
                                frozen_model_config_path = './resnet/configs/%s.json' % frozen_model_type,
                                frozen_trainbale = frozen_trainable)

        save_dir = 'recover_results' if frozen_trainable is False else 'unfreeze_recover_results'
        if frozen_model_type == 'b20':
            resnet.load_frozen_aug_weights('./resnet/results/%s_1' % frozen_model_type)
        else:
            resnet.load_frozen_aug_weights('./resnet/recover_results/%s_1' %  frozen_model_type )

        resnet.train_cifar10(
            training_save_dir='./resnet/%s/' % save_dir,epochs=100)


def recover_imagenet():

    model_types = [
        ('b20', 'b10'),
        ('b10', 'b0'),
        ('b0', '50'),
        ('50', '80'),
    ]
    for idx, types in enumerate(model_types):
        frozen_model_type, recover_model_type = types
        resnet = FrozenResNet50(config_path='./resnet/imagenet/configs/%s.json' % recover_model_type,
                                frozen_model_config_path='./resnet/imagenet/configs/%.json' % frozen_model_type)

        if idx == 0:
            resnet.load_frozen_aug_weights('./resnet/imagenet/results/%s_1' % frozen_model_type)
        else:
            resnet.load_frozen_aug_weights('./resnet/imagenet/recover_results/%s_1' % frozen_model_type)

        resnet.train_imagenet(training_save_dir='./resnet/imagenet/recover_results/')




def train_cifar10_early_exit():

    model_types = ['b20','b10','b0','50','80']

    for t in model_types:
        recover_model_type = frozen_model_type = t

        if t =='b20':
        resnet = FrozenResNet50(config_path='./resnet/configs/%s.json' % recover_model_type,
                                frozen_model_config_path='./resnet/configs/%s.json' % frozen_model_type,
                                frozen_trainbale=False)

        resnet.load_frozen_aug_weights('./resnet/recover_results/%s_1' % recover_model_type )

        print('Evaluating top output of %s' % recover_model_type)
        resnet.eval_cifar10()
        print('Training ee of %s' % recover_model_type)
        resnet.train_cifar10_early_exit(training_save_dir='./resnet/ee_results')



if __name__ == '__main__':
    #recover_cifar10(frozen_trainable=True)

    train_cifar10_early_exit()

