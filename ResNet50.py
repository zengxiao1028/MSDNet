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
from keras.preprocessing import image

class ResNet50(object):

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


    def find_layer_mapping(self):
        m = defaultdict(list)
        new_m = defaultdict(list)
        for layer in self.model.layers:
            layer_id = str(id(layer))
            for i, node in enumerate(layer.inbound_nodes):
                node_key = layer.name + '_ib-' + str(i)
                if node_key in self.model.container_nodes:
                    for inbound_layer in node.inbound_layers:
                        m[layer.name].append(inbound_layer.name)
        print(m)

        for k,v in m.items():
            for each in v:
                new_m[each].append(k)


        for k, v in new_m.items():

            while v[0].find('branch') < 0 and v[0] in new_m.keys():
                v = new_m[v[0]]

            new_m[k] = v
        return new_m


    def trim(self, filters_config, name):

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
        norml1 = [np.sum(np.fabs(w[:, :, :, idx])) for idx in
                  range(num_filter)]  # calculate the filter rank using l1 norm
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

        pos = self.trim_conv_block( trimmed_model, filters_config[1], stage=2, block='a',last_pos=pos)
        pos = self.trim_id_block(trimmed_model , filters_config[2], stage=2, block='b',last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[3], stage=2, block='c', last_pos=pos)

        pos = self.trim_conv_block(trimmed_model, filters_config[4], stage=3, block='a', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[5], stage=3, block='b', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[6], stage=3, block='c', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[7], stage=3, block='d', last_pos=pos)

        pos = self.trim_conv_block(trimmed_model, filters_config[8], stage=4, block='a', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[9], stage=4, block='b', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[10], stage=4, block='c', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[11], stage=4, block='d', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[12], stage=4, block='e', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[13], stage=4, block='f', last_pos=pos)

        pos = self.trim_conv_block(trimmed_model, filters_config[14], stage=5, block='a', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[15], stage=5, block='b', last_pos=pos)
        pos = self.trim_id_block(trimmed_model, filters_config[16], stage=5, block='c', last_pos=pos)

        return trimmed_model

    def trim_id_block(self, trim_model, filters, stage , block, last_pos=None):

        names = ['2a','2b','2c']
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'


        for idx, name in enumerate(names):
            w, b = self.model.get_layer(conv_name_base + name).get_weights()
            gamma, beta, mean, var = self.model.get_layer(bn_name_base + name).get_weights()

            # cut current layer
            num_filter = w.shape[3]  # original filter number, e.g., 48
            num_trim_filter = filters[idx]  # filter number after trim, e.g., 32
            norml1 = [np.sum(np.fabs(w[:, :, :, idx])) for idx in
                      range(num_filter)]  # calculate the filter rank using l1 norm
            pos = np.argsort(norml1)[:num_filter - num_trim_filter]

            w = np.delete(w, pos, axis=3)
            if last_pos is not None:
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

    def trim_conv_block(self, trim_model, filters, stage , block, last_pos=None):

        name = '1'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        w, b = self.model.get_layer(conv_name_base + name).get_weights()
        gamma, beta, mean, var = self.model.get_layer(bn_name_base + name).get_weights()

        # cut current layer
        num_filter = w.shape[3]  # original filter number, e.g., 48
        num_trim_filter = filters[2]  # filter number after trim, e.g., 32
        norml1 = [np.sum(np.fabs(w[:, :, :, idx])) for idx in
                  range(num_filter)]  # calculate the filter rank using l1 norm
        pos = np.argsort(norml1)[:num_filter - num_trim_filter]

        w = np.delete(w, pos, axis=3)
        if last_pos is not None:
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

    def train_cifar10(self):

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
        self.model.compile(opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        #### prepare training ########
        training_save_dir = './resnet/results'
        name = self.model.name
        os.makedirs(training_save_dir,exist_ok=True)
        result_counter = len([log for log in os.listdir(training_save_dir) if name == '_'.join(log.split('_')[:-1])]) + 1
        saved_dir = os.path.join(training_save_dir,name + '_' + str(result_counter))
        os.makedirs(saved_dir, exist_ok=True)
        shutil.copyfile(self.config_path, os.path.join(saved_dir, self.config_path.split('/')[-1]))
        best_checkpoint = ModelCheckpoint(os.path.join(saved_dir,  self.model.name +'_best.h5'),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     period=1)

        checkpoint = ModelCheckpoint(os.path.join(saved_dir, self.model.name +'.h5'),
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
                                steps_per_epoch= x_train.shape[0] // self.config['train']['batch_size'],
                                epochs=self.config['train']['epochs'],
                                validation_data=validation_generator,
                                validation_steps=x_test.shape[0] // self.config['train']['batch_size'],
                                callbacks=[best_checkpoint, checkpoint,tensorboard],
                                max_queue_size=64)

    def eval_cifar10(self):
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
        evaluation = self.model.evaluate_generator(validation_generator,
                                              steps=x_test.shape[0] // self.config['train']['batch_size'])
        print(evaluation)
        return evaluation


class Trimmer(object):

    def __init__(self, original_model_folder, target_config_path):
        self.original_model_folder = original_model_folder

        self.target_config_path = target_config_path


    def trim(self):

        with open(self.target_config_path) as config_buffer:
            target_config = json.load(config_buffer)

        trimmed_save_folder = './resnet/trimmed_models/' + target_config['model']['name']
        os.makedirs(trimmed_save_folder,exist_ok=True)
        log_file = open(os.path.join(trimmed_save_folder,"log.txt"),'w')

        log_file.write('loading original model from folder %s \n' % self.original_model_folder)
        resnet = ResNet50.init_from_folder(self.original_model_folder)


        if hasattr(resnet.model, '_collected_trainable_weights'):
            trainable_count = count_params(resnet.model._collected_trainable_weights)
        else:
            trainable_count = count_params(resnet.model.trainable_weights)
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(resnet.model.non_trainable_weights)]))
        log_file.write('Original Model total params: %d, Trainable params %d \n\n' % (trainable_count, non_trainable_count))


        log_file.write( 'Trimming model, using config : %s \n' % self.target_config_path)

        trimmed_model = resnet.trim(target_config['model']['filters'],target_config['model']['name'])

        if hasattr(trimmed_model, '_collected_trainable_weights'):
            trainable_count = count_params(trimmed_model._collected_trainable_weights)
        else:
            trainable_count = count_params(trimmed_model.trainable_weights)
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(trimmed_model.non_trainable_weights)]))
        log_file.write('Trimmed Model total params: %d, Trainable params %d \n' % (trainable_count, non_trainable_count))

        trimmed_save_folder = './resnet/trimmed_models/' + target_config['model']['name']
        trimmed_model.save(os.path.join(trimmed_save_folder, target_config['model']['name']+'.h5'))

        log_file.write(
            'saving trimmed model to  %s \n' % os.path.join(trimmed_save_folder, target_config['model']['name']+'.h5'))

        shutil.copyfile(self.target_config_path, os.path.join(trimmed_save_folder, self.target_config_path.split('/')[-1]))

        log_file.close()

        keras.backend.clear_session()

if __name__ == '__main__':


    #resnet100 = ResNet50('./resnet/configs/100.json')
    #resnet100.train_cifar10()

    trimmer = Trimmer('./resnet/results/100_1','./resnet/configs/90.json')
    trimmer.trim()

    resnet = ResNet50.init_from_folder('./resnet/trimmed_models/90')
    resnet.eval_cifar10()

    resnet.train_cifar10()

    # resnet100 = ResNet50('./resnet/configs/100.json')

    # trim_config_path = "./resnet/configs/90.json"
    # with open(trim_config_path) as config_buffer:
    #     config = json.load(config_buffer)
    # trimmed_model = resnet100.trim(config['model']['filters'],'trim')
    # trimmed_model.save(os.path.join('./resnet/ResNetModels',trimmed_model.name+'.h5'))
    # keras.backend.clear_session()
    #
    # resnet2 = ResNet50(trim_model_filter_configs, trimmed_model.name, include_top=True,
    #                    weights = os.path.join('./resnet/ResNetModels',trimmed_model.name+'.h5'))
    # resnet2.model.summary()
    # print(resnet2.trimfrom)




    #
    # config_path = './resnet/voc2007_config.json'
    # with open(config_path) as config_buffer:
    #     config = json.load(config_buffer)
    # resnet1 = _resNet50(config)
    # resnet1.train_cifar10(train_generator, validation_generator)

    # img = image.load_img('dog.jpg',target_size=(224,224))
    # x = image.img_to_array(img).astype(np.uint8)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # preds = resnet1.model.predict(x)
    # print('Predicted:', decode_predictions(preds))