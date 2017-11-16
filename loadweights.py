import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


if __name__ == '__main__':
    f = h5py.File('/home/biyi/PycharmProjects/MSDNet/models/VGG16-100p-92-2fc2048.h5', mode='r')
    print(f)
    f = f['model_weights']
    print(f)
    # original_keras_version = f.attrs['keras_version'].decode('utf8')
    # original_backend = f.attrs['backend'].decode('utf8')

    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    layer_filter = []
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]
        layer_filter.append(weight_values)


    for k, value in enumerate(layer_filter):
        print('\n----------')
        print(k + 1, value[0].shape)
        if k < 13:
            a = [np.sum(np.fabs(value[0][:, :, :, idx])) for idx in range(value[0].shape[3])]
            plt.hist(a, bins=64)
            plt.show()


    ########prune################







    ############Save#####################3

    # from keras import __version__ as keras_version
    # f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in layers]
    # f.attrs['backend'] = K.backend().encode('utf8')
    # f.attrs['keras_version'] = str(keras_version).encode('utf8')
    #
    # for layer in layers:
    #     g = f.create_group(layer.name)
    #     symbolic_weights = layer.weights
    #     weight_values = K.batch_get_value(symbolic_weights)
    #     weight_names = []
    #     for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
    #         if hasattr(w, 'name') and w.name:
    #             name = str(w.name)
    #         else:
    #             name = 'param_' + str(i)
    #         weight_names.append(name.encode('utf8'))
    #     g.attrs['weight_names'] = weight_names
    #     for name, val in zip(weight_names, weight_values):
    #         param_dset = g.create_dataset(name, val.shape,
    #                                       dtype=val.dtype)
    #         if not val.shape:
    #             # scalar
    #             param_dset[()] = val
    #         else:
    #             param_dset[:] = val


