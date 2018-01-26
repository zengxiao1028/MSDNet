import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.models import load_model
import tensorflow as tf
import os.path as osp
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import argparse

def transform(model_name , input_fld, output_fld):

    output_graphdef_file = 'model.ascii'

    graph_def = False
    output_node_prefix = 'outnode'


    # uncomment the following lines to alter the default values set above
    #model_type =  'imagenet50'

    #input_fld = os.path.join( './source', model_type)
    #output_fld = os.path.join( './target', model_type )

    input_model_file = '%s.h5' % model_name
    output_model_file = '%s.pb' % model_name


    output_fld = output_fld
    if not os.path.isdir(output_fld):
        os.mkdir(output_fld)
    weight_file_path = osp.join(input_fld, input_model_file)

    K.clear_session()
    K.set_learning_phase(0)
    net_model = load_model(weight_file_path)
    net_model.summary()
    num_output = len(net_model.outputs)
    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = output_node_prefix + str(i)
        pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)


    sess = K.get_session()

    if graph_def:
        f = output_graphdef_file
        tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
        print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))


    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_model_file, as_text=False)
    print('saved the freezed graph (ready for inference) at: ', osp.join(output_fld, output_model_file))


def main():

    folder = '/home/xiao/NestDNN/vgg4096'
    model_names = os.listdir(folder)

    for model_name in model_names:
        if model_name.find('S10p.h5')>0:
            transform(model_name.split('.')[0],folder,folder)


if __name__ == '__main__':
    main()
