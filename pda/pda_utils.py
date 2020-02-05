from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import logging
import os.path
import numpy as np
logger = logging.getLogger(__name__)
from matplotlib import pyplot as plt
import pickle
from art.poison_detection.activation_defence import ActivationDefence
from art.visualization import plot_3d
from pda.pda_visualization import plot_2d
from pda import model_path
import configparser

tf.get_logger().setLevel(logging.ERROR)

# def show_img_array(arr):
#     try:
#         cv2.imshow("image", arr)
#         cv2.waitKey()
#     except:
#         plt.imshow(arr)
# Save / Load models
class Save_Load_Model(object):
    def __init__(self, sess=None, saver=None, kr_model=None, path='./models/', fname=''):
        self.sess = sess # tf or kr session
        self.saver = saver # tf saver
        self.kr_model = kr_model # kr model
        self.path = path
        self.fname = fname # only filename without ext
        pass

    # preprocess path and fname
    def preprocess(func):
        def wrapper(self, *args):
            import os
            filename, ext = os.path.splitext(self.fname)
            # print(ext)
            if ext:
                # make fname only name without ext
                self.fname = filename
            return func(self)
        return wrapper

    # To save model
    def tf_ckpt_saver(self):
        save_path = self.saver.save(self.sess, self.path + self.fname + ".ckpt")
        return save_path
    # To load ckpt model
    def tf_print_ckpt(self):
        if self.saver==None:
            self.saver = tf.train.Saver()
        with tf.Session() as sess:
            self.saver.restore(sess, self.path + self.fname) #"./models/spacenet_model.ckpt"
        self.sess = sess
        return self.saver, self.sess
    # validate the restored sess and print its operations
    def tf_print_ckpt(self):
        for op in self.sess.graph.get_operations():
            print(op)

    # Save a frozen pb model file
    def tf_frozen_pb_saver (self):
        '''
        :return: graph of the frozen pb model
        '''
        # all nodes names
        nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # create a graph
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            nodes  # The output node names are used to select the usefull nodes
        )
        # write pb file with weights
        try:
            with tf.gfile.GFile(self.path + self.fname + '_frozen.pb', "wb") as f:
                f.write(output_graph_def.SerializeToString())
            return True
        except:
            return False

    # Load a frozen pb model
    def tf_frozen_pb_loader (self):
        def load_graph(frozen_graph_filename):
            # We load the protobuf file from the disk and parse it to retrieve the
            # unserialized graph_def
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Then, we import the graph_def into a new Graph and returns it
            with tf.Graph().as_default() as graph:
                # The name var will prefix every op/nodes in your graph
                # Since we load everything in a new graph, this is not needed
                tf.import_graph_def(graph_def, name="prefix")
            return graph

        graph = load_graph(self.path + self.fname + '_frozen.pb')
        return graph

    # Convert h5(keras) --> pb/pbtxt (tensorflow) file
    def tf_convert_from_h5(self, as_text=False):
        """
        sess for Keras can be obtained by using:
                from keras import backend as K
                sess = K.get_session()

        :return:
        """
        def freeze_session(session, keep_var_names=None,
                           output_names=None, clear_devices=True):
            """
            Freezes the state of a session into a pruned computation graph.

            Creates a new computation graph where variable nodes are replaced by
            constants taking their current value in the session. The new graph will be
            pruned so subgraphs that are not necessary to compute the requested
            outputs are removed.
            @param session The TensorFlow session to be frozen.
            @param keep_var_names A list of variable names that should not be frozen,
                                  or None to freeze all the variables in the graph.
            @param output_names Names of the relevant graph outputs.
            @param clear_devices Remove the device directives from the graph for better portability.
            @return The frozen graph definition.
            """
            from tensorflow.python.framework.graph_util import convert_variables_to_constants
            graph = session.graph
            with graph.as_default():
                freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
                output_names = output_names or []
                output_names += [v.op.name for v in tf.global_variables()]
                # Graph -> GraphDef ProtoBuf
                input_graph_def = graph.as_graph_def()
                if clear_devices:
                    for node in input_graph_def.node:
                        node.device = ""
                frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                              output_names, freeze_var_names)
                return frozen_graph

        output_names = [out.op.name for out in self.kr_model.outputs]
        frozen_graph = freeze_session(self.sess, output_names=output_names)
        if as_text:
            self.fname = self.fname + ".pbtxt"
        else:
            self.fname = self.fname + ".pb"
        tf.train.write_graph(frozen_graph, self.path, self.fname, as_text=as_text)

    # load keras model (model should have been saved using model.save())
    @preprocess
    def kr_h5_loader(self):
        from keras.models import load_model
        # self.fname = self.preprocess()
        json_file = self.path + self.fname + '.json'
        if os.path.exists(json_file):
            from keras.models import model_from_json
            with open(json_file) as jf:
                json_model = jf.read()
                jf.close()
            model = model_from_json(json_model)
            # load weights into new model
            model.load_weights(self.path + self.fname + '.h5')
        else:
            model = load_model(self.path + self.fname + '.h5')
        return model
    # Saver checkpoint
    def saver_ckpt(self):
        '''
        There will be four files:-
        1.tensorflowModel.ckpt.meta: Tenosrflow stores the graph structure separately from the variable values.
        The file .ckpt.meta contains the complete graph. It includes GraphDef, SaverDef, and so on.
        2.tensorflowModel.ckpt.data-00000-of-00001: This contains the values of variables(weights, biases, placeholders,
        gradients, hyper-parameters etc).
        3.tensorflowModel.ckpt.index: It is a table where Each key is the name of a tensor and its value is a serialized
        BundleEntryProto.
        serialized BundleEntryProto holds metadata of the tensors. Metadata of a tensor may be like: which of the “data”
        files contains the content of a tensor, the offset into that file, checksum, some auxiliary data, etc.
        4.checkpoint:All checkpoint information, like model ckpt file name and path
        :param sess:
        :return:
        '''
        saver = tf.train.Saver()
        saver.save(self.sess, self.path + self.fname)

    # Saver pb/pbtxt
    def saver_pb(self, as_text=False):
        '''
         a protocol buffer file .pbtxt. It is human readable if you want to convert it to binary : as_test = False
        Difference between .meta files and .pbtxt files, well you can say .pbtxt are human readable whereas .meta files
        are not. But if you keep as_text = false it is no more human readable. Still they are different . .meta files
        holds ,more than just the structure of the graph like MetaInfoDef , GraphDef SaverDef , CollectionDef .
        Whereas .pbtxt files holds only the structure of the graph.
        :param sess:
        :param as_text:
        :return:
        '''
        if as_text:
            self.fname = self.fname + ".pbtxt"
        else:
            self.fname = self.fname + ".pb"

        tf.train.write_graph(self.sess.graph.as_graph_def(),
                             self.path,
                             self.fname,
                             as_text=as_text)

    # Freeze the graph
    # ???? NOT WORKING
    def freeze_save(self,
                    input_graph_filename='path/to/saved/pb_file',
                    input_saver_def_path=False,
                    input_binary=True,
                    checkpoint_path='',
                    output_node_names="output_node",
                    restore_op_name=None,
                    filename_tensor_name=None,
                    output_graph_filename='',
                    clear_devices=False,
                    initializer_nodes=''):
        '''
        When we need to keep all the values of the variables and the
        Graph in a single file we do it with freezing the graphs.
        It's useful to do this when we need to load a single file in C++,
        especially in environments like mobile or embedded where we may not
        have access to the RestoreTensor ops and file loading calls that they rely on.
        :param input_graph_filename: is a already saved .pb filename path
        :param input_saver_def_path: A TensorFlow Saver file
        :param input_binary: True means input_graph is .pb, False indicates .pbtxt.
        :param checkpoint_path:
        :param output_node_names:
        :param restore_op_name:
        :param filename_tensor_name:
        :param output_graph_filename: String where to write the frozen `GraphDef`.
        :param output_node_names: the name of output_node in the graph
        :param clear_devices:
        :param input_meta_graph: (optional)
        :param input_saved_model_dir: (optional)
        :param saved_model_tags: (optional)
        :return:

        :example:
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph_test.py

        '''
        from tensorflow.python.tools import freeze_graph

        # restore_op_name = "save/restore_all"
        # filename_tensor_name = "save/Const:0"
        # output_frozen_graph_name = 'frozen_' + self.PATH_NAME+'.pb'
        if checkpoint_path == '':
            checkpoint_path = self.path + self.fname + '.ckpt',
        if output_graph_filename == '':
            output_graph_filename = self.path + self.fname

        freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                                  input_binary, checkpoint_path, output_node_names,
                                  restore_op_name, filename_tensor_name,
                                  output_graph_filename, clear_devices,
                                  initializer_nodes)

    # Optimize for inference
    # ???? NOT WORKING
    def optimize_freeze_save(self):
        # https://medium.com/@prasadpal107/saving-freezing-optimizing-for-inference-restoring-of-tensorflow-models-b4146deb21b5
        from tensorflow.python.tools import optimize_for_inference_lib
        output_optimized_graph_name = 'optimized_' + self.path + '.pb'
        output_graph_filename = self.path + self.path
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(self.path + output_graph_filename, "r") as f:
            data = f.read()
            input_graph_def.ParseFromString(data)

        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            ["I"],  # an array of the input node(s)
            ["O"],  # an array of output nodes
            tf.float32.as_datatype_enum)

        # Save the optimized graph

        f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
        f.write(output_graph_def.SerializeToString())

        # tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)



'''
# quick example of building a graph of tf model and save a freeze model based on 
# output tag of tensor
num_nodes = 64
vocabulary_size = 128
batch_size = 100
graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.                             
    cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))

    ###
      saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]), name="saved_sample_output")


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    mean_loss = 0
    # code omitted (no changes)
    # new code below:
    saver = tf.train.Saver(tf.all_variables())
    saver.save(session, './models/checkpoint.ckpt', write_meta_graph=False)
    tf.train.write_graph(graph.as_graph_def(), './models/', 'graph.pbtxt', as_text=True)


    from tf_model_save_load import tf_model

    saver_freeze = tf_model()
    saver_freeze.freeze_save(input_graph_filename='./models/graph.pbtxt',
                         input_binary=False,
                         checkpoint_path='./models/checkpoint.ckpt',
                         output_graph_filename='./models/graph_frozen.pb',
                         output_node_names='saved_sample_output',
                         clear_devices=False)
'''
####
'''
freeze_graph --input_graph=c:/models/graph.pbtxt --input_checkpoint=c:/models/checkpoint.ckpt 
                            --output_graph=c:/models/graph_frozen.pb 
                            --output_node_names=saved_sample_output

'''




''' Miscelanous '''
# has to be serialized before like:
# obj=pickle.dumps(obj)
def gzip_save(filename, obj,
              serialize=True,
              compresslevel=1,
              chunk = None #200*1024*1024 # 200MB
                ):
    import gzip, pickle
    if serialize:
        obj = pickle.dumps(obj)
    if not chunk:
        with gzip.open(filename, 'wb', compresslevel=compresslevel) as f:
            f.write(obj)
    else:
        with open(filename, 'rb') as infile:
            for n, raw_bytes in enumerate(iter(lambda: infile.read(chunk), b'')):
                print(n, chunk)
                with open('{}.part-{}'.format(filename[:-3], n), 'wb', compresslevel=compresslevel) as outfile:
                    outfile.write(raw_bytes)

def gzip_load(filename, serialize=True):
    import gzip, pickle
    with gzip.open(filename, 'rb') as f:
        obj = f.read()
    if serialize:
        return pickle.loads(obj)
    else:
        return obj


def pickle_save(filename, obj):
    import pickle
    f = open(filename, 'wb')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()
def pickle_load(filename):
    import pickle
    f = open(filename, 'rb')
    obj = pickle.load(f, fix_imports=True)
    f.close()
    return obj



''' Get total size of an object
https://goshippo.com/blog/measure-real-size-any-python-object/ '''
def get_size(obj, seen=None):
    import sys
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def is_in_ipynb():
    '''
    Check wether the working environment is Jupyter Notebook / Ipython or not
    :return: Boolean (True or False)
    '''
    from ipykernel.zmqshell import ZMQInteractiveShell as Zshell
    if 'get_ipython' in dir(Zshell):
        return True
    else:
        return False

def config_ini_to_dict(config_file):
    '''
    convert data from ini config file to a dictionary
    :param config: configparser.ConfigParser() object
    :return: dictionary

    example:
    import configparser
    config = configparser.ConfigParser()
    config.read('conf_clusters.ini')

    '''
    config = configparser.ConfigParser()
    if config.read(config_file)==[]:
        if config.read('./pda/'+config_file)==[]:
            print("config file not found!")
            return 0
    dictionary = dict()
    for section in config.sections():
        dictionary[section] = {}
        for option in config.options(section):
            str = config.get(section, option)
            # assign either int, or float, or string values
            try:
                dictionary[section][option] = int(str)
            except ValueError:
                try:
                    dictionary[section][option] = float(str)
                except ValueError:
                    dictionary[section][option] = str

    return dictionary

def load_spacenet_data(raw=True):
    """Loads spacenet dataset from `DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :type raw: `bool`
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    """

    # path = get_file('mnist.npz', path=DATA_PATH, url='https://s3.amazonaws.com/img-datasets/mnist.npz')
    from pda import dataset_file
    filename = os.path.join(model_path, dataset_file)

    if (True):  # pickle (spacenet_dict.pkl)
        # load pickle as images
        f = open(filename + '.pkl', 'rb')
        spacenet_dict = pickle.load(f, fix_imports=True)
        f.close()
    if(False):  # gzip (spacenet_dict.pkl.zip)
        spacenet_dict = gzip_load(filename + '.zip')
    if(False):
        f = np.load(filename + '.npz')
        spacenet_dict = f
        f.close()

    x_train = spacenet_dict['x_train']
    y_train = spacenet_dict['y_train']
    x_test = spacenet_dict['x_test']
    y_test = spacenet_dict['y_test']
    p_train = spacenet_dict['p_train']
    p_test = spacenet_dict['p_test']

    # Add channel axis
    if not raw:
        min_, max_ = 0., 1.
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)
        return (x_train, x_test), (y_train, y_test), (p_train, p_test), (min_, max_)
    else:
        min_, max_ = 0, 255
        return (x_train, x_test), (y_train, y_test), (p_train, p_test), (min_, max_)

def load_spacenet_model(model_name, path=None, max_=None, min_=None):
    from art.classifiers import KerasClassifier

    model_load = Save_Load_Model()
    if path!= None:
        model_load.path = path
    else:
        model_load.path=model_path
    model_load.fname= model_name+'.h5'
    model_p = model_load.kr_h5_loader()
    if (max_ and min_):
        # load clip values
        with open(model_path + model_name + '.txt', 'r') as clip_file:
            cv = clip_file.readlines()
            clip_file.close()
        max_ = float(cv[0].strip('\n'))
        min_ = float(cv[1])
    else:
        max_ = 255
        min_ = 0
    return KerasClassifier((min_, max_), model=model_p)

def load_spacenet_activation():
    filename = os.path.join(model_path, 'activations_flatten_2_1-Reshape')
    activations_flatten_train = pickle_load(filename)
    filename = os.path.join(model_path, 'activations_flatten_test')
    activations_flatten_testt = pickle_load(filename)
    return activations_flatten_train, activations_flatten_testt

def image_matrix(images):
    # https://stackoverflow.com/questions/19471814/display-multiple-images-in-one-ipython-notebook-cell
    from matplotlib.pyplot import figure, imshow, axis
    from matplotlib.image import imread

    mypath = '.'
    hSize = 5
    wSize = 5
    col = 4

    def showImagesMatrix(list_of_files, col=10):
        fig = figure(figsize=(wSize, hSize))
        number_of_files = len(list_of_files)
        row = number_of_files / col
        if (number_of_files % col != 0):
            row += 1
        for i in range(number_of_files):
            a = fig.add_subplot(row, col, i + 1)
            image = imread(mypath + '/' + list_of_files[i])
            imshow(image, cmap='Greys_r')
            axis('off')

    showImagesMatrix(images, col)

def images_show(images, y, p, figsize=(20, 20)):
    # https://stackoverflow.com/questions/19471814/display-multiple-images-in-one-ipython-notebook-cell
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    columns = 3
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        if p[i] == 1:
            is_poison = 'Poisoned'
        else:
            is_poison = 'Genuine'
        if y[i] == 0:
            is_building = '0: no building'
        else:
            is_building = '1: has building'
        plt.title('%s image, class%s' % (is_poison, is_building),
                  fontsize=18)
        plt.imshow(image)
    plt.show()

def csv_writer(fname=None, mode=None, myDict=None, header=None):
    import csv
    # create a csv file based on given header
    if mode == 'create':
        if not os.path.exists(fname):
            with open(fname, "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
        else:
            print('The csv file exists!')
    if mode == 'append':
        # read header automatically
        with open(fname, "r") as f:
            if not header:
                reader = csv.reader(f)
                for header in reader:
                    break

        # add row to CSV file
        with open(fname, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(myDict)