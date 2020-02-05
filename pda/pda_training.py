"""Generates poison data for Spacenet dataset (RGB 3-channels) and
then trains keras convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison.
 Uses 10-layer CNN model"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath
sys.path.append(abspath('.'))
import numpy as np
np.random.seed(44)

import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import load_model
from keras.models import model_from_json
from art.classifiers import KerasClassifier
from art.utils import load_mnist, preprocess
from art.poison_detection import ActivationDefence
from pda.pda_utils import pickle_save
import pprint
import json
from keras.utils import np_utils
import cv2
import os
import platform
import matplotlib.pyplot as plt
import pickle
from pda import model_path, data_path, FLAG_Ipynb
import configparser
import argparse

class PDA_Training (object):
    def __init__(self, dataset=None):

        if dataset:
            self.dataset = dataset #'spacenet' or 'mnist'
        else:
            self.dataset = 'spacenet'

        # parse from config file
        parser = configparser.ConfigParser()
        if FLAG_Ipynb:
            if parser.read('config_training.ini') == []:
                if parser.read('pda/config_training.ini') == []:
                    raise ValueError ('could not find config_general file!')
        else:
            if parser.read('config_training.ini') == []:
                if parser.read('pda/config_training.ini') == []:
                    raise ValueError('could not find config_general file!')
        self.model_name_clean = self.dataset + '_' + parser['default']['model_name_clean']
        self.model_name_poisoned = self.dataset + '_' + parser['default']['model_name_poisoned']
        self.epoch_nb = int(parser['training']['nb_epoch'])
        self.batch_size = int(parser['training']['batch_size'])
        self.nb_classes = int(parser['training']['nb_classes'])
        self.model_choice = parser['training']['model_choice']
        self.activation_train = parser['default']['activation_train']
        self.activation_test = parser['default']['activation_test']
        self.class1_dir = parser['dataset']['class1_dir']
        self.class0_dir = parser['dataset']['class0_dir']
        self.poison_dir = parser['dataset']['poison_dir']
        self.perc_poison = int(parser['dataset']['perc_poison'])
        self.FLAG_Train_Model = False
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.classifier = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.p_train = None
        self.p_test = None


        print('Dataset = ', self.dataset)

    def data_extractor_3ch(self, dataset=None, save=False):
        '''

        :param dataset:
        :param save: 'pickle', 'gzip', or 'npz' format
        :return:
        '''
        if (self.dataset == 'spacenet'):
            if platform.system() == 'Windows':
                base_path = 'd:/spacenet/'
            elif (platform.system() == 'Darwin'):
                base_path = '/Users/iman.zabett/Datasets/'
            # hasbuilding_dir = base_path + "hasbuilding_s"
            # nobuilding_dir = base_path + "nobuilding_s"
            cropped_dir = base_path + "cropped_limited_hasbuilding"
            cropped_no_dir = base_path + "cropped_limited_nobuilding"
            cropped_dir = self.data_path + self.class1_dir
            cropped_no_dir = self.data_path + self.class0_dir

            # poisoned dataset
            poison_cropped = base_path + 'poison_cropped'
            poison_cropped = self.data_path + self.poison_dir

            from os.path import join
            print('Loading images ...')
            # class 1
            images1 = [np.asarray(cv2.imread(join(cropped_dir, i)))
                       for i in os.listdir(cropped_dir)]
            labels1 = [1 for i in range(len(images1))]
            print(len(images1), "class 1, cropped images with building")
            # class 0
            images0 = [np.asarray(cv2.imread(join(cropped_no_dir, i)))
                       for i in os.listdir(cropped_no_dir)]
            print(len(images0), "class 0, cropped images with no building")
            labels0 = [0 for i in range(len(images0))]
            # images.extend(images2)
            # labels.extend(labels2)
            # X_data = images
            # Y_data = labels

            ### Poison training data
            # Adding prepared poisoned dataset
            print('Loading poisoned images ...')
            poisoned_images = [np.asarray(cv2.imread(os.path.join(poison_cropped, i)))
                               for i in os.listdir(poison_cropped)]
            poisoned_images = np.reshape(poisoned_images, [-1, 80, 80, 3])
            poisoned_labels = [0 for i in range(len(poisoned_images))]
            # poisoned_labels = np.reshape(poisoned_labels, [-1])
            print(len(poisoned_images), "poisoned images for no building class")

            def randomize(*args):
                """ Randomizes the order of data samples and their corresponding labels
                        Or several corresponding numpy arrays """
                if len(args) == 0:
                    return None
                    # print('No argument for function!')

                    # Iterate over all the arguments and calculate average
                permutation = np.random.permutation(args[0].shape[0])
                ret_arr = []
                for elem in args:
                    ret_arr.append(elem[permutation])

                return ret_arr

            # converting to numpy
            images1 = np.array(images1)
            labels1 = np.array(labels1)
            images0 = np.array(images0)
            labels0 = np.array(labels0)
            poisoned_images = np.array(poisoned_images)
            poisoned_labels = np.array(poisoned_labels)

            # Randomize all class1, class0, pois data
            images1, labels1 = randomize(images1, labels1)
            images0, labels0 = randomize(images0, labels0)
            poisoned_images, poisoned_labels = randomize(poisoned_images, poisoned_labels)

            # calculating minimum number of data to have a balanced dataset for both classes
            CLASS_NUM = min(len(images1), len(images0))
            print('CLASS_NUM is:', CLASS_NUM)
            # pick CLASS_NUM of data from class has building
            num_class_1 = CLASS_NUM
            # pick (1-poison_percentage)*CLASS_NUM of data from class no building
            # num_class_0 = round((1 - self.perc_poison)*CLASS_NUM)
            num_class_0 = CLASS_NUM
            # percentage of generating poisoned data to CLASS_NUM

            print('poison image percentage: ', self.perc_poison)
            # number of poisoned data for class 0
            num_pois = round(self.perc_poison/100 * CLASS_NUM)
            ### slicing datasets
            images1 = images1[:num_class_1]
            labels1 = labels1[:num_class_1]
            images0 = images0[:num_class_0]
            labels0 = labels0[:num_class_0]
            poisoned_images = poisoned_images[:num_pois]
            poisoned_labels = poisoned_labels[:num_pois]

            images0 = np.append(images0, poisoned_images, axis=0)
            labels0 = np.append(labels0, poisoned_labels, axis=0)

            images = np.append(images1, images0, axis=0)
            labels = np.append(labels1, labels0, axis=0)
            # creating is_poison index showing which data is poisoned
            is_poison = np.append(np.zeros(num_class_1 + num_class_0),
                                  np.ones(num_pois) * 1,
                                  axis=0).astype('uint8')
            # shuffling all
            images, labels, is_poison = randomize(images, labels, is_poison)
            labels = np_utils.to_categorical(labels, self.nb_classes)

            from sklearn.model_selection import train_test_split
            Random_Seed = None
            x_train, x_test, y_train, y_test, p_train, p_test = \
                train_test_split(images, labels, is_poison,
                                 test_size=0.2,
                                 random_state=Random_Seed)

            # ### Reports
            # print('\n\n### DATA REPORTS ###')
            # print('"Total Dataset" has %d data.\n'
            #       '%d for class1\n%d for class0\n'
            #       '%d number of poisoned images\n'
            #       '%.3f percentage of poisoned images'
            #       % (labels.shape[0],
            #          labels[labels[:, 0] == 1].shape[0],
            #          labels[labels[:, 0] == 0].shape[0],
            #          is_poison.sum(),
            #          (is_poison.sum() / p_train.shape[0]) * 100))
            # print()
            # print('\n"x_train" has total of %d train data.\n'
            #       '%d for class1\n%d for class0\n'
            #       '%d number of poisoned images\n'
            #       '%.3f percentage of poisoned images'
            #       % (x_train.shape[0],
            #          x_train[y_train[:, 0] == 1].shape[0],
            #          x_train[y_train[:, 0] == 0].shape[0],
            #          p_train.sum(),
            #          (p_train.sum() / p_train.shape[0]) * 100))
            # print()
            # print('\n"x_test" has total of %d test data.\n'
            #       '%d for class1\n%d for class0\n'
            #       '%d number of poisoned images\n'
            #       '%.3f percentage of poisoned images'
            #       % (x_test.shape[0],
            #          x_test[y_test[:, 0] == 1].shape[0],
            #          x_test[y_test[:, 0] == 0].shape[0],
            #          p_test.sum(),
            #          (p_test.sum() / p_test.shape[0]) * 100))
            # print('percentage and number of poisoned images in train is: \n%.3f and %d'
            #       % ((p_train.sum() / p_train.shape[0]) * 100.0, p_train.sum()))
            # print('percentage and number of poisoned images in test is: \n%.3f and %d'
            #       % ((p_test.sum() / p_test.shape[0]) * 100.0, p_test.sum()))

            # choice of random slice of dataset
            if (False):
                n_train = np.shape(x_raw)[0]
                num_selection = 500
                random_selection_indices = np.random.choice(n_train, num_selection)
                x_raw_train = x_raw_train[random_selection_indices]
                y_raw_train = y_raw_train[random_selection_indices]

            # save dataset
            if (save):
                from pda import dataset_file
                filename = os.path.join(self.model_path, dataset_file)
                spacenet_dict = {
                    'x_train': x_train,
                    'x_test':  x_test,
                    'y_train': y_train,
                    'y_test':  y_test,
                    'p_train': p_train,
                    'p_test':  p_test}
                # pickle obj.x_train as images
                if (save=='pickle'):  # pickle
                    f = open(filename+'.pkl', 'wb')
                    pickle.dump(spacenet_dict, f, pickle.HIGHEST_PROTOCOL)
                    f.close()
                if(save=='gzip'):  # gzip
                    from pda.pda_utils import gzip_save
                    gzip_save(filename+'.zip', spacenet_dict, serialize=True, compresslevel=9)
                if(save=='npz'): # npz
                    np.savez(filename+'.npz', spacenet_dict)
                # images_c = obj.x_train[obj.p_train == 0]
                # images_p = obj.x_train[obj.p_train == 1]

            # sources = np.arange(n_classes)  # [0,1]
            # targets = (np.arange(n_classes) + 1) % n_classes  # [1,0]
            sources = 1
            targets = 0

            # building train set

            # num_poison_train = x_poisoned_raw_train.shape[0]
            # num_poison_train = 300
            # print ('percentage and number of poisoned images in train is: \n%d and %d'
            #        %(perc_poison, num_poison_train))
            # x_train = np.append(x_raw_train, x_poisoned_raw_train[:num_poison_train], axis=0)
            # p_train = np.append(np.zeros(len(y_raw_train)),
            #                             np.ones(num_poison_train) * 1, axis=0).astype('uint8')
            # y_poisoned_raw_train = np.ones(num_poison_train) * 0 # poisoned no building
            # y_train = np.append(y_raw_train, y_poisoned_raw_train, axis=0)
            # randomizing
            # x_train, y_train, p_train = randomize(x_train, y_train, p_train)
            # y_train = np_utils.to_categorical(y_train, 2)

            # building test set
            # num_poison_test = x_poisoned_raw_test.shape[0]
            # print ('percentage and number of poisoned images in test is: \n%d and %d'
            #        %(perc_poison, num_poison_test))
            # x_test = np.append(x_raw_test, x_poisoned_raw_test[:num_poison_test], axis=0)
            # p_test = np.append(np.zeros(len(y_raw_test)),
            #                            np.ones(num_poison_test) * 1, axis=0).astype('uint8')
            # y_poisoned_raw_test = np.ones(num_poison_test) * 0 # poisoned no building
            # y_test = np.append(y_raw_test, y_poisoned_raw_test, axis=0)
            # randomizing
            # x_test, y_test, y_poisoned_raw_test = randomize(x_test, y_test, p_test)
            # y_test = np_utils.to_categorical(y_test, 2)

            # print('"x_train" has %d poisoned+clean train data.'
            #       '\n%d for class1\n%d for class0\n'
            #       % (x_train.shape[0],
            #          x_train[y_train[:,0].astype('uint8') == 1].shape[0],
            #          x_train[y_train[:,0].astype('uint8') == 0].shape[0]))
            # print('"x_test" has %d poisoned+clean train data.'
            #       '\n%d for class1\n%d for class0\n'
            #       % (x_test.shape[0],
            #          x_test[y_test[:,0].astype('uint8') == 1].shape[0],
            #          x_test[y_test[:,0].astype('uint8') == 0].shape[0]))
            # Generate and adding simple poisoned data

        if (self.dataset == 'mnist'):
            self.nb_classes = 10
            sources = [0,1,2,3,4,5,6,7,8,9]
            targets = [1,2,3,4,5,6,7,8,9,0]

            # Read MNIST dataset (x_raw contains the original images):
            (x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)

            n_train = np.shape(x_raw)[0]
            num_selection = 5000
            random_selection_indices = np.random.choice(n_train, num_selection)
            x_raw = x_raw[random_selection_indices]
            y_raw = y_raw[random_selection_indices]

            # Poison training data
            perc_poison = self.perc_poison / 100
            (p_train, x_poisoned_raw, y_poisoned_raw) = \
                generate_backdoor(x_raw, y_raw, perc_poison, sources=sources , targets=targets)
            x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
            # Add channel axis:
            x_train = np.expand_dims(x_train, axis=3)

            # Poison test data
            (p_test, x_poisoned_raw_test, y_poisoned_raw_test) = \
                generate_backdoor(x_raw_test, y_raw_test, perc_poison, sources=sources , targets=targets)
            x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
            # Add channel axis:
            x_test = np.expand_dims(x_test, axis=3)

            # Shuffle training data so poison is not together
            n_train = np.shape(y_train)[0]
            shuffled_indices = np.arange(n_train)
            np.random.shuffle(shuffled_indices)
            x_train = x_train[shuffled_indices]
            y_train = y_train[shuffled_indices]
            p_train = p_train[shuffled_indices]


        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.p_train = p_train
        self.p_test = p_test

    def model_training(self, data_dict, is_train=None, save=False, filename=''):
        x_train = data_dict['x_train']
        x_test = data_dict['x_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        p_train = data_dict['p_train']
        p_test = data_dict['p_test']

        # is_train = self.FLAG_Train_Model

        # loading pretrained model
        if not (is_train):
            print("Loading clean model from disk ...")
            # load clip values
            with open(self.model_path + '_' + self.model_name_clean + '.txt', 'r') as clip_file:
                cv = clip_file.readlines()
                clip_file.close()
            max_ = float(cv[0].strip('\n'))
            min_ = float(cv[1])
            if(False):
                # load json and create model
                json_file = open(self.model_path + '_' + self.model_name_clean + '.json', 'r')
                model_json = json_file.read()
                json_file.close()
                model = model_from_json(model_json)
                # load weights into new model
                model.load_weights(self.model_path + '_' + self.model_name_clean+'.h5')
            # load model+weights from single h5 file
            model_c = load_model(self.model_path + '_' + self.model_name_clean+'.h5')
            classifier_c = KerasClassifier((min_, max_), model=model_c)

            print("Loading poisoned model from disk ...")
            # load clip values
            with open(self.model_path + '_' +  self.model_name_poisoned + '.txt', 'r') as clip_file:
                cv = clip_file.readlines()
                clip_file.close()
            max_ = float(cv[0].strip('\n'))
            min_ = float(cv[1])
            if(False):
                # load json and create model
                json_file = open(self.model_path+'_'+self.model_name_poisoned+'.json', 'r')
                model_json = json_file.read()
                json_file.close()
                model = model_from_json(model_json)
                # load weights into new model
                model.load_weights(self.model_path+'_'+self.model_name_poisoned+'.h5')
            # load model+weights from single h5 file
            model_p = load_model(self.model_path+'_'+self.model_name_poisoned+'.h5')
            classifier_p = KerasClassifier((min_, max_), model=model_p)

        if(is_train):# Train the models

            # Create Keras convolutional neural network - basic architecture from Keras examples
            # Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
            def keras_vgg16_std(input_shape, nb_classes):
                import keras.applications.vgg16 as vgg16
                model = vgg16.VGG16(include_top=True,
                                      weights=None,
                                      input_tensor=None,
                                      input_shape=input_shape,
                                      pooling=None,
                                      classes=nb_classes)
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam', metrics=['accuracy'])
                return model
            def keras_model(input_shape, nb_classes):
                k.set_learning_phase(1)
                model = Sequential()
                model.add(Conv2D(filters=80, kernel_size=(3, 3), padding='same',
                                 activation='relu', input_shape=input_shape))
                model.add(Conv2D(filters=80, kernel_size=(3, 3), padding='same',
                                 activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Conv2D(filters=40, kernel_size=(3, 3), padding='same',
                                 activation='relu'))
                model.add(Conv2D(filters=40, kernel_size=(3, 3), padding='same',
                                 activation='relu'))
                model.add(Conv2D(filters=40, kernel_size=(3, 3), padding='same',
                                 activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Conv2D(filters=20, kernel_size=(3, 3), padding='same',
                                 activation='relu'))
                model.add(Conv2D(filters=20, kernel_size=(3, 3), padding='same',
                                 activation='relu'))
                model.add(Conv2D(filters=20, kernel_size=(3, 3), padding='same',
                                 activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
                model.add(Dropout(0.25))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.25))
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.25))
                model.add(Dense(nb_classes, activation='softmax'))
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam', metrics=['accuracy'])
                return model
            def keras_vgg16(input_shape, nb_classes):
                # https://engmrk.com/vgg16-implementation-using-keras/
                k.set_learning_phase(1)
                model = Sequential([
                Conv2D(filters=64, kernel_size=(3, 3),
                       padding='same', activation ='relu', input_shape=input_shape),
                Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding ='same'),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding ='same'),
                Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding ='same', ),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding ='same', ),
                Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding ='same', ),
                Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding ='same', ),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding ='same', ),
                Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding ='same', ),
                Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding ='same', ),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding ='same', ),
                Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding ='same', ),
                Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding ='same', ),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Flatten(),
                Dense(4096, activation='relu'),
                Dense(4096, activation='relu'),
                Dense(nb_classes, activation='softmax')
                    ])

                # model.summary()

                # Compile the model
                from keras.optimizers import RMSprop
                model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.1),
                              metrics = ['accuracy'])
                return model

            print('Epoch: ', self.epoch_nb)
            print('Batch_size: ', self.batch_size)
            print('Number of classes: ', self.nb_classes)

            # print('### Train classifier for only clean data')
            # min_, max_ = x_train[p_train == 0].min(), \
            #              x_train[p_train == 0].max()
            # model_c = keras_model(x_train.shape[1:], self.nb_classes)
            # classifier_c = KerasClassifier((min_, max_), model=model_c)
            # classifier_c.fit(x_train[p_train == 0], y_train[p_train == 0],
            #                nb_epochs=self.epoch_nb, batch_size=self.batch_size, shuffle=False, verbose=1,
            #                  validation_data=(x_test[p_test == 0], y_test[p_test == 0]))
            #
            # if(save):
            #     print("Saving clean model to disk ...")
            #     if (False):
            #         # serialize model to JSON
            #         model_json = model_c.to_json()
            #         with open(self.model_path + '_'+self.model_name_clean +'.json', "w") as json_file:
            #             json_file.write(model_json)
            #             json_file.close()
            #         # serialize weights to HDF5
            #         model_c.save_weights(self.model_path + '_'+self.model_name_clean + '.h5')
            #     # save classifier
            #     pickle_save(self.model_path + 'classifier_c.pkl', classifier_c)
            #     # save model + weights in a single HDF5 file
            #     model_c.save(self.model_path + '_'+self.model_name_clean + '.h5')
            #     # saving clip values
            #     with open(self.model_path + '_'+self.model_name_clean + '.txt', 'w') as clip_file:
            #         clip_file.write(str(max_) + '\n')
            #         clip_file.write(str(min_))
            #         clip_file.close()

            print('Train classifier for ' + filename)
            min_, max_ = x_train.min(), x_train.max()
            if (self.model_choice=='10-CNN'):
                model = keras_model(x_train.shape[1:], self.nb_classes)
            if (self.model_choice=='keras_vgg16'):
                model = keras_vgg16(x_train.shape[1:], self.nb_classes)
            classifier = KerasClassifier((min_, max_), model=model)
            classifier.fit(x_train, y_train, nb_epochs=self.epoch_nb,
                             batch_size=self.batch_size, shuffle=False, verbose=1)
                             # validation_data=(x_test, y_test))
            if(save):
                print("Saving model to disk ...")
                if(False):
                    # serialize model to JSON
                    model_json = model.to_json()
                    with open(self.model_path + filename +'.json', "w") as json_file:
                        json_file.write(model_json)
                        json_file.close()
                    # serialize weights to HDF5
                    model.save_weights(self.model_path + filename + '.h5')
                # save classifier
                pickle_save(self.model_path + filename + '.pkl', classifier)
                # save model + weights in a single HDF5 file
                model.save(self.model_path + filename + '.h5')
                # saving clip values
                with open(self.model_path + filename + '.txt', 'w') as clip_file:
                    clip_file.write(str(max_) + '\n')
                    clip_file.write(str(min_))
                    clip_file.close()

        k.set_learning_phase(0)

        self.classifier = classifier
        self.model = model

        return classifier
        # print('\n\n ### Evaluate Clean Classifer:')
        # # Evaluate the classifier on the test set
        # preds = np.argmax(classifier_c.predict(x_test), axis=1)
        # acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        # print("Test accuracy (clean + poisoned): %.2f%%" % (acc * 100))
        #
        # # Evaluate the classifier on poisonous data
        # preds = np.argmax(classifier_c.predict(x_test[p_test]), axis=1)
        # acc = np.sum(preds == np.argmax(y_test[p_test], axis=1)) / y_test[p_test].shape[0]
        # print("Test accuracy (only poisoned)(effectiveness of poison): %.2f%%" % (acc * 100))
        #
        # # Evaluate the classifier on clean data
        # preds = np.argmax(classifier_c.predict(x_test[p_test == 0]), axis=1)
        # acc = np.sum(preds == np.argmax(y_test[p_test == 0], axis=1)) / y_test[p_test == 0].shape[0]
        # print("Test accuracy (only clean): %.2f%%" % (acc * 100))
        #
        # score = model_c.evaluate(x_test, y_test, verbose=0)
        # print('Test loss model_c:', score[0])
        # print('Test accuracy model_c:', score[1])

    def model_evaluate(self, test_dict, classifier, model=None):
        x_test = test_dict['x_test']
        y_test = test_dict['y_test']
        p_test = test_dict['p_test']

        # Evaluate the classifier on the test set
        preds = np.argmax(classifier.predict(x_test), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("Test accuracy (clean + poisoned): %.2f%%" % (acc * 100))

        # Evaluate the classifier on poisonous data
        preds = np.argmax(classifier.predict(x_test[p_test]), axis=1)
        acc = np.sum(preds == np.argmax(y_test[p_test], axis=1)) / y_test[p_test].shape[0]
        print("Test accuracy (only poisoned)(effectiveness of poison): %.2f%%" % (acc * 100))

        # Evaluate the classifier on clean data
        preds = np.argmax(classifier.predict(x_test[p_test == 0]), axis=1)
        acc = np.sum(preds == np.argmax(y_test[p_test == 0], axis=1)) / y_test[p_test == 0].shape[0]
        print("Test accuracy (only clean): %.2f%%" % (acc * 100))

        if model:
            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss model:', score[0])
            print('Test accuracy model:', score[1])

    def poison_detection(self):
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
        p_train = self.p_train
        p_test = self.p_test
        classifier_p = self.classifier_p
        model_p = self.model_p
        # Calling poisoning defence:
        defence = ActivationDefence(classifier_p, x_train, y_train)

        # End-to-end method:
        print("\n\n------------------- Results using size metric -------------------")
        # print(defence.get_params())
        '''
        Valid Params
        defence_params = [  'nb_clusters', 'clustering_method', 
                            'nb_dims', 'reduce', 'cluster_analysis']
        method = ['KMeans']
        reduce = ['PCA', 'FastICA', 'TSNE']
        analysis = ['smaller', 'distance', 'relative-size', 'silhouette-scores']
        '''
        defence.detect_poison(n_clusters=2, ndims=10,
                              reduce="PCA", cluster_analysis='smaller')
        if (False):
            # plotting clusters datapoints
            defence.plot_clusters()
        # Evaluate method when ground truth is known:
        if (False):
            is_clean = (p_train == 0)
            confusion_matrix = defence.evaluate_defence(is_clean)
            print("Evaluation defence results for size-based metric: ")
            jsonObject = json.loads(confusion_matrix)
            for label in jsonObject:
                print(label)
                pprint.pprint(jsonObject[label])
            import matplotlib.pyplot as plt
            plot_confusion_matrix(confusion_matrix,
                                  nb_classes=self.nb_classes,
                                  target_class=0,
                                  normalize=True,
                                  title='Defence for size-based metric, class_0')
            plot_confusion_matrix(confusion_matrix,
                                  nb_classes=self.nb_classes,
                                  target_class=1,
                                  normalize=True,
                                  title='Defence for size-based metric, class_1')
            plt.show()

        # Visualize clusters:
        if (False):
            print("Visualize clusters")
            sprites_by_class = defence.visualize_clusters(x_train, 'spacenet_poison_size')
            # Show plots for clusters of class 1
            n_class = 1
            try:
                plt.imshow(sprites_by_class[n_class][0])
                plt.title("Class " + str(n_class) + " cluster: 0")
                plt.show()
                plt.imshow(sprites_by_class[n_class][1])
                plt.title("Class " + str(n_class) + " cluster: 1")
                plt.show()
            except:
                print("matplotlib not installed. "
                      "For this reason, cluster visualization was not displayed")

        # Try again using distance analysis this time:
        print("------------------- Results using distance metric -------------------")
        # print(defence.get_params())
        defence.detect_poison(n_clusters=2, ndims=10, reduce="PCA",
                              cluster_analysis='distance')
        if (False):
            # plotting clusters datapoints
            defence.plot_clusters()
        # Evaluate method when ground truth is known:
        if(False):
            confusion_matrix = defence.evaluate_defence(is_clean)
            print("Evaluation defence results for distance-based metric: ")
            # import pprint
            # jsonObject = json.loads(confusion_matrix)
            # for label in jsonObject:
            #     print(label)
            #     pprint.pprint(jsonObject[label])
            plot_confusion_matrix(confusion_matrix, nb_classes=self.nb_classes,
                                  target_class=0,
                                  normalize=True,
                                  title='Defence for distance-based metric, class_0')
            plot_confusion_matrix(confusion_matrix, nb_classes=self.nb_classes,
                                  target_class=1,
                                  normalize=True,
                                  title='Defence for distance-based metric, class_1')
        # Visualize clusters:
        if (False):
            print("Visualize clusters")
            sprites_by_class = defence.visualize_clusters(x_train, 'spacenet_poison_distance')
            # Show plots for clusters of class 1
            n_class = 1
            try:
                import matplotlib.pyplot as plt
                plt.imshow(sprites_by_class[n_class][0])
                plt.title("Class " + str(n_class) + " cluster: 0")
                plt.show()
                plt.imshow(sprites_by_class[n_class][1])
                plt.title("Class " + str(n_class) + " cluster: 1")
                plt.show()
            except:
                print("matplotlib not installed. "
                      "For this reason, cluster visualization was not displayed")

        # Other ways to invoke the defence:
        if(False):
            defence.cluster_activations(n_clusters=2, ndims=10, reduce='PCA')

            defence.analyze_clusters(cluster_analysis='distance')
            defence.evaluate_defence(is_clean)

            defence.analyze_clusters(cluster_analysis='smaller')
            defence.evaluate_defence(is_clean)

            plt.show()
        print("done :) ")

    def get_dict(self):
        return dict(
            x_train=self.x_train,
            x_test=self.x_test,
            y_train=self.y_train,
            y_test=self.y_test,
            p_train=self.p_train,
            p_test=self.p_test
        )
    def set_dict(self, dict):
        self.x_train = dict['x_train']
        self.y_train = dict['y_train']
        self.x_test = dict['x_test']
        self.y_test = dict['y_test']
        self.p_train = dict['p_train']
        self.p_test = dict['p_test']
    def get_dict_clean(self):
        return dict(
            x_train=self.x_train[self.p_train==0],
            x_test=self.x_test,
            y_train=self.y_train[self.p_train==0],
            y_test=self.y_test,
            p_train=self.p_train,
            p_test=self.p_test
        )
    def data_stats(self):
        print('Number of total data: ', self.x_train.shape[0] + self.x_test.shape[0])
        print('Number of total Poisonous data: ', self.p_train.cumsum()[-1])
        print('Number of Training data: ', self.x_train.shape[0])
        print('Number of Training Data in Class 0: ', self.x_train[self.y_train[:, 1] == 0].shape[0])
        print('Number of Training Data in Class 1: ', self.x_train[self.y_train[:, 1] == 1].shape[0])
        print('Number of Poisonous data in Class 0: ', self.p_train[self.y_train[:, 1] == 0].cumsum()[-1])
        print('Number of Poisonous data in Class 1: ', self.p_train[self.y_train[:, 1] == 1].cumsum()[-1])
        print('Number of Testing  data: ', self.x_test.shape[0])
        print('Percentage of poisoned images: %.3f' % ((self.p_train.sum() / self.p_train.shape[0]) * 100))

def plot_confusion_matrix(confusion_matrix,
                          target_class=None,
                          nb_classes=2,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    jsonObject = json.loads(confusion_matrix)
    classes = [i for i in jsonObject.keys()]
    cm = []
    for i in jsonObject:
        for j in jsonObject[i]:
            for k in jsonObject[i][j]:
                if k == 'rate':
                    cm.append(jsonObject[i][j][k])
                    # print(i,j,k,jsonObject[i][j][k])
    cm = np.array(cm)
    if target_class==0:
        cm = cm[:nb_classes**2].reshape((nb_classes, nb_classes)) # class 0
    if target_class==1:
        cm = cm[nb_classes**2:].reshape((nb_classes, nb_classes)) # class 1

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def generate_backdoor(x_clean, y_clean, percent_poison,
                      backdoor_type='pattern',
                      sources=np.arange(10),
                      targets=(np.arange(10) + 1) % 10):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """

    max_val = np.max(x_clean)

    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))

    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[y_clean == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == 'pattern':
            imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == 'pixel':
            imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, np.ones(num_poison) * tgt, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison

def add_single_bd(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for single images
    or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`

    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`

    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`

    :return: augmented matrix
    :rtype: `np.ndarray`
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 3:
        width, height = x.shape[1:]
        x[:, width - distance, height - distance] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


def add_pattern_bd(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`
    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`
    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`
    :return: augmented matrix
    :rtype: np.ndarray
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 3:
        width, height = x.shape[1:]
        x[:, width - distance, height - distance] = pixel_value
        x[:, width - distance - 1, height - distance - 1] = pixel_value
        x[:, width - distance, height - distance - 2] = pixel_value
        x[:, width - distance - 2, height - distance] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
        x[width - distance - 1, height - distance - 1] = pixel_value
        x[width - distance, height - distance - 2] = pixel_value
        x[width - distance - 2, height - distance] = pixel_value
    elif len(shape) == 4: # for 3 channels RGB
        width, height = x.shape[1:3]
        width, height = x.shape[1:]
        wd = width - distance
        hd = height - distance
        x[:, wd - 15:wd, hd - 3:hd, :] = pixel_value
        x[:, wd - 3:wd, hd - 15:hd, :] = pixel_value
        x[:, wd:wd + 10, hd:hd + 3, :] = pixel_value
        x[:, wd:wd + 3, hd:hd + 10, :] = pixel_value

        # x[:, width - distance, height - distance, :] = pixel_value
        # x[:, width - distance - 1, height - distance - 1, :] = pixel_value
        # x[:, width - distance, height - distance - 2, :] = pixel_value
        # x[:, width - distance - 2, height - distance, :] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config',
                        help='If the parameters is extracted from config file. '
                             'If True, then the command line parameters will be bypassed. '
                             'If False, then user needs to pass parameters from command line.',
                        type=bool,
                        default=True)
    parser.add_argument('-model_path',
                        help='The path to the project folder.')
    parser.add_argument('-n_gram',
                        help='selecting n_gram for feature selection. n_gram=3 means the algorithm'
                             'will calculate and combine 1, 2, and 3 grams together',
                        type=int,
                        default=3)
    parser.add_argument('-num_class',
                        help='defining how many classes the dataset has',
                        type=int,
                        default=2)
    parser.add_argument('-out_path',
                        help='this is the path to store feature names and featuer matrix files',
                        type=str,
                        default=None)
    args = parser.parse_args()

