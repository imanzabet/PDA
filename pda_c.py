import argparse

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--option1", dest="foo")
    # parser.add_argument("--option2", dest="foo")
    # parser.parse_args(['--option2', 'bar', '--option1', 'baz'])

    parser = argparse.ArgumentParser(
         # prog='Poison Defense Analytics',
         description='''This is PDA package command line description.''',
         epilog=''' The command line needs to have paths to 4 filename and directories
         (p_model, data_class1, data_class0, data_poison) and output path to "new_model" filename.
         It also accepts "p_perc" as poisonous data percentage and "clustering_method" as optional arguments.
          ''')
    parser.add_argument('--config',
                        help='Whether the parameters is extracted from config file or defined by command line. '
                             'If "True", then the command line parameters will be bypassed. '
                             'If "False", then user needs to pass parameters from command line.',
                        choices=('True','False'),
                        default='True')
    parser.add_argument('-m','--p_model',
                        help='The path to the poisonous model.',
                        type=str,
                        default=None)
    parser.add_argument('-db','--data_class1',
                        help='The path to the class 1 (building) of dataset.',
                        type=str,
                        default=None)
    parser.add_argument('-dn','--data_class0',
                        help='The path to the class 0 (no building) of dataset.',
                        type=str,
                        default=None)
    parser.add_argument('-dp','--data_poison',
                        help='The path to the poison directory of dataset.',
                        type=str,
                        default=None)
    parser.add_argument('-r','--p_perc',
                        help='The percentage of added poison images in class 0.',
                        type=int,
                        default=30)
    parser.add_argument('-l','--clustering_method',
                        help='choose of clustering algorithm among KMeans, DBSCAN, AgglomerativeClustering.',
                        type=str,
                        default='KMeans')
    parser.add_argument('-n','--new_model',
                        help='this is the path to store new_model and generated data',
                        type=str,
                        default='.')
    args = parser.parse_args()

    if (args.config=='False' or args.config=='false'):
        print('Reading parameters from command line ...')
        print('Extracting dataset ...')
        from pda.pda_training import PDA_Training
        prep = PDA_Training(dataset='spacenet')
        prep.class1_dir = args.data_class1
        prep.class0_dir = args.data_class0
        prep.poison_dir = args.data_poison
        prep.data_path = ''
        prep.perc_poison = args.p_perc
        prep.data_extractor_3ch()
        prep.data_stats()

        # from pda.pda_utils import load_spacenet_data
        # # load spacenet dataset
        # (x_train, x_test), (y_train, y_test), (p_train, p_test), (_, _) = load_spacenet_data()

        print('Loading Pre-trained poisonous model ...')
        from pda.pda_utils import load_spacenet_model
        model_name = args.p_model
        classifier_p = load_spacenet_model(model_name, path='')
        print('List of model layers:')
        classifier_p._model.layers

        print('Applying Defense ...')
        from pda.pda import PDA
        defense = PDA(classifier_p, prep.x_train, prep.y_train)
        clustering_method = args.clustering_method
        from pda.pda_utils import config_ini_to_dict
        clustering_hyparam = config_ini_to_dict(config_file='config_clustering.ini')[clustering_method]
        defense.detect_poison(n_clusters=2, ndims=10, clustering_method=clustering_method,
                              clustering_hyparam=clustering_hyparam,
                              reduce="PCA", cluster_analysis='smaller')

        print('Clustering results ...')
        import json, pprint

        is_clean = (prep.p_train == 0)
        confusion_matrix = defense.evaluate_defence(is_clean)
        print("Evaluation defense results for size-based metric: ")
        jsonObject = json.loads(confusion_matrix)
        for label in jsonObject:
            print(label)
            pprint.pprint(jsonObject[label])

        print('Training New_Model ...')
        from pda.pda_training import PDA_Training

        prep = PDA_Training(dataset='spacenet')
        # remove poisonous data
        x_train_new, y_train_new = defense.remove_poisons()
        cleaned_data = dict(
            x_train=x_train_new,
            x_test=prep.x_test,
            y_train=y_train_new,
            y_test=prep.y_test,
            p_train=prep.p_train,
            p_test=prep.p_test
        )
        prep.model_path = ''
        new_model_path = args.new_model
        prep.model_choice = '10-CNN'
        prep.epoch_nb = 10
        classifier_new = prep.model_training(cleaned_data, is_train=True, save=True,
                                             filename=new_model_path)