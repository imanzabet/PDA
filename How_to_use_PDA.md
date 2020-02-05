## How to use Poison Defense Analytic (PDA)
In this tutorial, we try to describe how to set up PDA environment for processing data, training, clustering, and detecting poisonous data.

### A. Config files

In `pda` path, there are four config files (`.ini`) which need to be set first.

`./pda/...` <br />
`./pda/config_general.ini` <br />
`./pda/config_training.ini` <br />
`./pda/config_clustering.ini` <br />
`./pda/config_dimred.ini` <br />

##### a. config_general.ini
In this file, we need to define some general variables for either `win` or `mac` operating system:
<br />
1- **`data_path`**: This directory holds the entire dataset in three different directory. 
`class 1`, `class 0`, and `poison` directory. The name for each directory will be defined correspondingly 
in `config_training.ini`. <br />
Also, this directory is working directory for PDA package and will used to store generated temporary files.
<br /><br />
2- **`model_path`**: This directory holds: <br />
a. saved/pickled `variables` <br />
b. saved/pickled `deep learning models` <br />
for further usages.


##### b. config_training.ini
This config file refers to settings for training section.
1. **`[default]`** section consists of names for stored trained deep learning models and calculated activations.

2. **`[dataset]`** section consists of directory names for `class 1`, `class 0`, and `poison` dataset. <br />
These directories are being used as subdirectory in `data_path` definded in `./pda/config_general.ini`.
In order to train models, we need to put our dataset inside relevant subdirectories.

3. **`[training]`** section refers to the hyperparameters using to train the deep learning model. <br />
`model_choice` is referring to the specific architecture being used to build AI-Model. Such as  `10-CNN` or `vgg_16`. <br />
`nb_epoch`, `batch_size` are referring to number of epochs and size of batch to train model. <br />
`nb_classes` referrs to number of classes that AI-Model supposed to classify. For our problem `nb_classes = 2`.


##### c. config_clustering.ini
This config file contains all the necessary hyper-parameters we need for clustering algorithms.
1. **`[KMeans]`** section contains of parameters using for KMeans clustering algorithm. <br />
2. **`[DBSCAN]`** section contains of parameters using for DBSCAN clustering algorithm. <br />
3. **`[AgglomerativeClustering]`** section contains of parameters using for Agglomerative clustering algorithm.

##### d. config_dimred.ini
In this config file, we choose specific dimensionality reduction algorithm with its relevant hyper-parameters.
1. **`[PCA]`** section contains of parameters using for PCA algorithm. <br />
2. **`[ICA]`** section contains of parameters using for ICA algorithm. <br />
3. **`[TSNE]`** section contains of parameters using for TSNE algorithm. <br />


### B. Setting up config files
Many of the parameters in the config files are optional to change, here we emphasize those params
which need to be set before using PDA.
##### a. data_path
##### b. model_path
##### c. dataset directory's names
All `class1_dir`, `class0_dir`, `poison_dir` for dataset need to be set.


### C. Using Command line
PDA also designed to be used with command line. In this case, python script called `pda_c.py` in the root folder of PDA will be used.<br/>

The command line needs to have paths to 4 filename and directories.
(`p_model`, `data_class1`, `data_class0`, `data_poison`) and output path to `new_model` filename.

It also accepts `p_perc` as poisonous data percentage and `clustering_method` as optional arguments.

##### Example
`python pda_c.py --config=False -m D:\PycharmData\Sat_Img\models\spacenet_poison_model -db D:\Spacenet\cropped_limited_hasbuilding -dn D:\Spacenet\cropped_limited_nobuilding -dp D:\Spacenet\poison_cropped -n D:\Spacenet\new_model`

##### Help (`python pda_c.py --help`)
```
usage: pda_c.py [-h] [--config {True,False}] [-m P_MODEL] [-db DATA_CLASS1]
                [-dn DATA_CLASS0] [-dp DATA_POISON] [-r P_PERC]
                [-l CLUSTERING_METHOD] [-n NEW_MODEL]

This is PDA package command line description.

optional arguments:
  -h, --help            show this help message and exit
  --config {True,False}
                        Whether the parameters is extracted from config file
                        or defined by command line. If "True", then the
                        command line parameters will be bypassed. If "False",
                        then user needs to pass parameters from command line.
  -m P_MODEL, --p_model P_MODEL
                        The path to the poisonous model.
  -db DATA_CLASS1, --data_class1 DATA_CLASS1
                        The path to the class 1 (building) of dataset.
  -dn DATA_CLASS0, --data_class0 DATA_CLASS0
                        The path to the class 0 (no building) of dataset.
  -dp DATA_POISON, --data_poison DATA_POISON
                        The path to the poison directory of dataset.
  -r P_PERC, --p_perc P_PERC
                        The percentage of added poison images in class 0.
  -l CLUSTERING_METHOD, --clustering_method CLUSTERING_METHOD
                        choose of clustering algorithm among KMeans, DBSCAN,
                        AgglomerativeClustering.
  -n NEW_MODEL, --new_model NEW_MODEL
                        this is the path to store new_model and generated data

The command line needs to have paths to 4 filename and directories (p_model,
data_class1, data_class0, data_poison) and output path to "new_model"
filename. It also accepts "p_perc" as poisonous data percentage and
"clustering_method" as optional arguments.`
```