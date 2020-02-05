## Poison Defense Analytic (PDA)
This package aims to build a robust AI-Model which will be robust against 
poison attacks. This work is an extension of already existing package of 
"poison detection" from "Adversarial Robustness Toolbox" developed by IBM.
In this work, first we build a AI-Model based on poisonous dataset, 
Spacenet dataset has been selected as an application.
Note that, poisonous data can be generated from different methods. This work 
is part of a whole project of poison attack and defense, and we focus on defense
side of the project.

After building poison model, then we detect poison datapoints 
and remove them. This procedure requires an analytic among different 
dimensionality reduction algorithms and unsupervised clustering algorithms
with appropriate evaluations and visualizations at each phase. 
By detecting and removing poisonous datapoints, we will have 
a cleaned dataset which is used to build a new and robust AI-Model.
The robust AI-Model will be able to detect and bypass future poison attacks.

#### Spacenet Dataset as use-case
In this use-case for PDA, we used [Spacenet Dataset](https://registry.opendata.aws/spacenet/). 
SpaceNet is a corpus of commercial satellite imagery and labeled training data to use for machine learning research. 
The dataset is currently hosted as an Amazon Web Services (AWS) Public Dataset.
 
###### Catologs
Spacenet comes with 6 cataloges for different regions. We used the first catalog for city of Rio de Janeiro.

|AOI	|Area of Raster (Sq. Km)   | Building Labels (Polygons)	| Road Labels (LineString) |
| ----- | ------------------------ | ------------------------------ | -----------------------  |
|AOI_1_Rio	|2,544	|382,534	|N/A
|AOI_2_Vegas	|216	|151,367	|3685 km
|AOI_3_Paris	|1,030	|23,816	|425 km
|AOI_4_Shanghai	|1,000	|92,015	|3537 km
|AOI_5_Khartoum	|765	|35,503	|1030 km
|AOI_6_Atlanta	|655 x 27	|126,747	|3000 km

#### Exploring Spacenet
###### Preprocessing
We used several preprocessing algorithms in order to crop the images contain of building and non-building area.
These cropped images give us the advantage of identifying the most relevant features for two class of `Class 0` : "No building"
and `Class 1` : "Has building". Creating two classes of cropped images helped us to have smaller images resoulutions
which can be used to train a Deep learning models easier.

###### Generating Poisonous images (Backdoors)
We then were able to generate portions of backdoors based on several state-of-the-art poison attack algorithms.
These generated backdoors are used with proportion of entire dataset. For example, 20% or 30% of dataset.

###### cropped images in entire dataset
The cropped images are in the size of 80 by 80 with 3-Channels (RGB).

Here are sample of the `class 0`:


![Alt text](pda/images/all_class0_demo.png?raw=true "Title")
And here are sample of `class 1`:

![Alt text](pda/images/all_class1_demo.png?raw=true "Title")
###### cropped images of poisonous images
As we can see, the generated poisoned images are based on the `class 0` with addition of features of `class 1`.
The added features will help the classifier will not to be trained correctly, therefore it will show poor performance of predicting test images.
![Alt text](pda/images/poison_images_demo.png?raw=true "Title")   

#### Deep Learning Models
We use a few well-known Convolutional Neural Network (CNN) Models for building classifiers.
CNN Model is a class of deep neural networks, most commonly applied to analyzing visual imagery.

A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. 
The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a 
multiplication or other dot product. The activation function is commonly a RELU layer, 
and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and 
normalization layers, referred to as hidden layers because their inputs and outputs are masked by the 
activation function and final convolution. The final convolution, in turn, often involves backpropagation 
in order to more accurately weight the end product.

Below there is a typical CNN Model architecture:
![Alt text](pda/images/CNN_Model.png?raw=true "Title") 

###### Layer activations
By giving each image as input of a pre-trained model, we can get activations for each convolutional layer. 
As a a sample of an input, we give the following image (left-hand side) and we get activation of it (right-hand side).

![Alt text](pda/images/sample_image.png?raw=true "Title")
![Alt text](pda/images/sample_activation.png?raw=true "Title")

The lighter colors have higher value, darker have lower.

The activation of each layer can be calculated. According to different filters, 
we will get different images as outputs of each layer. 
For example activations for layers 1, 4 and 6 will be:

###### _Layer 1 activations_:
![Alt text](pda/images/act_conv_1.png?raw=true "Title")

###### _Layer 4 activations_:
![Alt text](pda/images/act_conv_4.png?raw=true "Title")

###### _Layer 6 activations_:
![Alt text](pda/images/act_conv_6.png?raw=true "Title")

By collecting all the activations of the last layer into an array, we will get array of feature for each image.
By extracting the features for entire training dataset, we will be able to train the "dense layer" and "classifier"
in order to predict further images.
#### Install requirements
`pip install -r path/to/project/requirements.txt`