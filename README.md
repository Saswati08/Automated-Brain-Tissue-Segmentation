# Automated-Brain-Tissue-Segmentation
# About
Automated brain tissue segmentation into white matter(WM), gray matter (GM), and cerebro-spinal fluid (CSF) from magnetic resonance images (MRI) is helpful
in the diagnosis of neuro-disorders such as epilepsy, Alzheimer’s, multiple sclerosis, etc. The task was to to design a Fully Convolutional Neural Network (FCN) 
which can specifically deal with MRI data and precisely segment the tissue into the aforementioned classes. The challenge to design a pipeline given a limited 
number of training sample (three volumes) with high resolution.
# Dependencies 
The .py and .pynb files are written in python version 3.6.9. You need to download **numpy(1.18.5)**, **torch(1.5.x)** and **matplotlib(3.2.2)** before running them.
# Execute Code
The dataset can be downloaded from the **[link](https://github.com/Saswati08/Automated-Brain-Tissue-Segmentation/blob/master/Brain_segmentation_dataset.zip)** After downloading the dataset you have 
to run the **preprocessing.py** file for preprocessing.To run go to terminal and write **python3 preprocessing.py**.
[**Note** - Make sure the dataset is in the working directory]. This will automatically store the processed .npy file in the working directory.For training you can use different architectures as given in segnet_*.py 
files. To have an idea how the whole pipeline works as in preprocessing, training and checking on validation you can refer pipe_to_train_and_validate_data.ipynb 
where functions like dice coefficient and confusion matrix has been implemented from scratch. 
# Preprocessing
Data set was given in the form of brain volumes in .mat format. After initial steps of preprocessing to obtain data in the form numpy arrays, dataset is iterated 
along the third dimension to obtain 2D images of size 256 X 128.There are 4 classes in this image segmentation problem. 0 class is assigned for background. 
As found from analysis pixel from class 0 are huge in number dominating other classes. Some images had only background class pixels. Moreover there was huge class
imbalance as very few class 1 pixels are present in the images. On top of this we have a very small dataset(3 volumes, 786 images) which makes the model prone to 
overfitting. To overcome the above problems, 2D patches have been made from each image where we have at least 1 pixel of ground truth 1 and no pixels of background
class 0. Using this technique we have 5798 patches. In this way we perform data augmentation and handle class imbalance to a great extent.



