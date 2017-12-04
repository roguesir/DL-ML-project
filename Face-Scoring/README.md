# Face Scoring - Score Face by CNN Model based on TensorFlow


## Dataset
* Doownload: http://www.hcii-lab.net/data/SCUT-FBP/EN/download.html

![Result Pic](https://github.com/roguesir/DL-project/blob/master/Face-Scoring/web_image.jpg)

After downloading, we need match the labels and images.

![Result Pic](https://github.com/roguesir/DL-project/blob/master/Face-Scoring/face_image.jpg)

## Run

There are three train scripts about the train model. 

* python3 train.py
* python3 train_alex.py
* python3 train_vgg.py

## Test

![Result Pic](https://github.com/roguesir/DL-project/blob/master/Face-Scoring/test_image.jpg)

## Model Downloading

I train three models with different architectures: one is with three convolutional layers and two fully connected layers; another is with five convolutional layers and three fully connected layers like AlexNet; the last one is with 13 convolutional layers and three fully connected layers like VGG-16.

### Download link:

First model: http://pan.baidu.com/s/1bp8gqfH

AlexNet model: http://pan.baidu.com/s/1mhM0Jkk