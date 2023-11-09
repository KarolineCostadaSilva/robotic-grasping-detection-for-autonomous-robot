# Robotic-grasping-cornell

In this project, Deep Convolutional Neural Networks (DCNNs) is used to simultanously detect a grasping point and angle of an object, so that a robot arm can pick the object. In general, this is the implementation of a small model of the model presented in this paper https://arxiv.org/abs/1802.00520. We do not implement the Grasp Proposal Networks, so instead of using 2-stage DCNNs, we use one-stage DCNNs.

## Platform

- python 3.6.8
- pytorch 1.0.0

## Codes

1. Data preprocessing
2. Training
3. Demo

### 1. Data preprocessing

- Download [Cornell Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)
- Run `dataPreprocessingTest_fasterrcnn_split.m` (please modify paths according to your structure)

### 2. Training

```
$ python train.py --epochs 100 --lr 0.0001 --batch-size 8
```

### 3. Demo

- Download the pretrained model [GoogleDrive](https://drive.google.com/drive/folders/1Th4hMEdSr3kTQBXJTIDstSJn2eUWvoaa?usp=sharing)
- Put in the folder `./models`
- Run demo:

```
$ python demo.py
```

## Acknowledgment

This repo borrows some of code from
https://github.com/ivalab/grasp_multiObject_multiGrasp

## Tools

<p align="left"> <a href="https://matplotlib.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/2048px-Created_with_Matplotlib-logo.svg.png" alt="matplotlib" width="40" height="40"/> </a> <a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://user-images.githubusercontent.com/50221806/86498201-a8bd8680-bd39-11ea-9d08-66b610a8dc01.png" alt="numpy" width="40" height="40"/> </a> <a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://github.com/opencv/opencv/wiki/logo/OpenCV_logo_no_text.png" alt="opencv" width="40" height="40"/> </a> <a href="https://www.python.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" alt="python" width="40" height="40"/> </a> <a href="https://scikit-image.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Scikit-image_logo.png" alt="scikit-image" width="40" height="40"/> </a> <a href="https://scipy.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/SCIPY_2.svg/1200px-SCIPY_2.svg.png" alt="scipy" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/640px-PyTorch_logo_icon.svg.png" alt="scipy" width="35" height="40"/> </a> </p>