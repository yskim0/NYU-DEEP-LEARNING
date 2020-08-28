# Applications of Convolutional Network

## Zip Code Recognition

> training several Convolutional Networks   
> for a series of non-overlapping digits is given as an input   

### Recognition with CNN
![zip_code](https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-1/O1IN3JD.png)   
1. 4 different sized convolutional networks == 4 different sized kernel,   
  __each producing one set of outputs with different window size__   
2. takes a majority vote and selects the category that corresponds to the highest score in that window   
  __selecting digits within 0~9__, _error correction leveraging input restrictions is needed to ensure the outputs are true zip codes_   
3. utilize shortest path algorithm by computing the minimum cost of producing digits and transitions between digit   

## Face Detection

_2 Problems exist..._   
1. False Positives: non-face objects recognized as face object   
2. Different Face Size: differing size may not be detected, __solved by using multi-scale versions of the same image__   

### A multi-scale face detection system
![multi](https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-1/CQ8T00O.png)   
The higher the scale is, the more clustered white regions(high scores) which represent detected faces   

### Non-maximum Suppression

only take the highest-scoring of the overlapping bounding boxes and remove the others,   
resulting in a single bounding box at the optimum location

### Negative Mining

create a negative dataset of non-face patches which the model has _errorneously_ detected as faces   
  1. collecting negative dataset: running the model on inputs known to contain no faces
  2. retraining the detector: using the negative dataset   
  _(repeating 1-2 process)_

## Semantic Segmentation
assigning a category to every pixel in an input image

### CNN for Long Range Adaptive Robot Vision

labeling regions from input images to distinguish between roads and obstacles   
  1. take a patch from the image and manually labeling it traversable or not
  2. train the convolutional network for several times on the patches: ask it to predict the color of the patch   
  3. applied to the entire image: labelling all the regions of the image as assigned colors   
  
![robot](https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-1/5mM7dTT.png)   
<br>

#### Stereo Labels
positions of every pixel in 3D space are estimated by measuring the relative distances between the pixels that apper in both the cameras in a stereo pair   
  - Limitations & Motivation for ConvNet: ConvNet detects objects at much greater distances than the stereo vision
  - Served as Model Inputs: pre-processing for building a scale-invariant pyramid of distance-normalized images == multiple scales   

#### Model Outputs
outputs a label for every pixel in the image up to the horizon, which work as the classifiers

```
continuous access to the stereo labels allows retraining __only the last layer of the network__   
== adapting to the new environment   
~previous layers are trained and fixed~
```

* detect objects up 50-100m   
* process around 1 frame per second, solved by __Low-Cost Visual Odometry Model__   

#### Scene Parsing and Labelling
![archi](https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-1/VpVbkl5.jpg)   
outputs an object category for every pixel with multi-scale architecture   
<br>

__Multiscale approach enables a vider vision by providing extra rescaled images as inputs__   
  1. reduce the same input image by the factor of 2 and 4 separately
  2. feed them to the same ConvNet (same weights, kernels) => outputs another two sets of Level1&2 features
  3. upsample these features to make Level 2 features the same size as the original image
  4. stack three sets of upsampled features together and feed them to a classifier
