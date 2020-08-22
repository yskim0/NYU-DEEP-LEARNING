# NYU Deep Learning - Week 3

## Visualization of neural networks

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/Network.png" width = "450">

When we step through the network one hidden layer at a time, we see that with each layer **we perform some affine transformation** followed by **applying the non-linear ReLU operation**, which eliminates any negative values. 

- Transformation of each layer = It's like a folding a plane




## Parameter transformations

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/PT.png" width = "200">


### Hypernetwork

- the weights of one network is the output of another network

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/HyperNetwork.png" width = "200">

### Weight Sharing

- function H(u) replicates one component of u into multiple components of w. (weight sharing)
- H -> like a "Y" branches

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/Motif.png" widht = "300">

- 기울기들이 안쪽에 축적되는 것을 막아야 함 -> `zero_grad` 필요!

## Detecting Motifs in Images

- swipe our **“templates”** over images to detect the shapes **independent of position and distortion of the shapes.**

- `template matching` : Shift Invariance. equivalence to shift
- **How can we design these “templates” automatically?**
- **Can we use neural networks to learn these “templates?**

=> `convolutions`

## Discrete convolution

<img src="https://user-images.githubusercontent.com/48315997/90951276-5aed1100-e494-11ea-9176-163d7d6d8237.png" width = 400>

- Convolution
- Corss-correlation : not reversed

## Regular twists that can be made with the convolutional operator in DCNNs

- Striding : can do larger step
- Padding : the output of convolution to be of **the same size as the input**
    - ReLU 비선형성을 사용할 때에는 제로 패딩 사용


## Deep Learning = Learning Hierarchical Representations


<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/cnn_features.png" width = 500>

- our world is **Compositional** -> we want to capture the **hierarchical representation** of the world

- local pixels assemble to form simple motifs 

## Convolutional Network Architecture

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-2/detailed_convNet.png" width = 450>

Conv. -> pool -> Conv. -> pool -> ...

## Overall Architecture

- Normalization
    - variation on whitening (optional)
    - Subtractive methods e.g. average removal, high pass filtering

- Filter Bank
    - Increase dimensionality
    - Edge detections

- Non-Linearity
    - Sparsification
    - ReLU

- Pooling
    - Aggregating over a feature map
    - Max Pooling
    - LP-Norm Pooling, Log-Prob Pooling

## LeNet5 and digit recognition

```py
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 20, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1)
        x = self.fc2(x)
    return F.logsoftmax(x, dim=1)
```

Although fc1 and fc2 are fully connected layers, they can be thought of as convolutional layers whose kernels cover the entire input. Fully connected layers are used for efficiency purposes.

## Advantages of CNN

- no need to specify the size of the input
    - changing the size of the input changes the size of the output.

## Feature binding problem

An object is a collection of features, but **how to bind all of the features to form this object?**

-> We can solve it with CNN.

- only two layers of convolutions with poolings plus another two fully connected layers

## Example: dynamic input length

we repeat the convolution and pooling again and **eventually we get 1 output.**

## What are CNN good for

- `Locality`
    - there is **a strong local correlation between values**. 
    - If we take two nearby pixels of a natural image, those pixels are very likely to have the same colour. *As two pixels become further apart, the similarity between them will decrease.* 
    - The local correlations can help us detect local features, **which is what the CNNs are doing.** 
    - *If we feed the CNN with permuted pixels, it will not perform well at recognizing the input images, while FC will not be affected. The local correlation justifies local connections.*

- `Stationarity`
    - the features are essential and can appear anywhere on the image, justifying the shared weights and pooling. 
    - Moreover, statistical signals are uniformly distributed, which means **we need to repeat the feature detection for every location on the input image.**

- `Compositionality`
    - the natural images are **compositional**, meaning the features compose an image in a hierarhical manner. 
    - This justifies **the use of multiple layers of neurons**, which also corresponds closely with Hubel and Weisel’s research on simple and complex cells.



## Properties of natural signals

- Stationarity
    - 특정 주제가 신호에 반복적으로 나타남을 뜻함.
    - 주어진 시간범위 안에 같은 타입의 패턴이 계속 관찰되는 현상

- Locality
    - 가까운 포인트들끼리는 거리가 비교적 더 먼 곳에 있는 포인트보다 더 연관성을 가짐
    - 한 신호와 다른 반전된 신호가 완벽히 겹쳐지게 된다면 그 두 신호의 컨벌루션(합성곱)은 최곳값을 갖는다
        -  두 개의 1차원 신호들의 합성곱(상호상관cross-correlation)은 두 신호의 내적dot product이며, 두 벡터가 얼마나 비슷한지 혹은 가까운지를 측정

- Compositionality
    - 모든 것은 다른 하위 개체들로 이루어져 있다


**If our data exhibits stationarity, locality, and compositionality, we can exploit them with networks that use sparsity, weight sharing and stacking of layers.**


## Locality => sparsity

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-3/Figure%202(b)%20After%20Applying%20Sparsity.png" width = 150>

집약성 특성의 장점을 이용해 거리가 먼 뉴런들과의 연결들을 끊을수 있고, 히든 레이어 뉴런들은 모든 입력값을 생성span하지는 않지만, 전반적인 구조는 모든 입력 뉴런들을 설명할 수 있다. 


## Stationarity => parameters sharing

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-3/Figure%203(b)%20After%20Applying%20Parameter%20Sharing.png" width = 150>

Using sparsity and parameter sharing => 연결망 구조에 걸쳐 작은 매개변수집합을 여러번 사용할 수 있음.

- Parameter sharing
    - faster convergence
    - better generalization
    - not constained to input size
    - kernel indepence => high parallelisation

- Connection sparsity
    - reduced amount of computation


- Kernel size of even number might lower the quality of the data, thus we always have **kernel size of odd numbers**, usually 3 or 5.

## Padding

Padding generally *hurts the final results*, but it is convenient programmatically. We usually use zero-padding: `size = (kernel size - 1)/2.`

## Standard spatial CNN

- Multiple layers
    - Convolution
    - Non-linearity (ReLU and Leaky)
    - Pooling
    - Batch normalisation : very helpful to get the network to train well
- Residual bypass connection : very helpful to get the network to train well

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-3/Figure%205%20Information%20Representations%20Moving%20up%20the%20Hierachy.png" width = 300>

-  there is a trade off between the spatial information and the characteristic information and the representation becomes denser.

## Pooling

- Max Pooling, Average Pooling ...
- Main purpose : reduces the amount of data so that we can compute in a reasonable amount of time

## Practice - random permutation

The performance of the FC network almost stayed unchanged (**85%**), but the accuracy of CNN dropped to **83%**. 

This is because, **after a random permutation, the images no longer hold the three properties of locality, stationarity, and compositionality**, that are exploitable by a CNN.
