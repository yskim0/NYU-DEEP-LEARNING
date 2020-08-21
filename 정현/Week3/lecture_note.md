# Parameter Transformation

## Visualization of neural networks

### Why is it hard with 2 neuron in each hidden layer? Why does it have to be more neurons?
Each hidden layer has one bias, therefore if one bias moves out of a top-right quadrant, it became 0.   
No matter how later layer it is, the value remains 0.   
Thus, by adding more neurons or more hidden layers, the meaningful values could remain until the end of a NN.

## Parameter Transformations
![Figure.5](https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/PT.png)   
* u <- u - a * (dH/du).t * (dC/dw).t
* w <- w - b * (dH/du) * (dH/du).t * (dC/dw).t
> dimension
> - u: [Nu * 1]
> - w: [Nw * 1]
> - (dH/du).t: [Nu * Nw]
> - (dC/dw).t: [Nw * 1]

### Weight Sharing
H(u) replicates one component of u into multiple components of w.
> w1 = w2 = u1   
<br>
equal shared parameters -> the gradient will be summed in the backprop.

> the gradient of C(y,y') w.r.t. u1 => the gradient of C(y,y') w.r.t. w1 + C(y,y') w.r.t. w2

### Hypernetwork
![Figure.6](https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/HyperNetwork.png)   
the network H(x,u) -> the weights of G(x,w)

### Motif Detection in Sequential Data
![Figure.7](https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/Motif.png)   
sliding window on data moving the weight-sharing function to detect a particular motif   
-> output goes into a maximum function (sum up five gradients)   
-> backpropagate the error to update the parameter w
> zero_grad()

### Motif Detection in Images
hand-crafted method using local detectors and summation   
![Figure.8](https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/MotifImage.png)   
* C has 2 end-points
  - endpoint templates
* D has 2 corners
  - corner templates
=> find out the shape similar to the templates: thresholded output -> distinguish letters by summing them up   

![Figure.9](https://atcold.github.io/pytorch-Deep-Learning/images/week03/03-1/ShiftInvariance.png)   
template matching: shift-invariant   
   
   
**design "templates" automatically, use NN to learn these "templates" _by Convolution_**   


## Discrete Convolution

### Convolution

#### yi = sum_j(wj * x(i-j))
i-th output(yi) == reversed w * window of the same size in x (shifting this window by one entry each time)   
reads the weight stored in memory backward

### Cross-Correlation

#### yi = sum_j(wj * x(i+j))
_interchangeable with convolution_   
reads the weight stored in memory forward

### Higher Dimensional Convolution

#### y(ij) = sum_kl(w(kl) * x(i+k,j+l))
> w: convolution kernel

### Regular Twists that can be made with the Convolutional Operator in DCNNs
1. Striding: shifting the window in x with a larger step at a time   
2. Padding: make the output of convolution to be of the same size as the input by padding the input ends with a number of zero entries (counted in ReLU usage)

## Deep Convolution Neural Networks (DCNNs)

> Deep Neural Networks: repeated alternation between (linear operators + point-wise nonlinearity layers)
> * linear operator == convolution operator

Stacking multiple layers -> _hierarchical representation of the data_   
: **Compositional world representation**

* * *

# CNN

## Overall Architecture Breakdown
1. Normalization
2. Filter Banks
3. Non-linearities
4. Pooling

> Fully Connected Layer: Convolutional layers whose kernels cover the entire input. (kernel size == input size)
> - efficiency purpose
> - no need to specify the size of the input, but the changes in input size changes the output size

## Advantages of CNN
do not have to break the input image into segments(== recognizing an image), but just have to apply the CNN over the entire image   
: kernels will cover all locations in the entire image and record the same output regardless of where the pattern is located

## Feature Binding Problem
How can we recognize / classify the object as THE object? How to bind all of the features that represent / form the object?

**_How to Solve it?_**   
2 convolution layers + poolings + 2 FC layers _with enough non-linearity (special features) and data to train CNN_

## What are CNN good for
natural signals in the form of multidimensional arrays with...
1. Locality: local correlations -> detect local features (CNN's intention)
2. Stationarity: essential and common features -> shared weights and pooling, uniformly distributed
3. Compositionality: composing an image in a hierarchical manner (multiple layers of neurons)
