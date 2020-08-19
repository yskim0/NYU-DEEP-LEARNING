# Lecture_Intro
Multi-layer Neural Networks <br>
Backpropagation: training multi-layer neural nets, computing gradients by chain rule, method to know about how parameters of input goes to the output you want <br> <br>

## Supervised Learning
bunch of inputs + outputs(labels) <br>

### perceptron: 2 layer NN
1. 2nd: trainable, adaptive weights
2. 1st: fixed, random weights = associative layer (small input & output dim): single non-linear NN with random weights: random projections

### Mechanism
input
-> feature extractor: relevant characteristics of the input (useful for the input), vector of features
-> trainable classifier <br> <br>

### Features
- num. of units after the multiplication by a matrix == num. of row of the final matrix
- num. of col of matrix == the size of the input

* loss module: objective function comparing the output with target output (a distance, a discrepancy penalty, a divergence…) => scalar
* optimization: averaging the value of entire training set to get minimized wrong output: compute the gradient descent -> reach the convex

## train end-to-end unsupervised learning algorithm
### Basic Concepts from Brain
- complex cells: pool the activities of simple cells (orient-selective cells), integrates simple cells -> doesn’t change
- simple cells: orient changes -> distributed to another simple cell 
- Compositionality from natural data forms hierarchy: massive>part>subpart>motifs>contours/edges/textures>pixels

## Deep Learning
: control the entire task end-to-end not using hand-crafted engineering <br>
: cascading/sequence modules with trainable parameters (non-linear ex.ReLU) <br>
-> stack multiple layers <br>
-> train end-to-end

## Learning Representations
useful train data

### Pre-processing
: disentangle features into linear manifold directions == pre-processing
: input(n*n) <br>
-> multiple layers (non-linear function //clustering, quantization, sparse coding - pooling) <br>
-> linear: parallel <br> <br>

* SVM (super vector machines): 2 layers neural net (1st unsupervised learning – 2nd linear classifier)

# Practice_Space Stretching
## Basic Concept
- y = Wx Matrix, singular values [0~1, 0~1]: how much x, y dim expand or contract
- negative == reflection (RGB in clock-way), positive == default (RGB in clock-wise way)

## Functions
1. nn.Sequential: container which contains few Modules == sequence of Modules
2. nn.Linear: bias=True: affine transformation // bias=False: matrix multiplication
ex) nn.Linear(2,2,bias=False): linear transformation, map 0 to 0  fetch the matrix inside the screen and keep it at the center…
 nn.Sequential( <br>
nn.Linear(2,5), : shoot from 2D to 5D(height=5), bias: 5D == 5 rows(height) with 2 columns, bias with height size(5) <br>
nn.ReLU(), <br>
nn.Linear(5,2) : back to 2D to fit it to screen <br>
)
3. model.to(device): ship model to GPU(or CPU)
4. with torch.no_grad(): remove the variance

> when s gets bigger, the sparsity increases
