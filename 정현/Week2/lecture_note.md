# Gradient Descent Optimization Agorithm

## Parametrised Models: y' = G(x,w)

* inputs: varies from sample to sample
* trainable parameters(implicit): shared across training samples
> implicit: aren't passed when the function is called, but saved inside the function <br>
> == Object-Oriented

### Supervised Learning

output goes into the cost function   
* C(y, y') // true output: y, model output: y'   
![Supervised Learning](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure1.jpg)

### Computation Graphs

* Variables (tensors, scalar / continuous, discrete)   
* Deterministic Functions   
![deterministic_function](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/deterministic_function.PNG)
  - multiple inputs -> multiple outputs
  - implicit parameter variable (w)
* Scalar-valued Function   
![scalar-valued](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/scalar-valued.PNG)
  - represent cost functions
  - implicit scalar output
  - take multiple inputs to a single output value (i.e distance between inputs)

### Loss Functions: Minimized during training

1. Per Sample Loss: L(x,y,w) = C(y,G(x,w))   
2. Average Loss:   
S = { (x[p],y[p]) | p = {0,...,P-1} } (P = minibatch)
L(S,w) = 1/P * sum(L(x,y,w))   
![Average_Loss](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Average_Loss.png)   
> Supervised Learning ) loss == output of the cost funciton

### Gradient Descent: finds the Minima of a function (continuous & differentiable)

1. Full batch Gradient Descent Update: w <- w - a * dL(S,w)/dw
2. SGD Update: pick a p = {0,...,P-1}, w <- w - a * dL(x[p],y[p],w)/dw
> positive semi-definite matrix : not the steepest gradient

### Reinforcement Learning: Gradient Estimation (without the explicit form for the gradient)

* RL cost function is not differentiable
* network computes the output is gradient-based
> Actor Critic Methods: consists of a second C module (known & trainable)
> * one is able to train C module to approximate the cost/reward function (negative cost == differentiable)

## Advantages of SGD and backpropagation for traditional neural nets

### Advantages of SGD
instead of computing the full gradient of the objective function (average of all samples), SGD takes one sample -> compute the loss (L) -> gradient of the loss -> take one step in the negative gradient direction
> w <- w - a*dL(x[p],y[p],w)/dw : gradient of the per-sample loss function for a given sample (x[p],y[p])   
![Figure2](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure2.png)   
: stochastically going down not directly

### Traditional Neural Network
interspersed layers of linear operations and point-wise non-linear operations   
1. take the input vector multiplied by a matrix formed by the weights
2. take all the components of the weighted sums vector and pass it through some simple non-linearity (i.e ReLU, tanh, ...)   
![2 layer NN](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure3.png)   

> Network Stacking
> ![Figure4](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure4.png)
> * s[i] = sum(w[i,j] * z[j])
> * z[i] = f(s[i])

### Backpropagation through a non-linear funciton
![Figure5](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure5.png)
> _Stochastic Operation_
> dC/ds = dC/dz * dz/ds = dC/dz * h'(s)
> dz = ds * h'(s)
> dC = dz * dC/dz = ds * h'(s) * dC/dz

### Backpropagation through a weighted sum
![Figure6](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure6.png)
> ds[0] = w[0] * dz
> ds[1] = w[1] * dz
> ds[2] = w[2] * dz
> dC = ds[0] * dC/ds[0] + ds[1] * dC/ds[1] + ds[2] * dC/ds[2]
> dC/dz = dC/ds[0] * w[0] + dC/ds[1] * w[1] + dC/ds[2] * w[2]

## PyTorch implementation of neural network and a generalized backprop algorithm

![Figure 7](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure%207.png)   
```
import torch
from torch import nn
image = torch.randn(3, 10, 20)
d0 = image.nelement()

class mynet(nn.Module):
    def __init__(self, d0, d1, d2, d3):
        super().__init__()
        self.m0 = nn.Linear(d0, d1)
        self.m1 = nn.Linear(d1, d2)
        self.m2 = nn.Linear(d2, d3)

    def forward(self,x):
        z0 = x.view(-1)  # flatten input tensor
        s1 = self.m0(z0)
        z1 = torch.relu(s1)
        s2 = self.m1(z1)
        z2 = torch.relu(s2)
        s3 = self.m2(z2)
        return s3
model = mynet(d0, 60, 40, 10)
out = model(image)
```
* nn.Linear(): Linear Layers
  - separate objects which contain a parameter vector
  - adds the bias vector implicitly
  
### Backprop through a functional module: generalized form of backpropagtaion

![Figure9](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure9.png)   
> zg : [dg * 1]
> df : [df * 1]
> dc/dzf = dc/dzg * dzg/dzf
> [1 * df] = [1 * dg] * [dg * df] : row vector

* Jacobian Matrix: computing the gradient of the cost function w.r.t zf given gradient of the cost function w.r.t zg
> (dzg/dzf)ij = (dzg)i/(dzf)j
> - each entry ij : partial derivative of the ith component of the output vector w.r.t to the jth component of the input vector

### Backprop through a multi-stage graph

![Figure10](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure10.png)
> dc/dzk = dc/dz(k+1) * dz(k+1)/dzk = dc/dz(k+1) * dfk(zk,wk)/dzk
> dc/dwk = dc/dz(k+1) * dz(k+1)/dwk = dc/dz(k+1) * dfk(zk,wk)/dwk

* 2 Jacobian Matrices
  1. z[k]
  2. w[k]

## A concrete example of backpropagation and intro to basic neural network modules

![02-2-1](https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-2/02-2-1.png)
> dC(y,y')/dw = 1*dC(y,y')/dy' * dG(x,w)/dw
> - dC(y,y')/dw : row vector (1 * N, N == num.of components of w)
> - 1*dC(y,y')/dy' : row vector (1 * M, M == dimension of the output)
> - dy'/dw = dG(x,w)/dw : matrix (M * N)

## Basic Neural Net Modules
1. Linear : Y = W * X
> dC/dX = W.t * dC/dY
> dC/dW = dC/dY * X.t
2. ReLU : Y = (x)+
> dC/dX = 0 (x<0)  or  dC/dY (otherwise)
3. Duplicate : Y1 = X, Y2 = X
  * Y-splitter == both outputs are equal to the input
  * backprop : gradients get summed
  * split into n branches similarly
> dC/dX = dC/dY1 + dC/dY2
4. Add : Y = X1 + X2
  * one is perturbed during summing two variables makes the output perturbed by the same quantity
> dC/dX1 = dC/dY * 1  and  dC/dX2 = dC/dY * 1
5. Max : Y = max(X1, X2)
> dY/dX1 = 1 (X1>X2)  or  0 (else)
> dC/dX1 = dC/dY * 1 (X1>X2)  or  0 (else)

## LogSoftMax vs. SoftMax
1. SoftMax
  > yi = exp(xi)/sum(exp(xj))
  > h(s) = 1 / (1 + exp(-s))
  - when s is large, h(s) is 1
  - when s is small, h(s) is 0
    + flat value : the gradient == 0 < gradient vanishing problem >
    
2. LogSoftMax
  > log(yi) = log(exp(xi)/sum(exp(xj)) = xi - log(sum(exp(xj)))
  > log(exp(s)/(exp(s)+1)) = s - log(1+exp(s))
  
## Practical tricks for backpropagation
1. ReLU as the non-linear activaiton funciton
2. Cross-Entropy Loss as the objective function for clssification problems
3. Stochastic Gradient Descent on minibatches during training
4. Shuffle the order of the training examples when using SGD
5. Normalize the inputs
6. Schedule to decrease the learning rate
7. L1 and L2 Regularization for weight decay
8. Weight Initializaiton
9. Dropout
