# Gradient Descent Optimization Agorithm

## Parametrised Models: y' = G(x,w)

* inputs: varies from sample to sample
* trainable parameters(implicit): shared across training samples
> implicit: aren't passed when the function is called, but saved inside the function <br>
> == Object-Oriented

### Supervised Learning

output goes into the cost function   
* C(y, y') // true output: y, model output: y'   
![Supervised Learning](./images/Figure1.jpg)

### Computation Graphs

* Variables (tensors, scalar / continuous, discrete)   
* Deterministic Functions   
![deterministic_function](./images/deterministic_function.png)
  - multiple inputs -> multiple outputs
  - implicit parameter variable (w)
* Scalar-valued Function   
![scalar-valued](./images/scalar-valued/png)
  - represent cost functions
  - implicit scalar output
  - take multiple inputs to a single output value (i.e distance between inputs)

### Loss Functions: Minimized during training

1. Per Sample Loss: L(x,y,w) = C(y,G(x,w))   
2. Average Loss:   
S = { (x[p],y[p]) | p = {0,...,P-1} } (P = minibatch)
L(S,w) = 1/P * sum(L(x,y,w))   
![Average_Loss](./images/Average_Loss.png)   
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
![Figure2](./images/Figure2.png)   
: stochastically going down not directly

### Traditional Neural Network
interspersed layers of linear operations and point-wise non-linear operations   
1. take the input vector multiplied by a matrix formed by the weights
2. take all the components of the weighted sums vector and pass it through some simple non-linearity (i.e ReLU, tanh, ...)   
![2 layer NN](./images/Figure3.png)   

> Network Stacking
> ![Figure4](./images/Figure4.png)
> * s[i] = sum(w[i,j] * z[j])
> * z[i] = f(s[i])

### Backpropagation through a non-linear funciton
![Figure5](./images/Figure5.png)
> _Stochastic Operation_
> dC/ds = dC/dz * dz/ds = dC/dz * h'(s)
> dz = ds * h'(s)
> dC = dz * dC/dz = ds * h'(s) * dC/dz

### Backpropagation through a weighted sum
![Figure6](./images/Figure6.png)
> ds[0] = w[0] * dz
> ds[1] = w[1] * dz
> ds[2] = w[2] * dz
> dC = ds[0] * dC/ds[0] + ds[1] * dC/ds[1] + ds[2] * dC/ds[2]
> dC/dz = dC/ds[0] * w[0] + dC/ds[1] * w[1] + dC/ds[2] * w[2]

## PyTorch implementation of neural network and a generalized backprop algorithm

