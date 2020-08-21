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

##
