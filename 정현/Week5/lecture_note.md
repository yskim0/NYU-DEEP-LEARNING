# Optimisation Techniques

## Gradient Descent

![G_d](https://blog.paperspace.com/content/images/2018/05/sgd.png)   
__find the lowest point == valley__   
+ function L: continuous, differentiable_   
+ γ: __step size == learning rate__, a little larger than the optimal to converge   
![lr](https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-1/step-size.png)   


## Stochastic Gradient Descent

__stochastic estimation: the gradient of the loss for a single data point__   

> GD: update the weights according to the gradient over the total loss f   
> SGD: update the weights according to the gradient over f_i   
>   + _choosing i randomly => noisy but unbiased estimator of f_   

1. prevent from redundant computations of redundant information => cheaper to cimpute   
2. small noise of SGD step that makes similar effect to GD step    
3. __Annealing__: noise prevents convergence to a shallow local minima   

### Mini-batching   

_instead of calculating just one instance, consider the loss over multiple randomly selected instances_   

* Distributed network training tech: split a large mini-batch between the machines of a cluster    
  -> aggregate the resulting gradients   
  
* Not adjustable to full sized batch   

## Momentum

![momentum](https://miro.medium.com/proxy/1*LjVeDQEHZBKC6C0TiUngWg.png)   
__Momentum__
  + _p_: SGD momentum, _running average of the gradients_   
  + move _w_ in the direction of the new momentum _p_   

![alter](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRNfZVTicDslsYwzwGJf4wEnArLIEcZvg5HZw&usqp=CAU)   
__Stochastic Heavy Ball Method__
   + next step: a combination of previous step's direction _(w_k-w_k-1)_ and the new negative gradient   

### Intuition

* Momentum: keeps the ball moving in the same direction of current movement, damplening the oscillations which occur when only using SGD   
  - _β_ : (0,1) value   
  ![beta](https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-1/momentum-beta.png)   
    + 0+ : change in direction quicker   
    + 1- : takes longer to make turns   
* Gradient: pushes the ball in some other direction to current one (opposite to momentum)   
![momentum_overall](https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-1/momentum.png)   

### Practical guidelines

```
Momentum usually be used with Stochastic Gradient Descent
```

* _β_ = 0.9 or 0.99 usually   
* __learning rate must be decreased when momentum parameter is increased to maintain convergence, *vice versa*__   

### Why does Momentum works?

1. Acceleration   
_(not effetively increase practice)_
accelerates momentum only in quadratics   

__2. Noise Smoothing__    
SGD requires (extra) averaging a whole bunch of updates and then take a step in that direction   
BUT, extra averaging process is not needed, _because Momentum adds smoothing to the optimization process, making each update a good approximation to the solution_   
![result](https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-1/sgd-vs-momentum.png)   

* * * 

```
SGD makes bounces around the floor when the step reaches valley,   
and decreased learning rate makes these bounces slower.   
WITH Momentum, by smoothing out the steps, __no bouncing__ occurs.
```

## Adaptive Methods

_SGD formulation_: every single weight in network is updated using an equation with the same learning rate(γ)   
__==> adapt a learning rate for each weight individually by using the information gotten from gradients for each weight__   
<br>
![overall](https://image.slidesharecdn.com/random-170910154045/95/-49-638.jpg?cb=1505089848)   
<br>

### RMSprop (Root Mean Square Propagation)
normalizing the gradient by its root-mean-square   
![rms](https://blog.paperspace.com/content/images/2018/06/momprop2-2.png)   

### ADAM (Adaptive Moment Estimation) == RMSprop + momentum
the momentum update is converted to an exponential moving average and the learning rate doesn't need to be changed when dealing with β   
![adam](https://blog.paperspace.com/content/images/2018/06/adam.png)   

### Practical Side
_similar to SGD without momentum, RMSprop also suffers from noise which bounces at the floor when it's close to a local minimizer_   
__Thus, ADAM is generally recommended over RMSprop because it combines RMSprop with momentum, which results in improvement similar to SGD with momentum, which is not-noisy and good, stable estimate of the solution__   
![result](https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-2/5_2_comparison.png)   

_However_, ADAM has several disadvantages, listed below...   
1. very simple test problems, which the method doesn't converge   
2. generalization errors, giving non-zero loss to unseen data points   
3. need to maintain 3 buffers   
4. 2 momentum parameters with non-one value   

## Normalization Layers
improve the network- _the optimization and generalization performance_ -itself by additional layers in between existing layers, usually between linear layers and activation functions but also after the activation functions, without changing the power of the network

### Normalization Operations
![norm](https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-2/5_2_norm_operations.png)   
__< Computer Vision >__   
* Batch Norm: applied only over one channel of the entire input images   
* Group Norm: applied over one image but across a number of channels   
__< Language >__   
* Layer Norm: applied within one image across all channels   
* Instance Norm: applied only over one channel and image   

### Why does Normalization help?
1. networks become easier to optimize, allowing for the use of larger learning rates and speeding up the training of NN   
2. regularization effect makes extra noise resulting in better geneeralization   
3. reduces sensitivity to weight initialization   

### Practical Considerations

```
torch.nn.BatchNorm2d(num_features, ...)
torch.nn.GroupNorm(num_groups, num_channels, ...)
```

> BatchNorm / InstanceNorm: multiple training samples => mean/std used are fixed after training   
> GroupNorm / LayerNorm: one training sample => no fixation is needed   

## The Death of Optimization

