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
SGD makes bounces around the floow when the step reaches valley,   
and decreased learning rate makes these bounces slower.   
WITH Momentum, by smoothing out the steps, __no bouncing__ occurs.
```
