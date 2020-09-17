# Energy-Based Models

## EBM approach
predicting energy, pairs (x,y) that has the lowest score of F(x,y), for inference   

``` y' = argmin_y{F(x,y)} ```   

### Gradient-Based Inference
the energy function which is smooth and differentiable to perform the gradient-based method for inference   

``` using gradient descent: finding compatible y's ```   

> __Graphical Models__   
> + The energy function decomposes as a sum of energy terms   
> + Each energy terms takes a subset of variables   
> + Getting particular form to calculate efficiently

## EBM with Latent Variables
Latent Variables _provides auxiliary information for hard-to-interpret materials_   

### Inference
![graph](https://atcold.github.io/pytorch-Deep-Learning/images/week07/07-1/fig1.png)   

with latent variables,
+ minimizing energy function of y and z simultaneously is needed   
+ varying latent variable over a set   
  -> prediction output y varies as the manifold of possible predictions   
  -> multiple outputs   

``` y', z' = argmin_y,z{E(x,y,z)} ```   

> F_inf(x,y) = argmin_z{E(x,y,z)}   
> F_β(x,y) = -1/β * log{∫_z{exp(-β * E(x,y,z))}}   
> β -> inf, y' = argmin_y{F(x,y)}   

### Examples
1. Video Prediction
  - video compression
  - self-driving car

2. Translation
: multiple correct translation   
  1) produce all the possible translations   
  2) parametrise to respond to a given text (choose the best one) by varying some latent variables   
  
## Energy-Based Models v.s. Probabilstic Models
