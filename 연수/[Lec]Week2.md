# NYU Deep Learning - Week 2

## Parameterized Model

<img width="300" alt="스크린샷 2020-08-20 오후 1 06 10" src="https://user-images.githubusercontent.com/48315997/90715666-ed9b7d80-e2e5-11ea-8828-e22d25264f6e.png">

### Block diagram notations

- Variables(x, y bar) : tensor, scalar, continuous, discrete...

- Deterministic function(G(x,w))
    - implicit parameter variable (here : w)

- Scalar-valued function(C(y,y bar)) : implicit output
    - Here : Cost function
    - single scalar output(implicit)

## Loss function, average loss

- Simple per-sample loss funciton
    - `L(x,y,w) = C(y,G(x,w))`
- A set of samples
    - `S = {(x[p],y[p])/ p = 0...P-1}`
- Average loss over the set

    <img width="300" alt="스크린샷 2020-08-20 오후 1 09 59" src="https://user-images.githubusercontent.com/48315997/90715874-75818780-e2e6-11ea-828a-48c47aa00c84.png">

- Block Diagram

<img width="150" height="300" alt="스크린샷 2020-08-20 오후 1 10 37" src="https://user-images.githubusercontent.com/48315997/90715912-8c27de80-e2e6-11ea-81be-cbdee6783cd2.png">

## Gradient Descent

- Finds **the minima of a function**, assuming that one can easily compute the gradient of that function. 

- **It assumes that the function is continuous and differentiable almost everywhere (it need not be differentiable everywhere).**
    - RL은 대부분 not differentiable -> environment 사용함

- RL -> Gradient **Estimation** without the explicit form for the gradient
    - The RL cost function is **not differentiable** most of the time but the network that **computes the output is gradient-based.**
        - main difference between supervised learning and reinforcement learning

    > A very popular technique in RL is **Actor Critic Methods**. A critic method basically consists of a second C module which is a known, trainable module. One is able to train the C module, which is differentiable, to approximate the cost function/reward function. The reward is a negative cost, more like a punishment. That’s a way of making the cost function differentiable, or at least approximating it by a differentiable function so that one can backpropagate.


- *gradient is always orthogonal to the lines of equal cost* (완벽히 이해되진 않음)

- Full (batch) gradient

    <img width="150" alt="스크린샷 2020-08-20 오후 1 13 28" src="https://user-images.githubusercontent.com/48315997/90716056-f2acfc80-e2e6-11ea-8941-3fc239f01385.png">

- Stochastic Gradient (SGD)

    <img width="200" alt="스크린샷 2020-08-20 오후 1 14 21" src="https://user-images.githubusercontent.com/48315997/90716103-11ab8e80-e2e7-11ea-8074-e744b3616976.png">

- single sample만 사용할 경우 -> very noisy trajectory & faster than batch
    - Every sample will pull the loss towards a different direction.

- SGD exploits(uses) the **redundancy** in the samples
    - For generalization, it has to be some redundancy
    - In practice, we use mini-batches for **parallelization**


## Traditional Neural Net

<img width="500" alt="스크린샷 2020-08-20 오후 1 20 37" src="https://user-images.githubusercontent.com/48315997/90716447-f2f9c780-e2e7-11ea-9ea3-95bd0c6b386f.png">

## Backpropagation through a non-linear function

<img width = "500" src="https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure5.png">

- **Chain Rule**

    - `g(h(s))' = g'(h(s))*h'(s)`
    - dc/ds = dc/dz * dz/ds
    - dc/ds = dc/dz*h'(s)
    
    <br>

    - dz = ds*h's
        - dc = dz * dc/dz = ds * h'(s) * dc/dz
        - Hence, dc/ds = dc/dz*h'(s)

## Backprop through a weighted sum

<img width="500" src="https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure6.png">

![스크린샷 2020-08-20 오후 1 47 09](https://user-images.githubusercontent.com/48315997/90717903-a6b08680-e2eb-11ea-986c-b2897910541f.png)

## Block diagram of a traditional neural net

<img width = "300" src = "https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure%207.png">

## Pytorch Implementation

```py
import torch
from torch import nn
image = torch.randn(3,10,20)
d0 = image.nelement()

class mynet(nn.Module):
    def __init__ (self,d0,d1,d2,d3):
        super().__init__() # nn.Module의 init 메서드 호출
        self.m0 = nn.Linear(d0,d1)
        self.m1 = nn.Linear(d1,d2)
        self.m2 = nn.Linear(d2,d3)
    def forward(self, x):
        z0 = x.view(-1) # flatten
        s1 = self.m0(x)
        z1 = torch.relu(s1)
        s2 = self.m1(z1)
        z2 = torch.relu(s2)
        s3 = self.m2(z2)

        return s3

model = mynet(d0,60,40,10)
out = model(image)
```
- The `nn.Linear` class also adds the bias vector implicitly.

- In the forward function, you define how your model is going to be run, from input to output.

- > We do not need to compute the gradient ourselves since PyTorch knows how to back propagate and calculate the gradients given the forward function.

## Backprop through a funcitonal module

<img width = "200" src="https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-1/Figure9.png">


## Backprop = propation through a transformed graph

<img width = "300" src="https://atcold.github.io/pytorch-Deep-Learning/images/week02/02-2/02-2-1.png">

<br>

<img width="240" alt="스크린샷 2020-08-20 오후 2 32 56" src="https://user-images.githubusercontent.com/48315997/90720696-0c077600-e2f2-11ea-8e18-f79a57d64067.png">

## Basic Modules

- Linear : `Y = W * X`
    - dC/dX = W^T * dC/dY
    - dC/dW = dC/dY * X^T

- ReLU
    - dC/dX = 0(x<0), otherwise dC/dY

- Duplicate : Y1 = X, Y2 = X
    - dC/dX = dC/dY1 + dC/dY2

- Add : Y = X1 + X2
    - dC/dX1 = dC/dY
    - dC/dX2 = dC/dY

- Max
    - dY/dx1 = 1(x1>x2), otherwise 0
    - dC/dx1 = dC/dY(x1>x2), otherwise 0

- LogSoftMax


### LogSoftMax vs. SoftMax

- softmax : convenient way of transforming a group of numbers into a group of positive numbers between 0 and 1 that sum to one

![스크린샷 2020-08-20 오후 2 44 07](https://user-images.githubusercontent.com/48315997/90721367-9c928600-e2f3-11ea-84b2-945f09a6de27.png)

- **use of softmax leaves the network susceptible to vanishing gradients.**

    - when s is large, h(s) is 1, and when s is small, h(s) is 0.
     Because the sigmoid function is flat at h(s) = 0 and h(s) = 1, the gradient is 0, which results in a vanishing gradient.

- So `LogSoftmax` came up!

![스크린샷 2020-08-20 오후 2 48 36](https://user-images.githubusercontent.com/48315997/90721689-3ce8aa80-e2f4-11ea-9c94-95985f80e734.png)

- When s is very small, the value is 0, and when s is very large, the value is s.
    - vanishing gradient problem is avoided

## Backprop in Practice

- Use ReLU non-linearities
- Use Corss-Entropy loss for classification
- Use SGD on minibatches
- Shuffle the training samples
- Normalize the input variables (zero mean, unit variance)
- Schedule to decrease the lr
- Use L1 and/or L2 regularization for weight decay
    - weight decay : weight들의 값이 증가하는 것을 제한함으로써, 모델의 복잡도를 감소시킴으로써 제한하는 기법, 즉 weight를 decay(부식시킨다라는 의미를 조금 감량시키는 의미로 생각하면 될 것 같습니다.) 시켜서 Overfitting을 방지하는 기법으로 소개됩니다 || [출처](https://deepapple.tistory.com/6)
    - L2 reg. 

    ![스크린샷 2020-08-20 오후 2 59 04](https://user-images.githubusercontent.com/48315997/90722388-b339dc80-e2f5-11ea-932a-a47d42171013.png)

    -> ![스크린샷 2020-08-20 오후 2 59 52](https://user-images.githubusercontent.com/48315997/90722460-cea4e780-e2f5-11ea-965a-c7ef62e71422.png)

    - weight decay : wi의 계수가 됨 (less than 1)

- weight initialization
    - if the inputs to a unit are independent, the variance of weighted sum will be equal to the sum of the variances of the input * weighted by the squared of the weights

- Dropout
    - Dropout is another form of regularization.
    - It can be thought of as another layer of the neural net: it takes inputs, randomly sets n/2 of the inputs to zero, and returns the result as output.


