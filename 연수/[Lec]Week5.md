# NYU Deep Learning - Week 5

## Gradient descent

> “The worst optimization method in the world”

- Problem : `min f(w)`

- Solution

<img width="500" alt="스크린샷 2020-08-28 오전 3 06 39" src="https://user-images.githubusercontent.com/48315997/91478831-7fede380-e8db-11ea-92c8-7a26c922ec61.png">

- f : 연속적, 미분 가능
- 최적화 함수의 최소점을 찾고 싶지만, 실제 방향을 알 수 없고 국소적인 방향만 볼 수 있으므로 **음의 경사 방향**만이 최고의 정보이다.
    - 경사 하강법은 기본적으로 steepests descent(음의 경사) 방향을 따라 이루어짐.

- gamma : step size
    - 이상적인 이동 크기는 최적의 이동 수치보다 약간 크게 잡는 것
    - 실제 적용시에는 학습률을 발산하지는 않을 정도의 크기로 사용

## Stochastic optimization

확률적 경사 하강법에서는 **경사 벡터의 확률적 추정**으로 실제 경사 벡터를 대체함.
- `확률적 추정`이란 한 데이터 지점에서 손실의 경사

<img width="337" alt="스크린샷 2020-08-28 오전 3 15 04" src="https://user-images.githubusercontent.com/48315997/91479616-ad875c80-e8dc-11ea-9439-cffc2766147d.png">

- f(i) = l(xi,yi,w) :: i번째 인스턴스에서의 Loss
- SGD에서는 f(i)의 경사에 따라 w를 갱신함.
    - **전체 손실 f의 경사가 아님.**

- *만일 i가 무작위로 선택된다면 f(i)에 노이즈가 있어도 f의 불편 추정량(unbiased estimator)이 될 것임.*

<img width="400" alt="스크린샷 2020-08-28 오전 3 18 19" src="https://user-images.githubusercontent.com/48315997/91479897-21296980-e8dd-11ea-8a31-c1f4927cc9c5.png">

- 어떤 SGD 갱신이라도 전체 배치 갱신의 기대값과 같음.
- SGD의 노이즈는 얕은 local minimum을 피하고, 더 좋은 최저점을 찾도록 도와줌 -> `annealing`

- SGD의 이점
1. 불필요한 연산 많아지는 것 방지
2. 학습 초반, 경사 내부의 정보에 비해 노이즈는 작다.
3. Annealing - SGD를 갱신할 때 노이즈는 얕은 국소 최저점으로의 수렴을 방지한다.
4. 계산 비용이 저렴하다(모든 데이터 지점을 갈 필요가 없으므로)

## Mini-batching

다수의 random instance들의 loss를 계산함.

- 경사 하강법은 절대로 전체 크기의 배치(Full-batch)로 사용돼서는 안된다. 
- 전체 배치 크기를 트레이닝 시키고 싶을 경우에는 `LBFGS` 최적화 기법을 사용해야 한다.

## Momentum

<img width="200" alt="스크린샷 2020-08-28 오전 3 23 30" src="https://user-images.githubusercontent.com/48315997/91480380-da883f00-e8dd-11ea-8fc9-de76a3d77c0b.png">

모멘텀에서는 p, w를 갱신시키는 두 번의 반복 작업이 들어간다.

- p : `SGD momentum`
    - 0~1사이의 beta를 이용해 이전 모멘텀값을 감쇠시킨 다음 확률적 경사를 더함
    - p는 경사들의 running averages
    - p의 방향으로 w를 움직임.

### Intuition

<img width="600" alt="스크린샷 2020-08-28 오전 3 26 38" src="https://user-images.githubusercontent.com/48315997/91480668-49fe2e80-e8de-11ea-83fb-c5ea70f5e61f.png">

**모멘텀은 SGD를 사용할 때 자주 발생하는 진동 현상을 감쇠한다.**

- beta : Dampening factor. 0~1

### Practical Aspects of momentum

SGD + momentum 

beta = 0.9 or 0.99

### Acceleration

<img width="450" alt="스크린샷 2020-08-28 오전 3 30 30" src="https://user-images.githubusercontent.com/48315997/91481066-d4df2900-e8de-11ea-8691-dd4f14ca6b67.png">

- Nesterov 모멘텀에서 상수값을 선택한다면 빠르게 수렴시킬 수 있지만, **convex problem에만 적용가능하고 신경망에는 적용시킬 수 없다.**

- 표준 모멘텀은 실제로 이차방정식에서만 가속된다.
- SGD에는 노이즈가 있고, 가속은 노이즈와 함께해서는 잘 작동하지 않기 때문에 SGD에서는 가속이 잘 되지 않는다.

### Noise Smoothing

실질적으로 모멘텀이 동작하는 이유임.

- 모멘텀은 경사값들을 평균낸다. running average

<img width="80" alt="스크린샷 2020-08-28 오전 3 33 56" src="https://user-images.githubusercontent.com/48315997/91481397-4fa84400-e8df-11ea-9d99-ad0b6981e7d3.png">

- When using SGD, this is suboptimal! We should actually take an average over past time steps

- 모멘텀 + SGD => **requires no averaging, the last value may directly be returned!**

## Adaptive methods

SGD가 동작하지 않을때 사용된 방법들 -> Adaptive Methods

- SGD에서 네트워크 상 모든 개별 가중치는 동일한 learning rate를 사용한 공식을 통해 갱신됨.
- Adaptive Methods -> **개별적인 가중치를 위한 학습률을 적용**
- 계층별로 적응적 학습률

## RMSprop

<img width="700" alt="스크린샷 2020-08-28 오전 3 38 24" src="https://user-images.githubusercontent.com/48315997/91481783-eecd3b80-e8df-11ea-8449-c594d6a6cc19.png">

- Key IDEA: normalize by the **root-mean-square of the gradient**
- 노이즈의 양을 추정하기 위해 exponential moving average를 통해 v를 갱신함.

### Adam

= Adaptive Moment Estimation.

= RMSprop + Momentum

![스크린샷 2020-08-28 오전 3 40 51](https://user-images.githubusercontent.com/48315997/91481990-466ba700-e8e0-11ea-811b-e84ecc205401.png)

## Normalization layers

네트워크의 Norm. layer가 신경망 구조 자체를 향상시킨다.

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-2/5_2_norm_layer_b.png" width = "150">

- 식

<img width="500" alt="스크린샷 2020-08-28 오전 3 51 52" src="https://user-images.githubusercontent.com/48315997/91482931-d0683f80-e8e1-11ea-89cb-84445a62cb8d.png">

- x : 입력 벡터
y : 출력 벡터
mu : x 표준편차의 추정
a : learnable scaling factor
b : learnable bias term

- 학습가능한 매개변수 a,b 가 없으면, 출력 벡터의 분포 y는 평균 0, 표준 편차 1로 고정된다.
- a,b는 네트워크의 표현력을 유지해준다.
    - 출력값은 어떤 특정한 범위라도 넘어설 수 있다.
    - a,b는 정규화를 되돌리지 않는데 이 둘은 학습가능한 매개변수이고 mu, sigma보다 더 안정적이기 때문이다.



### 입력 벡터 정규화

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-2/5_2_norm_operations.png" width = 400>

- Batch Norm : 입력값에서 하나의 채널에만 적용
- Layer Norm : 한 이미지의 모든 채널들에 걸쳐 적용
- Instance Norm : 한 이미지에서 한 채널에만 적용
- Group Norm : 하나의 이미지, 다수의 채널에 걸쳐 적용

실제로 배치 정규화와 그룹 정규화는 컴퓨터 비전 문제들에서 잘 동작하고, 계층 정규화와 인스턴스 정규화는 언어 문제들에서 많이 사용되고 있다.

## Why does normalization help?

- 최적화하기 쉽다.
- 초기 가중치값의 영향을 줄인다.


## 1D convolution 

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week05/05-3/Illustration_1D_Conv.png" width = 300>

```py
In [1]: import torch

In [2]: from torch import nn

In [3]: conv = nn.Conv2d?

In [4]: conv = nn.Conv2d

In [5]: conv = nn.Conv2d(2, 16, 3)

In [6]: conv
Out[6]: Conv2d(2, 16, kernel_size=(3, 3), stride=(1, 1))

In [7]: conv = nn.Conv1d(2, 16, 3)

In [8]: conv
Out[8]: Conv1d(2, 16, kernel_size=(3,), stride=(1,))

In [9]: conv.weight.size()
Out[9]: torch.Size([16, 2, 3])

In [10]: x = torch.randn(1, 2, 64)

In [11]: conv.bias.size()
Out[11]: torch.Size([16])

In [12]: conv(x).size()
Out[12]: torch.Size([1, 16, 62])

In [13]: conv = nn.Conv1d(2,16,5)

In [14]: conv(x).size()
Out[14]: torch.Size([1, 16, 60])

In [15]: # conv = nn.Conv2d(20, 16,

In [16]: x = torch.rand(1, 20, 64, 128)

In [17]: x.size()
Out[17]: torch.Size([1, 20, 64, 128])

In [18]: conv = nn.Conv2d(20, 16, (3,5))

In [19]: conv.weight.size()
Out[19]: torch.Size([16, 20, 3, 5])

In [20]: conv(x).size()
Out[20]: torch.Size([1, 16, 62, 124])

In [21]: conv = nn.Conv2d(20, 16, (3,5), 1, (1, 2))

In [22]: conv(x).size()
Out[22]: torch.Size([1, 16, 64, 128])
```
