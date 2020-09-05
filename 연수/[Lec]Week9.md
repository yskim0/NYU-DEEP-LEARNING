# NYU Deep Learning - Week 9

## Discriminative recurrent sparse autoencoder(DrSAE)

![1](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-1/q7pSvUJ.png)

- 인코더 We는 LISTA에서 쓰이는 인코더와 유사하다.
- 신경망을 세 가지 기준에 따라 학습시킨다.
    - L1: 희소하게 만들기 위해 특징 벡터 Z를 기준으로 L1을 적용한다.
    - reconstruction X  : 출력 단게에서 입력값을 재생산하는 디코딩 행렬을 이용함에 따라 실행된다. Wd가 나타내는 제곱근 오차를 최소화함으로써 이루어짐.
    - Add a Third Term(Wc) : 카테고리 예측하기 위해 시도하는 단순 선형 분류기

- 위의 세 가지 기준을 동시에 최소화하기 위해 학습된다.
- 이점 : 시스템으로 하여금 입력값을 재구성할 수 있는 표현들을 찾도록 함.
    - 기본적으로 가능한 입력값에 대한 정보를 많이 담고 있는 특징들을 추출하는 쪽으로 시스템을 편향하고 있음. => 특징들을 강화시킴.

### Group Sparsity

- 희소한 특징들을 생성하기 위한 것이며, 단지 합성곱에 의해 추출되는 보통의 특징들이 아니라 기본적으로 **풀링 이후 희소해지는 특징**들을 만들어내기 위한 것

![2](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-1/kpDK8Xu.png)

- L1으로 향하는 잠재변수 Z대신, L2를 통하여 그룹을 넘어간다.
- 한 그룹 안에 있는 각 요소들에 대한 L2 norm.을 얻고, 이 norm.들의 합을 얻는다.
    - `regulariser`로 사용되고, Z의 그룹에서 희소성을 가질 수 있다.


## AE with group sparsity

- image restoration or some kind of self-supervised learning 
    - small dataset일때나 가능할 것

> When you have an encoder and decoder that is convolutional and you train with group sparsity on complex cells, after you are done pre-training, the system you get rid of the decoder and only use the encoder as a feature extractor, say the first layer of the convolutional net and you stick a second layer on top of it.

![3](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-1/7akkfhv.png)

- 그룹 희소성을 동반한 합성곱 RELU 구조
- `stack autoencoder` : *You can also train another instance of this network. This time, you can add more layers and have a decoder with the L2 pooling and sparsity criterion, train it to reconstruct its input with pooling on top. This will create a pretrained 2-layer convolutional net.*

----


## World models for autonomous control

자기지도학습의 가장 중요한 활용 중 하나는 `world models for control`을 배우는 것이다.


### World model?

자율 지능 시스템은 4개의 모듈로 이루어진다.
![4](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-2/week9_world_models_arch.png)

- 1. 지각 모듈은 세계를 관찰하고 상태 표현을 계산한다.
    - 불완전함 : 에어전트가 모든 세계를 관찰하지 않고, 관찰 정확도가 제한적이기 때문에.
    - feed-forward 모델에서 지각 모듈이 초기 시간 단계에 대하여 유일하게 존재한다는 것에 주목
- 2. 실행 모듈(정책 모듈)은 표현된 세계 상태를 기반으로 약간의 동작을 취하는 것을 예상한다.
- 3. 모델 모듈은 표현된 세계 상태가 주어진 동작의 결과를 예측하고, 약간의 가능성이 있는 잠재 특징이 주어진다.
    - 이 예측은 다음 상태에 대한 추측으로서의 다음 시간 단계로 넘어가고, 초기 시간단계부터 비롯된 인지모듈의 역할을 가진다.
- 4. 비평 모듈은 제안된 행동 수행 비용으로 들어가는 예측하는 방향으로 가게 된다. 
    - ex. given the speed with which I believe the pen is falling, if I move muscles in this particular way, how badly will I miss the catch?

![5](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-2/week9_world_models.png)


## The classical setting

classical optimal control에서는 actor/policy module이 없고 **행동 변수(action variable)만 존재한다.**

- 이 공식은 `Model Predictive Control` 이라 불리는 고전적인 방법에 의해 최적화되어 있다.

> We can think of this system as an unrolled RNN, and the actions as latent variables, and use backpropagation and gradient methods (or possibly other methods, such as dynamic programming for a discrete action set) to infer the sequence of actions that minimizes the sum of the time step costs.

## An improvement

![6](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-2/week9_policy_network.png)

- 매번 복잡한 역전파 처리과정을 거치는 것은 귀찮으므로, VAE에서 sparse coding을 개선하기 위해 쓰였던 똑같은 속임수를 사용한다.
    - 세계 표현에서 비롯한 최적의 행동 순서를 직접적으로 예측하기 위하여 인코더를 훈련시킨다. 이 체제에서 인코더는 **정책 신경망(policy network)**이라 불린다.

- 한번 훈련시키고 나면, 우리는 policy network를 이용하여 perception 이후의 optimal action sequence를 예측할 수 있다.

## Reinforcement Learning(RL)

- 지금까지의 내용과 강화학습과의 차이점
    - 1. 강화학습 환경에서는 cost function이 black box이다. 즉 에이전트는 보상 체계를 이해하지 못한다.
    - 2. 강화학습 설정에서, 그 환경으로 가기 위해 세계의 forward 모델을 사용하지 않는다. 대신에 실제 세계와 상호작용하고 무슨 일이 벌어지는지 관찰하는 것으로써 결과를 배운다. 

- 강화학습의 문제점 : 비용 함수가 미분 불가능함.

- Actor-Critic methods are a popular family of RL algorithms which train both an actor and a critic.
    - 많은 RL 방법들이 비용함수 모델(critic)을 학습시킴으로써 비슷하게 동작한ㄷ.
    - critic의 역할은 가치 함수의 기대값을 학습하는 것이다. 이는 모듈을 통한 **역전파를 가능**하게 한다.
    - actor는 환경을 걷어들이는 행동들을 제안


## Generative Adversarial Network

GAN에는 많은 변형 모델이 있지만, 여기서는 **대조적 방법을 사용하는 에너지 기반 모델의 GAN**을 다루고자 함.

- 대조 표본 에너지를 밀어올리고, 학습 표본 에너지를 밀어내린다.


기본적으로 GAN은 크게 두 파트로 이루어진다.
- 대조 표본을 생산하는 **Generator**
- 본질적으로 비용함수이면서 에너지 모델로서 행동하는 판별 장치 **Discriminator**
- 위 두개 모두 신경망임.

GAN의 input : training samples, contrastive samples
- training samples에 대하여 GAN은 판별기를 통해 이러한 표본들을 통과하고 에너지를 줄인다.
- contrastive samples에 대하여 GAN은 약some distribution으로부터 잠재변수를 샘플링하고, training sample과 유사하게 만들도록 generator를 돌린다. 그리고 이 샘플들의 에너지를 밀어 올리기 위해 판별기에 통과시킨다.


<br>

판별기에 대한 손실함수 : sigma L_d(F(y), F(y_bar))
- L_d는 F(y)를 감소시키고 F(y_bar)를 증가시킨다.
- 이러한 맥락에서 y는 레이블이고, y_bar는 y 스스로를 제외하고 가장 낮은 에너지를 부여하는 응답 변수이다.

<br>

발생기에 대한 손실함수 : L_g(F(y_ber)) = L_g(F(G(z)))
- z는 latent variable
- G : generator neural net

<br>

이것이 생산적 적대 신경망이라 불리는 이유는 서로 양립할 수 없고 그들을 동시에 최소화해야하는 두 가지 객체 함수를 가지고 있기 때문이다. 
- 이는 경사 하강 문제가 아닌데, 왜냐하면 목표가 이 두가지 함수 사이의 내쉬 균형을 찾기 위한 것이 아니며 경사하강은 기본값에 의해 이를 통제할 수 없다.

<br>

true manifold에 가까운 sample들을 가지게 되면 문제가 생길 수 있다.
- 무한히 얇은 매니폴드를 가지고 있다고 가정해보자. 
    - 판별기는 매니폴드 바깥에서 0인 확률과 매니폴드에서의 무한한 확률을 만들어내야 한다.
    - 이것은 어려워서 GAN은 시그모이드를 사용하고, 매니폴드 밖에서는 0, 매니폴드상에서는 1을 만든다.
    - 만약 우리가 매니폴드 바깥에서 0을 만들고자 얻은 판별기에서 시스템을 성공적으로 학습시킨다면 **에너지 함수는 완전히 쓸모없어진다**
        - 에너지 함수가 smooth하지 않기 때문 (데이터 매니폴드 바깥에서의 에너지는 모두 무한대가 될 것이고, 데이터 매니폴드 상에서의 에너지는 모두 0인곳에서)
    - 연구자들은 에너지 함수 규제화를 통하여 이 문제를 고치기 위한 많은 방법을 제안함. 그 중 판별기 가중치 사이즈를 제한하는 Wasserstein GAN(WGAN)이 있다.

----

## GAN

![7](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-3/GANArchitecture.png)

- GAN은 unsupervised machine learning 에서 사용되는 신경망 유형이다.
- 두 적대적 모듈 Generator, Cost 신경망으로 구성된다.

> These modules compete with each other such that the cost network tries to filter fake examples while the generator tries to trick this filter by creating realistic examples x_hat. Through this competition, the model learns a generator that creates realistic data. They can be used in tasks such as future predictions or for generating images after being trained on a particular dataset.

![8](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-3/GANMapping.png)

- GAN은 EBM 모델의 예시이다. 
- 비용 신경망은 분홍색 x에 의해 정의되는 true data 분포에 근접한 입력값을 위한 낮은 비용을 생산하기 위하여 훈련된다.
    - 파란색 x와 같은 분포로 오는 데이터는 높은 비용을 가질 것임.
- 한편, 비용 신경망을 속이기 위한 실제 생성 데이터에 대한 랜덤 변수 z의 맵핑을 개선하고자 발생 신경망을 훈련시킨다.
    - 발생기는 비용함수의 출력값에 대한 점과 더불어 훈련이 이루어지며, x_hat의 에너지를 최소화하기 위하여 노력한다.

![image](https://user-images.githubusercontent.com/48315997/92299892-71659300-ef91-11ea-8a32-98002f5d8308.png)

- 높은 비용이 데이터 매니폴드 바깥지점에, 낮은 비용이 그 범위 안에 배치되는 것을 위하여 비용 신경망 L_c에 대한 손실함수는 위와 같다.


## GAN과 VAE 차이점

![9](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-3/GANvsVAEArchitecture.png)

![10](https://atcold.github.io/pytorch-Deep-Learning/images/week09/09-3/GANvsVAEMapping.jpg)


- GAN은 z를 샘플링하는 것으로 시작함. (이부분은 VAE의 잠재 공간과 비슷하다)
    - z를 x_hat과 연결하고자 Generator를 이용함.
    - 이 x_hat은 판별장치/비용 신경망을 통하여 보내지는데, 얼마나 진짜같은지를 평가하기 위한 것

- 가장 큰 차이점 중 하나는 **우리가 생성 신경망 x_hat의 출력값과 실제 데이터 x 사이의 직접적인 관계(즉 reconstruciton loss)를 판단할 필요가 없다는 것이다.**
    - 대신, 판별 장치/비용 신경망이 실제 데이터 x 혹은 더 실제 같은 것들과 비슷한 점수를 만드는 x_hat을 만들기 위한 generator를 학습시킴으로써 x_hat과 x가 비슷해지도록 강제한다.



## Major pitfalls in GANs

1. Unstable convergence

- 발생기가 학습할수록 개선좋아짐으로써, 판별 장치 성능은 더 나빠지는데 실제와 가짜 데이터 사이의 차이점을 더이상 쉽게 말해줄 수 없기 때문이다. 만약 발생기가 완벽하다면, 실제와 가짜 데이터의 매니폴드는 각각 맨 윗부분에 놓일 것이고 판별 장치는 많은 분류오류를 만들어낼 것이다.

- 판별 장치 피드백은 시간이 흐를수록 의미가 없어진다.
- 결과적으로 발생기와 판별 장치 사이에는 균형보다 불안정한 균형점이 있다.

2. Vanishing gradient

3. Mode collapse

## DCGAN code

### Generator

```py
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 입력값은 합성곱으로 가는 Z이다.			
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 상태 크기. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 상태 크기. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 상태 크기. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 상태 크기. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 상태 크기. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output
```
- 순차적으로 끝날 때, 신경망은 출력값을 (-1,1)(−1,1)로 낮추기 위하여 ‘nn.Tanh()’을 사용한다.

### Discriminator

```py
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 입력값은 (nc) x 64 x 64이다.
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
```

- 음의 영역에서 경사 vanishing을 막기 위해 LeakyReLU 사용
- 순차적으로 끝날 때, 판별 장치는 입력값을 분류하기 위하여 ‘nn.Sigmoid()’를 사용한다.


### Training


- 판별 장치 신경망을 업데이트

```py
# 실제 데이터로 학습시킨다.
netD.zero_grad()
real_cpu = data[0].to(device)
batch_size = real_cpu.size(0)
label = torch.full((batch_size,), real_label, device=device)

output = netD(real_cpu)
errD_real = criterion(output, label)
errD_real.backward()
D_x = output.mean().item()

# 가짜 데이터로 학습시킨다.
noise = torch.randn(batch_size, nz, 1, 1, device=device)
fake = netG(noise)
label.fill_(fake_label)
output = netD(fake.detach())
errD_fake = criterion(output, label)
errD_fake.backward()
D_G_z1 = output.mean().item()
errD = errD_real + errD_fake
optimizerD.step()
```

- 발생 신경망을 업데이트

```py
netG.zero_grad()
label.fill_(real_label)  # 가짜 레이블은 발생 비용에 대한 실제이다.
output = netD(fake)
errG = criterion(output, label)
errG.backward()
D_G_z2 = output.mean().item()
optimizerG.step()
```
