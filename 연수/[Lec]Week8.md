# NYU Deep Learning - Week 8

## 에너지 기반 모델의 대조적 방법

1. 다른 부분의 에너지 값을 올리면서, F(xi, y') 훈련 데이터 포인트의 에너지를 낮추는 **대조적 방법**, F(xi,yi)
2. 정규화 방식을 이용해 낮은 에너지 영역을 최소화/제한하는 에너지 함수 F를 만드는 **구조적 방법**

- Maximum Likelihood method : 데이터 포인트의 에너지를 낮추고 다른 곳의 에너지를 모두 증가시킨다.
    - 훈련 데이터 포인트의 에너지 값들을 확률적으로 낮추고, y' /= yi인 다른 모든 데이터의 에너지 값들을 낮춘다.
    - 최대 우도는 에너지의 절대값이 아닌 오직 에너지의 차이에만 관심을 둔다.
    - 확률 분포의 합은 항상 1이 되도록 정규화되기 때문에, 주어진두 개의 데이터 포인트 사이의 비율을 비교하는 것이 단순히 절대값을 비교하는 것보다 유용함.


## Self-Supervised Learning에서의 Constrastive Methods

대조적 방법에서는 관측된 훈련 데이터 포인트의 에너지를 낮추고, 훈련 데이터 manifold 외부에 조냊하는 데이터 포인트들의 에너지를 높인다고 위에 설명하였다.

<br>

**대조적 임베딩 방법(constrastive embedding methods)** 를 self-supervised learning에 사용하는 것이 지도 학습 모델에 견줄만한 성능을 내놓을 수 있음을 발견함.


### 대조적 임베딩

**positive**
    
![1](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-1/fig1.png)

x가 이미지이고 y가 x를 transformation한 (x,y)의 쌍
- 대조적 임베딩은 CNN 방식을 취하고, x와 y를 이 신경망에 입력하여 두 가지 feature vectors h, h'를 얻는다.
- x, y는 positive 쌍이기 때문에, 이 특징 벡터가 비슷해야 한다.
- h와 h' 사이의 유사도를 최대화 하는 `similarity metric` (ex.cosine similarity)과 손실함수를 선택한다.
- 이렇게 함으로써, 훈련 데이터 매니폴드가 이미지에 갖는 에너지 값을 낮춘다.

<br>

**negative**

![2](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-1/fig2.png)

하지만 매니폴드 바깥 포인트들의 에너지도 올려야한다.

따라서 **negative** sample이 필요한데, 이는 내용이 다른 이미지를 만들어내는 것이다.


=> 유사한 쌍(positive)의 에너지 값을 낮추고, 유사하지 않은 쌍들의(negative) 에너지 값을 높인다.

### Self-Supervised Result(MoCo, PIPL, SimCLR)

![3](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-1/fig3.png)


PIPL의 목적함수 NCE(Noise Constrastive Estimator)

![image](https://user-images.githubusercontent.com/48315997/92114174-6395fd80-ee2b-11ea-92b9-b22f37f4ce1c.png)

- 코사인 유사도 사용
- PIPL의 특이점 : Convolutional feature extractor에서 바로 나온 출력을 직접 사용하지 않는다.
    - 대신 다른 f, g 정의
        - 기본 합성곱 특징 벡터 추출기 위의 독립적인 레이어

- 미니 배치에서, 하나의 positive 쌍과 다수의 negative 쌍을  가진다.
- 변환된 이미지들의 특징 벡터 I^t와 미니 배치 안의 나머지 특징 벡터(positive 한 개, 나머지 negative) 사이의 유사도를 계산한다.
- 다음으로 positive 쌍에서 소프트맥스와 같은 점수로 score 계산
    - **소프트 맥스 함수를 최대화하는 것은 나머지 점수를 최소화하는 것과 같고, 이것이 에너지 기반 모델에서 원하는 것임.**
- 따라서 최종 손실 함수를 이용하여 유사한 쌍에서는 에너지를 낮추게 하고, 유사하지 않은 쌍에서는 에너지를 높이도록 한다.


위의 과정을 위해서는 많은 양의 negative 샘플들이 필요하다.


## Denoising autoencoder

![4](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-1/fig6.png)

- 손상된 데이터가 데이터 manifold에서 멀어짐에 따라 에너지 함수가 2차적으로 증가하도록 시스템을 훈련시킴

- 문제점
    - high dimensional continuous space에서 데이터 조각을 손상시키는 수 많은 방법들이 존재함.
    - 많은 위치에서 단순히 에너지를 높여나가며 에너지 함수를 형성해나갈 수 있다는 보장이 없음.
    - **잠재 변수를 갖지 않기 때문에 이미지 처리에 있어 성과가 낮음.**

## Other constrastive Methods

contrastive divergence, Ratio Mathing, Noise Contrastive Estimation, Minimum Probability Flow...


### Contrastive Divergence

입력 샘플을 똑똑하게 손상시켜서 데이터 표현을 학습하는 모델.

- 연속 공간에서 훈련 샘플 y를 고르고 이것의 에너지를 낮춤.
- 이 샘플에 대해서 일종의 gradient-based process를 사용해서 노이즈가 있는 에너지 표면에서 아래로 이동함.
- 에너지값이 낮으면 가지고 있고, 낮지 않다면 어떤 확률값에 따라 그것을 버린다.
- 위 과정을 반복하면 y의 에너지 값이 낮아진다.
- y와 대조 샘플 y_bar의 손실함수 값을 비교하여 매개 변수를 업데이트할 수 있음.

### Persistent Contrastive Divergence

대조 발산의 개선된 모델 중 하나임.

이 시스템은 많은 **particles**를 사용하고, 그 위치를 기억함.
- 이 입자들은 CD에서 그랬던 것처럼, 에너지 표면에서 아래로 이동함.
- 결국 이 입자들은 에너지 표면 상에서 낮은 에너지들의 위치를 파악하고 이들의 에너지를 높인다.
- 그러나 이 시스템은 차원의 확장에 따른 스케일 조절이 잘 되지 않음.

-----


## Regularized latent Variable Energy Based models

잠재변수를 포함하는 모델은 관측된 입력 x와 추가적인 잠재 변수 z에 따라 예측 y_bar의 분포를 만들 수 있다.
에너지 기반 모델 또한 **잠재 변수**를 포함할 수 있다.

![5](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-2/fig1.png)

- 잠재변수 z가 최종 예측 y_bar를 만들어내는데 지나친 power를 갖게 되면, **모든 실제 출력 y는 적절하게 선택된 z를 바탕으로 입력 x에서 완벽하게 재구성된다.**
    - 이는 y와 z 모두에 대해 inference 과정 동안 에너지가 최적화되어 모든 지점의 에너지 함수 값이 0이 됨을 의미한다.

- 이에 대한 해결책은 **잠재 변수 z의 정보 용량을 제한**하는 것이다. ex. 잠재변수 정규화
    - `E(x,y,z)=C(y,Dec(Pred(x),z))+λR(z)`
    - 이 방법은 값이 작은 z의 공간과 이에 대해 낮은 에너지를 갖는 y의 공간을 제어하는 값을 제한한다.
    - lambda 값은 이러한 트레이드 오프를 조절함
    - R의 유용한 예시로는 L1 norm인데, 이는 거의 모든 곳에서 미분 가능한 유효 차원의 근사치로 볼 수 있다.
    - *L2 norm을 제한하면서 z에 노이즈를 추가하여 정보 내용(VAE)도 제한할 수 있다.*


### Sparse Coding

`E(x,y)= ||y-Wz||^2 + lambda||z||_{L1}`

- The n-dimensional vector z will tend to have a maximum number of non-zero components m « n
    - 각각의 Wz는 W의 m개 칼럼의 span에 속하게 된다.

- After each optimization step, the matrix W and latent variable z are normalized by the sum of the L_2 norms of the columns of W. This ensures that W and z do not diverge to infinity and zero.

### FISTA

![6](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-2/fig2.png)

||y-Wz||^2와 lambda||z||_{L1} 두 항을 번갈아 가며 최적화하며 z에 대한 스파스 코딩 에너지 함수 E(y,z)를 최적화 하는 알고리즘이다.

- Z(0)으로 초기화

![스크린샷 2020-09-03 오후 9 44 11](https://user-images.githubusercontent.com/48315997/92116299-9d1c3800-ee2e-11ea-8d1e-41e6484c7549.png)

- 내부 항은 ||y-Wz||^2 항의 gradient
- Shrinkage 함수는 0쪽으로 이동하며 lambda||z||_{L1}을 최적화 한다.

### LISTA

FISTA는 고차원 대규모 데이터셋(이미지)에 적용하기에는 비용이 크다.

**이를 효율적으로 만든ㄹ기 위해 최적의 잠재 변수 z를 예측하도록 네트워크를 훈련시키는 방법이 LISTA이다.**

![7](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-2/fig3.png)

- 예측된 잠재변수와 최적의 잠재 변수 z의 차이를 측정하는 추가 항이 더해진다.


![스크린샷 2020-09-03 오후 9 46 26](https://user-images.githubusercontent.com/48315997/92116548-ec626880-ee2e-11ea-9dc8-b66116b2c970.png)


![8](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-2/fig4.png)

- We의 그래디언트는 표준BPTT를 사용해 계산됨.
- 훈련이 완료된 네트워크는 FISTA보다 적은 반복으로 좋은 z를 만들어낸다.


## Variational autoencoder

![9](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-2/fig8.png)

정규화된 잠재변수 EBM에서 희소성(sparse)부분만을 제외한 것과 유사한 구조를 가지고 있다.

잠재변수 z는 z에 대한 에너지 함수를 최소화하여 계산되는 것이 아니라, **에너지 함수는 로그값이 z_bar에 연결되는 비용의 분포에 따라 랜덤하게 z를 샘플링**하는 것으로 가눚된다.
- 이 분포는 평균이 z_bar인 가우시안 분포이고, 이에 따라 z_bar에 가우시안 노이즈가 추가된다.


![10](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-2/fig10.png
- 정규화 없는 에너지 최소화 과정에 따른 퍼지볼의 움직임


<br>

이 시스템은 코드 벡터 z_bar를 가능한 한 크게 만들어서 z(노이즈)의 영향을 최소화 하고자 한다.
- 크게 만드는 다른 이유 : 디코더가 서로 다른 샘플을 혼동하게 하는 overlapping을 방지하기 위해


하지만, 만일 매니폴드가 있다면 퍼지볼들이 주변에서 클러스터링 되기를 원한다.
따라서 코드 벡터는 평균과 분산이 0에 가까워지도록 정규화된다.
- 이를 위해 코드 벡터들을 원점과 스프링으로 연결한다.

![11](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-2/fig11.png)
- 스프링의 강도 : 퍼지볼이 원점과 얼마나 가까이 위치할지를 결정
    - 너무 약하다면, 굉장히 멀리 위치한다
    - 너무 강하면, 원점에 지나치게 가까이 붙어서 높은 에너지 값
    - 이 모두를 방지하기 위해, 시스템은 해당 샘플이 유사한 경우에만 서로 겹치는 것을 허용한다.

- 퍼지볼의 사이즈는 조정이 가능하다.
    - 서로 붙어버리지 않도록 분산을 1에 가깝게 만드는 패널티 함수(KL Divergence)에 의해 제한된다.


----


## VAE vs. AE

![12](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-3/fig_1.png)

- 고전적 AE에서는 h를 generating
- VAE : 잠재변수 z를 가지고 가우시안 분포를 가지는 E(z), V(z) 만든다.
    - smaple `z` from the above distribution parametrized by the encoder. Specifically, E(z)and V(z) are passed into a sampler to generate the latent var. `z`
    - `z` is passed into the decoder to generate x


## VAE loss function

![13](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-3/fig_2.png)

- reconstruction항과 regularization 항으로 이루어짐.


## reparameterization trick

`z`를 얻기위해 가우시안 분포에서 샘플을 추출하는데, 이는 경사하강법을 수행할 때 샘플링 모듈을 통해 어떻게 역전파를 수행해야하는지 모르게 되는 문제점이 생긴다.

따라서 `sampling z`를 위해 **reparameterization trick**을 사용한다.
- 위 그림에서 z에 대한 식 참조
- 이 경우 훈련에서의 역전파가 가능해진다. (요소 별 곱셈과 덧셈을 거침)

## VAE loss function 분리

VAE 손실 함수는 재구성 항과 정규화 항을 갖는다.

![스크린샷 2020-09-04 오후 1 41 10](https://user-images.githubusercontent.com/48315997/92200238-4c4f2280-eeb4-11ea-85bd-5c7028ef6d34.png)

추정된 각각의 z값을 2d 공간의 원으로 생각해볼 수 있는데 여기서 E(z)는 원의 중심이고 주변 영역은 V(z)에 의해 결정되는 z의 가능한 값이다.

![14](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-3/fig_3.png)

- 각각의 원은 추정된 z의 영역을 나타내며, 화살표는 어떻게 재구성 항이 각각의 추정된 값들을 다른 값으로 밀어내는지를 보여준다.
- 만일 z의 추정된 값들 중 어느 두 개 사이에 겹치는 부분이 있다면(두 원이 겹치는 경우) 재구성 할 때의 **모호성**을 만들어낸다.
- 따라서 재구성 손실은 점들을 서로 밀어낸다.
- 하지만 계속해서 밀어낸다면 시스템이 폭발할 수 있기 때문에 벌칙항이 필요하다.

![스크린샷 2020-09-04 오후 1 44 24](https://user-images.githubusercontent.com/48315997/92200415-bf589900-eeb4-11ea-85f6-38b25ac5a3b8.png)


## penalty 항

VAE 손실함수에서 아래 항을 전개하면 다음을 얻는다.

![스크린샷 2020-09-04 오후 1 47 58](https://user-images.githubusercontent.com/48315997/92200591-3f7efe80-eeb5-11ea-8daa-678ce6884ee0.png)


위 식에서 `v_i = V(z_i) - log(V(z_i)) -1`를 따로 빼내고 그래프를 그려보면 아래와 같이 나온다.

![15](https://atcold.github.io/pytorch-Deep-Learning/images/week08/08-3/fig_4.png)

- z_i의 분산이 1일 때 표현식이 최소화 된다.
- 그러므로 penalty loss는 잠재변수들의 분산을 약 1로 유지시킨다. 이는 시각적으로 원들이 약 1의 반지름을 가질 것임을 의미한다.
- 마지막 항 E(z_i)^2는 z_i 사이의 거리를 최소화하여 재구성 항이 초래할 수 있는 폭발을 방지한다.

