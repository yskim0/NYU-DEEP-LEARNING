# NYU Deep Learning - Week 6

## Zip Code Recognition

### Recognition with CNN

## Face Recognition

- 30x30 window 크기를 가진 CNN을 학습 시킨다면 2가지 문제점이 있음.
    - False Positive : 거짓 양성 오류
    - 다른 얼굴 크기 => **Multi-scale** 버전 만들기

### Multi-Scale Face Recognition

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-1/CQ8T00O.png" width = 400>

- Non-maximum suppression : 중첩된 bounding box들 중에서 가장 높은 점수의 한 개만 유지하고 나머지는 버림.
- Negative Mining : False Positive 문제 해결을 위해서 고안된 방법
    - 모델이 얼굴로 인식하지만 얼굴이 아닌 이미지 조각을 만들어 negative dataset을 만듦.
    - 네거티브 데이터셋에 대해 얼굴 인식 모델을 다시 학습시킴.
    - 이 과정을 통해 robustness을 높일 수 있음.

## Semantic Segmentation

### Scene Parsing and Labeling

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-1/VpVbkl5.jpg" width=500>

- Multi-Scale CNN
- CNN 출력을 입력에 다시 input시키면 46*46 크기의 `Laplacian Pyramid`가 만들어짐.
    - **중앙 픽셀의 범주를 결정하기 위해 46*46 픽셀의 context를 사용하고 있음을 의미함.**
- 그러나 context size는 더 큰 개체의 범주를 결정하기에 충분하지 않음.
- **Multi-Scale 방식은 추가적으로 다시 스케일된 이미지를 입력으로 제공하면서 더 넓은 비전을 가능하게 한다.**
    - 과정
    1. 동일한 이미지를 2,4배 줄인다.
    2. 이 두 개의 추가로 크기 조정된 이미지들은 동일한 ConvNet에 입력되고 Level 2 Features를 두 세트 얻는다.
    3. 이 피쳐값들을 Upsample해서 원본 이미지의 레벨 2 피쳐값과 동일한 사이즈가 되도록 한다.
    4. 세 개 세트의 특징값을 쌓아서 classifier에 입력한다.


## Recurrent Networks

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-2/RNN_unrolled.png" width = 400>

- input : x1,x2,x3
- 이 네트워크는 순환고리가 없기 때문에 경사하강법을 구현할 수 있음.
- 모든 블록이 같은 가중치를 공유한다. 
- 세 개의 인코더, 디코더, G 함수는 각각 서로 다른 시간 단계상에서 같은 가중치를 갖는다.
- BPTT(Backprop through time)이 잘 작동하지 않는다.

<br>

**RNN의 문제점**
- 경사 소멸
- 경사 폭발


## 곱셈 모듈

## Attention

## Gated Recurrent Units(GRU)

- GRU는 곱셈 모듈의 활용으로서 경사값 소멸/폭발 문제를 풀기위해 고안되었음.
- 리셋게이트 r과 업데이트 게이트 z로 두 가지 게이트가 있음.
    - 리셋 게이트 : 새로운 입력을 이전 메모리와 어떻게 합칠지 정해주고, 업데이트 게이트는 이전 메모리를 얼만큼 기억할지 정해줌.
    - 리셋 게이트 = 1, 업데이트 게이트 = 0 => RNN과 동일
<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-2/GRU.png" width = 300>

- LSTM과의 차이점
    - GRU 게이트 수 = 2, LSTM 게이트 수 = 3
    - GRU는 입력 게이트와 forget 게이트가 z로 합쳐졌음. LSTM은 나누어졌다고 생각할 수 있음.
    - GRU는 출력값을 계산할 때 추가적인 비선형 함수를 적용하지 않음.
    
## Long Short-Term Memory(LSTM)

![스크린샷 2020-08-28 오후 7 39 52](https://user-images.githubusercontent.com/48315997/91552131-3f877780-e966-11ea-8115-e17451b6733d.png)

- lstm 유닛은 셀상태 c_t를 이용하여 정보를 전달함.
    - 셀 상태의 정보가 유지되거나 제거될지는 게이트를 이용하여 결정함.
- forget gate f(t): 현재의 입력과 이전 상태를 통해 이전 셀 상태인 c_t-1로부터 어느정도의 정보를 유지할 것인지를 정하고, c_t-1 의 계수로 0과 1사이의 숫자를 출력함
- tanh : 셀 상태를 업데이트할 새 후보를 계산하고, 포겟 게이트처럼 입력게이트 i_t는 어느정도의 업데이트가 적용될지 결정함.

- i,f,o : 각각 입력, 까먹음, 출력 게이트. 각 게이트의 수식은 동일한 형태이며 파라미터 행렬만 다름.
    - 게이트라고 부르는 이유는, sigmoid 함수가 이 벡터들의 값을 0~1 로 제한시키고, 이를 다른 벡터와 elementwise곱을 취한다면 그 다른 벡터값의 얼마만큼을 통과시킬지 정해주는 것과 같기 때문.
    - i : 새 hidden state 값을 계산하는데 있어서 입력 벡터값을 얼만큼 사용할지 정해줌
    - f : 이전 state 값을 얼만큼 기억하고 있을지 정해줌
    - o : 현재 내부 state 값의 얼마늠을 LSTM 모듈의 바깥쪽에서 볼 수 있을지 정해줌
    - 모든 게이트들은 d_s로 hidden state와 같은 차원을 가짐.

- c_t : LSTM 유닛의 내부 메모리. 이전에 저장된 메모리인 c_t-1과 까먹음 게이트의 곲, 그리고 새로 계산된 hidden state g와 입력 게이트의 곱을 합친 형태
    - 이전 메모리와 현재 새 입력을 **어떻게 합칠까**에 대한 부분
    - f = 0 -> 이전 메모리 모두 무시. i = 0 -> 새로운 입력값 모두 무시

- 메모리값 c_t가 주어지면, 메모리와 출력게이트의 곱으로 최종적으로 hidden state s_t가 계산됨

>  입력 게이트를 전부 1로 두고, 까먹음 게이트를 전부 0으로 (이전 메모리는 무조건 까먹는 것으로) 하고, 출력 게이트를 전부 1로 설정한다면 (메모리 값 전부를 보여줌) RNN 모델 기본형과 거의 같습니다. 출력값을 특정 범위 내로 압축시키는 tanh만 추가된 형태일 것입니다. LSTM의 이 게이팅 메커니즘이 모델에서 긴 시퀀스를 확실히 잘 기억하도록 (long-term 의존도를 잘 고려하도록) 하는 핵심 기법입니다.

## Sequence to Sequence 

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-2/Seq2Seq.png" width = 400>

- 인코더-디코더 구조
    - 인코더, 디코더 모두 다층 LSTM으로 이루어짐

- 왼쪽의 인코더 : 시계열 단어의 수는 번역될 문장의 길이와 같음.
    - 마지막 시간 단계의 레이어는 문장 전체의 의미를 함축하는 벡터를 출력하고 이 값이 디코더로 전달됨.
- 디코더에서 단어들은 순서대로 생성됨

- 한계
    - 문장 전체의 의미가 인코더와 디코더 사이의 은닉 상태값으로 압축되어 들어가야 함.
    - LSTM은 20단어 이상으로 정보를 저장하지 못함.
        - Bi-LSTM 등장

## Seq2Seq & Attention

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-2/Seq2SeqwAttention.png" width = 400>

- 문장 전체를 하나의 벡터로 압축시키는 것보다, **각 시간 단계에서 기존의 언어의 같은 의미를 갖는 특정 위치에 시스템을 집중**시키는 **어텐션** 방식이 더 설득력있음.

> 어텐션에서 각 시간 단계에서 현재의 단어를 생성하기 위해서 우리는 입력 문장의 어떤 단어의 은닉 표현에 집중할 것인지를 결정할 필요가 있다. 필수적으로, 네트워크는 인코딩된 입력이 현재 디코더의 출력과 얼마나 잘 맞는지를 점수로 평가할 수 있게 된다. 이 점수가 소프트맥스에 의해 정규화 되고, 계수들은 인코더의 서로 다른 시간 단계에서 은닉 상태들의 가중합을 계산하는데 사용된다. 가중치를 조정하며 시스템은 입력값에서 집중할 부분을 찾아낼 수 있다. 여기서 마법은 이러한 계수들을 역전파 방식으로 알아낼 수 있다는 점이다. 

## Memory Network

뇌 속에 두 가지 중요한 부분이 있다는 아이디어로부터 시작됨 : 대뇌피질, 해마체

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-2/MemoryNetwork1.png" width = 300>

- 메모리 네트워크에서는 x가 있고, 이 x를 k1,k2,k3 ... 와 내적을 통해 비교함.
- 이후에 소프트맥스에 넣어 합이 1이 되는 숫자들의 array를 얻는다.
- 벡터 v1,v2,v3 ... 를 소프트맥스에서 나오는 스칼라로 곱한 후 더해 결과를 얻는다.

> 메모리 네트워크에서는 입력 값을 받아 메모리의 주소를 출력하는 신경망이 있고, 이를 이용해 값을 네트워크로 가져온 후, 최종적으로 결과를 출력한다. 이는 CPU와 외부 메모리에 읽고 쓰는 컴퓨터와 매우 흡사하다.

## RNN 개요

> RNN은 데이터의 시퀀스sequence를 다루는 구조의 한 종류이다. 시퀀스란 무엇인가? CNN 수업에서 신호는 도메인에 따라 1차원, 2차원, 3차원도 될 수 있다는 것을 배웠다. 도메인이란 우리가 ‘어디로 부터 매핑을 하였는지’ 와 ‘어디로 매핑을 하는지’에 따라 정의가 내려지며, 도메인은 그저 X에 대한 일시적인 입력이기 때문에, 시퀀스 데이터를 다루는 것은 기본적으로 1차원 데이터를 다루는 것과 동일하다. 그럼에도 불구하고 RNN을 이용해서 두 방향을 가진 2차원 데이터를 다루는 것이 가능하다.

### Vec -> Seq

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-3/vec_seq.png" width=400>

- ex. 이미지를 입력으로 갖고, 입력 이미지를 설명하는 단어의 시퀀스를 출력하는 모델

### Seq -> Vec

심볼들의 시퀀스를 지속적으로 feed하고 끝에서만 최종 출력을 가짐.

### Seq -> Vec -> Seq

언어번역의 대표적 방법

### Seq -> Seq

입력을 feed함과 동시에 신경망은 출력을 갖기 시작함.

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-3/fourth.png" width = 500>

## BackProp through time (BPTT)

RNN 모델을 학습시키기 위해서는 BPTT가 반드시 사용되어야 함.

시간에 따른 역전파 그림은 아래와 같음.

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-3/bptt.png" width = 400>

## Vanishing and Exploding Gradient

- det. > 1 => exploding
- eigenvalue .=. 0 => vanishing


이상적인 해결방법은 `skip connection`이다.
이와 같은 과정을 위해 신경곱(multiply networks)이 사용된다.

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-3/rnn_2.png" width = 400>

> 위와 같은 경우, 기존의 신경망을 4개 영역으로 나눈다. 첫번째 신경망을 보면, 시간 1에서의 입력값을 갖고 출력값을 은닉망의 처음 중간상태로 보낸다. 이 상태는 경사도를 통과시키는\circ∘와 경사 전파를 막는 -−로 이루어진 3개의 다른 신경망으로 구성된다. 이러한 기법을 게이트 순환 신경망 gated recurrent network이라고 한다. 

## LSTM

<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week06/06-3/lstm.png" width = 500>


