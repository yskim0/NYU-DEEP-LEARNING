# NYU Deep Learning - Week 10


## Success story of supervision : `Pre-training`

- 컴퓨터 비전이 최근까지 성공할 수 있었던 이유는 **supervised learning for ImageNet classification** 덕분.
- 이를 통해 다른 컴퓨터 비전 작업에 대한 초기화 값으로 사용함.

- 하지만, ImageNet과 같이 대규모 데이터셋에 대한 annotaion을 가져오는 것음 엄청난 cost가 필요함.
    - 레이블이 데이터 샘플 자체인 속성을 가지는 **Self-supervised learning**이 등장하게 됨.


## Self-supervised learning

<img width="500" alt="스크린샷 2020-09-09 오후 2 18 40" src="https://user-images.githubusercontent.com/48315997/92557534-5fdcfd80-f2a7-11ea-9ab7-e263068990da.png">


- two ways to define self-supervised learning
    - Basis supervised learning definition. *i.e.* 네트워크는 사람의 입력없이 반 자동적으로 레이블을 얻는 지도학습을 따른다.
    - Prediciton Problem. *i.e.* 데이터의 일부가 숨겨져 있고 나머지는 표시됨. 목표는 숨겨진 데이터를 예측하거나 숨겨진 데이터의 일부 속성을 예측하는 것임.

- Self-supervised learning vs. Supervised learning vs. Unsupervised learning
    - supervised learning : 사전 정의된(pre-defined) 레이블이 있음.
    - unsupervised learning : supervision, label, correct output 없이 데이터 샘플만 있음.
    - `self-supervised learning` : 주어진 데이터 샘플에 레이블을 co-occurring modality에서 가져오거나 데이터 샘플의 동시 등장 부분(co-occurring part)에서 가져옴.

### In NLP

- Word2Vec
    - 입력 문장이 주어지면, 해당 문장에서 누락 된 단어를 예측하는 작업이 포함되며, 누락된 단어는 pretext task를 위해 특별히 생략됨.
    - 레이블 집합은 어휘에서 가능한 모든 단어가 되며, 올바른 레이블은 문장에서 생략된 단어임.
    - 단어 수준 표현을 학습하기 위해 일반적인 그래디언트 방법을 이용하여 네트워크를 훈련시킬 수 있음.

### Why self-supervised learning?

- 자기지도학습은 데이터의 서로 다른 부분이 상호 작용하는 방식을 관찰하여 데이터 표현을 학습할 수 있음.
- 따라서 엄청난 양의 annotated data 가 필요하지 않음.
- 데이터 샘플과 연관될 수 있는 여러 양식들에 활용할 수 있음.

### In computer vision

- 일반적으로 자기 지도 학습을 사용하는 컴퓨터 비전 파이프 라인에는 pretext 작업과 실제(downstream) 작업이 포함된다.
    - 실제 (downstream) 작업은 분류 또는 탐지 작업과 같이 annotated data sample이 충분하지 않을 수 있다.
    - pretext 작업은 downstream 작업에 학습된 표현 또는 모델 가중치를 사용하기 위한 목적, 시각적 표현을 학습하기 위한 자기 지도 학습 과제이다.

### Pretext task

- 컴퓨터 비전 문제에 대한 pretext 작업은 이미지, 비디오 등을 사용해 개발 할 수 있음.
- 각 pretext 작업에는 일부는 표시되고 일부는 숨겨진 데이터가 있으며 작업은 숨겨진 데이터 또는 숨겨진 데이터의 일부 속성을 예측하는 것


![pred](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-1/img03.jpg)

- 예시 : predicting relative position of image patches

    - input : 2 image patches, one is anchor image patch. the other is the query image patch
    - 2개의 이미지 패치가 주어지면 네트워크는 앵커 이미지 패치에 대한 쿼리 이미지 패치의 상대적 위치를 예측해야 함.
    - 앵커가 주어졌을 때 쿼리 이미지에 대해 8개의 가능한 위치가 있으므로 `8-way` 분류 문제로 모델링 가능함.
    - 앵커에 대한 쿼리 패치의 상대적 위치를 입력하여 이 작업에 대한 라벨을 자동으로 생성 가능.


#### 상대 위치 예측 작업으로 학습한 시각적 표현

- 네ㅡ워크에서 제공하는 주어진 이미지 패치 기반 특징 표현에 대해, nearest neighbours를 통해 학습된 시각적 표현의 효율성을 평가할 수 있음.

- NN을 계산하려면,
    - 데이터셋의 모든 이미지에 대한 CNN features를 계산함. => 검색의 샘플 pool 역할을 함
    - 필요한 이미지 패치에 대한 CNN 특징 벡터 계산
    - 사용 가능한 이미지의 특징 벡터 풀에서 필요한 이미지의 특징 벡터에 가장 가까운 이웃을 식별

#### Predicting Rotation of Images

- 간단한 아키텍처를 가지고 있으며, 최소한의 샘플링이 필요한 가장 인기있는 pretext 작업  중 하나
- 이미지에 0, 90, 180, 270 도 회전을 적용하고 회전된 이미지를 네트워크로 전송하여 이미지에 적용된 회전 종류를 예측함.
- 네트워크는 단순히 4방향 분류를 수행하여 회전을 예측한다.

#### Colourisation

- 회색 이미지의 색상을 예측함.

> It is important to note that colour mapping is not deterministic, and several possible true solutions exist.

- 다양한 채색을 위해 VAE 및 잠재 변수를 사용하는 최근 작업이 있었음.

#### Fill in the blanks

- 이미지의 일부를 숨기고 이미지의 나머지 주변 부분에서 숨겨진 부분을 예측.
- 데이터의 implicit structure(암시적 구조)를 학습하기 때문에 작동


### Pretext Tasks for Videos

순서 예측, 공백 채우기 및 개체 추적과 같은 일부 pretext 작업에 활용될 수 있는 self-supervised 개념.

#### Shuffle & Learn

- 여러 개의 프레임이 주어지면, 세 개의 프레임을 추출하고 올바른 순서로 추출되면 positive, 셔플되면 negative로 레이블을 지정
    - 프레임의 순서가 올바른지 예측하는 binary classification 문제가 됨.
    - 따라서 시작점과 끝점이 주어지면 중간이 유효한 interpolation인지 확인하는 작업
    - 이는 세 개의 프레임이 독립적으로 feed forward하는 `Triplet Siamese` 네트워크를 사용할 수 있음.

- shuffle & learn을 할 때 색상 또는 의미 개념에 초점을 맞추고 있는지는 명확하깆 않음.
- human key-point estimation은 추적 및 포즈 추정에 유용함.

#### Pretext Tasks for videos and sound

![1](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-1/img12.png)

- 비디오 프레임을 vision subnetwork로 전달, 오디오를 audio subnetwork로 전달
- 프레임에서 소리가 나는 것을 예측하는 데 사용할 수 있음.


## Understanding what the “pretext” task learns

- Pretask 작업은 complementary해야 한다.
- 단일 pretext 작업은 SS 표현을 배우는 데 정답이 아닐 수 있다.
- pretext 작업은 예측하려는 내용이 크게 다르다.
    - contrastive methods는 pretext 보다 더 많은 정보를 생성함.



## Scaling Self-Supervised Learning

### Jigsaw Puzzles

----

## The hope of generalization

- hope that pre-training task and the transfer taske are **aligned**

## What we want from pre-trained features?

- represent how images relate to one another
    - `ClusterFit` : improving Generalization of Visual Representations
- Be robust to "nuisance factors" -- Invariance


Two ways to achieve the above properties
- `Clustering` -> ClusterFIt
- `Contrastive Learning` -> PIRL


## ClusterFit : Improving Generalization of Visual Representations

### Methods

1. Cluster : Feature clustering

![2](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-2/fig03.png)

2. Fit : Predict Cluster Assignment

![3](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-2/fig04.png)

-  train a network from scratch to predict the pseudo labels of images.
    - pseudo labels : obtained from first step through clustering

### "Standard" pretrain + transfer *vs.* "Standard" pretrain + ClusterFit

![4](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-2/fig05.png)

### Why ClusterFit Works

- In the clustering step only the essential information is caputred.


## Self-supervised Learning of Pretext Invariant Representations (PIRL)

### Contrastive Learning

- is basically a general framework that tries to learn a feature space that can combine together or put together points that are related and push apart points that are not related.

![5](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-2/fig10.png)

- Contrasitve learning and loss function

- Features for each of these data points would be extracted through a shared network, which is called Siamese Network
- Then a contrastive loss function is applied to **try to minimize the distance between the blue points as opposed**

### How to define related or unrelated?

- it’s not so clear how to define the relatedness and unrelatedness in this case of self-supervised learning.

- **The other main difference from something like a pretext task is that contrastive learning really reasons a lot of data at once.**


### Nearby patches vs. distant patches of an Image


![6](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-2/fig12.png)

### Patches of an image vs. patches of other images


![7](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-2/fig13.png)

- The more popular or performant way of doing this is to look at patches coming from an image and contrast them with patches coming from a different image. 
    - This forms the basis of a lot of popular methods like **instance discrimination**, MoCo, PIRL, SimCLR.


### Underlying Principle for Pretext Tasks

![8](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-2/fig14.png)

- pretext tasks always reason about a single image at once
    - the pretext tasks always reason about a single image.

- has to capture some property of the transform. 

### How important has invariance been?

- Invariance has been the word course for feature learning.


### PIRL

- PIRL stands for **pretext invariant representation learning**, where the idea is that you want the representation to be invariant or capture **as little information as possible** of the input transform.

#### Using a Large Number of Negatives

#### How it works

![9](https://atcold.github.io/pytorch-Deep-Learning/images/week10/10-2/fig16.png)

-  우리가 많은 negative 이미지를 원할 경우, negative 이미지를 동시에 feed-forward 하기를 원함. 이는 실제로 매우 큰 배치 크기가 필요하다는 것을 의미하지만 큰 배치는 좋은 해결책이 아니다.
-> Memory Bank 사용해서 해결

- Memory Bank 
    - 데이터 세트의 각 이미지에 대한 특징 벡터를 저장하고 대조적 학습을 수행
    - 다른 negative image나 배치 내 다른 이미지의 특징백터를 사용하는 것이 아니기 때문에, 메모리에서 특징 factor를 검색하는 방식
    - 메모리에서 관련되지 않은 다른 이미지(negative)의 특징 벡터을 검색 할 수 있으며 이를 substitute하면서 대조적 학습을 수행 할 수 있음.


### PIRL Pre-training

- by standard pre-training evaluation set-up


## Invariance vs. performance

- the invariance of PIRL is more than that of the Clustering, which in turn has more invariance than that of the pretext tasks.
- the performance to is higher for PIRL than Clustering, which in turn has higher performance than pretext tasks. 

## Shortcomings

- 어떤 데이터 변환이 중요한지 명확하지 않음.
- Saturation with model size and data size
- What invariances matter?
