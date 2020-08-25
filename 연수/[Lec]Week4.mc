# NYU Deep Learning - Week 4

## STFT (Short-Time Fourier Transform)

국소 푸리에 변환

- 기존의 푸리에 변환 : **해당신호가 주파수 영역으로 보았을 때 어떤 주파수의 성분을 어느만큼 가지고 있는지** 가시적으로 표현함.
    - 하지만 시간의 흐름에 따라 신호의 주파수가 변했을 겨웅에, 어느 시간대 주파수가 어떻게 변했는지 알 수 없음.

=> STFT : 시간에 따라 주파수 성분이 변하는 신호의 `time-frequency` 정보를 어떻게 효율적으로 분석할 수 있을까에서 나옴

- 즉, 데이터에서 시간에 대해 구간을 짧게 나누어 나누어진 여러 구간의 데이터를 각각 푸리에 변환하는 방법


## Property: locality

## Property: stationarity

## Different dimensions of different types of signals.

![그림](https://atcold.github.io/pytorch-Deep-Learning/images/week04/04-1/fig7.png)
