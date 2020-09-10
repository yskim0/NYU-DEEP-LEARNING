# NYU Deep Learning - Week 11

## Activation Functions

- ReLU

- RReLU

- LeakyReLU
- PReLU
- Softplus

- ELU
- CELU
- SELU
- GELU
- ReLU6
- Sigmoid
- Tanh
- Softsign
- Hardtanh
- Threshold
- Tanhshrink
![1](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-1/Tanhshrink.png)

- Softshrink
- Hardshrink
- LogSigmoid
![2](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-1/LogSigmoid.png)

- Softmin, Softmax
- LogSoftmax

---

## Loss Functions

- MSELoss
- L1Loss : |x_n-y_n|
    - L1 vs. L2 for CV
        - MSE(L2 loss) -> blurry image
        - L1 loss -> minimize the L1 distance is the medium.
        - L1 results in sharper image for prediciton.

- SmoothL1Loss
- NLLLoss : negative log likelihood loss
- CrossEntropyLoss : This function combines `nn.LogSoftmax` and `nn.NLLLoss` in one single class
- AdaptiveLogSoftmaxWithLoss
- BCELoss
- KLDivLoss 
- BCEWithLogitsLoss
- MarginRankingLoss
    - Margin losses are an important category of losses
- TripletMarginLoss
![3](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-2/tml.png)

- SoftMarginLoss
- MultiLabelMarginLoss
- HingeEmbeddingLoss
- CosineEmbeddingLoss
- **CTC Loss** : Connectionist Temporal Classification
    - Calculates loss between a continuous (unsegmented) time series and a target sequence.
    - The alignment of input to target is assumed to be “many-to-one”
    - Application Example: Speech recognition system
    - ![4](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-2/Fig2.png)

## Energy-Based Models (Part IV) - Loss Function

 ![image](https://user-images.githubusercontent.com/48315997/92727880-c8f96980-f3aa-11ea-8b3e-871ca26ea97a.png)

- Designing a Good Loss Function
    - **Push Down** on the energy of the correct answer.
    - **Push Up** on the energies of the incorrect answers.

### Examples of Loss functions

- Energy Loss

![스크린샷 2020-09-10 오후 9 17 18](https://user-images.githubusercontent.com/48315997/92728019-03630680-f3ab-11ea-8fd7-a80c3c8eb10e.png)

This loss function simply pushes down on the energy of the correct answer.

- Negative Log-Likelihood Loss

![스크린샷 2020-09-10 오후 9 22 02](https://user-images.githubusercontent.com/48315997/92728434-ad429300-f3ab-11ea-8c51-89c4ebf870a0.png)

pushes down on the energy of the correct answer while pushing up on the energies of all answers in proportion to their probabilities

- Perceptron Loss

### Generalized Margin Loss

**First, we need to define the Most Offending Incorrect Answer**

<img width="700" alt="스크린샷 2020-09-10 오후 9 23 55" src="https://user-images.githubusercontent.com/48315997/92728591-f0046b00-f3ab-11ea-8d6a-cb7dbe151d19.png">

- Examples
    - Hinge Loss
    - Log Loss
    - Square-Square Loss

- Other Losses

![5](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-2/other.png)


---

## Prediction and Policy learning Under Uncertainty (PPUU)

### Cost

- Lane cost
- Proximity cost
![6](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-3/figure6.png)

### Variational predictive network

![7](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-3/figure10.png)

- The z_t is chosen such that the MSE is minimized for a specific prediction. By tuning **the latent variable**, you can still **get MSE to zero by doing gradient descent into latent space.**
    - but this is very expensive
- So, **we can actually predict that latent variable using an encoder.** Encoder takes the future state to give us a distribution with a mean and variance from which we can sample z_t.


![8](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-3/figure12.png)

- Variational predictive network - inference

### Action insensitivity & latent dropout

![8](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-3/figure15.png)

- problem is an  information leak. 
    - fix this problem by simply dropping out this latent and sampling it from the prior distribution at random

![9](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-3/figure18.png)

### Training the agent

![10](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-3/figure21.png)

- We want the prediction of our model after taking a particular action from a state to be as close as possible to the actual future

- Our cost function now includes both the task specific cost(proximity cost and lane cost) and this expert regulariser term

- Now as we are also calculating the loss with respect to the actual future, we have to remove the latent variable from the model because the latent variable gives us a specific prediction, but this setting works better if we just work with the average prediction.

### Minimizing the Forward Model Uncertainty

![11](https://atcold.github.io/pytorch-Deep-Learning/images/week11/11-3/figure24.png)

- Uncertainty regulariser based model architecture
