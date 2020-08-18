# NYU Deep Learning - Week 1

## Traditional Machine Learning vs. Deep Learning

- Traditional ML : Feature Extractor(**Hand** engineered) -> Trainable Classifier
- DL : all Features are **Trainable**

## Multi-Layer Neural Nets

- Multiple Layers of simple units
- each units computes a **weighted sum** of its inputs
- Weighted sum is passed through a **non-linear function**
    (ex. ReLU)
- *The learning algorithm changes the weights*

## Supervised Machine Learning = Function Optimization

- comparing the resulting output with target output
- **optimize** the objective function which is the loss, computing a distance/penalty/divergence between the result and target. 
- computing **Gradient**
    - Assumption : **convex**
    - Stochastic Gradient Descent(SGD) : gradient descent with batch(small samples)

## Computing Gradients by Back-Propagation

- use **Chain Rule**

<img width="621" alt="스크린샷 2020-08-18 오후 4 04 14" src="https://user-images.githubusercontent.com/48315997/90481276-77bfd680-e16c-11ea-8b9a-c87e4bf8f974.png">


## Multilayer Architectures == Compositional Structure of Data

- In compositional hierarchy, combinations of objects at one layer in the hierarchy form the objects at the next layer


## The manifold hypothesis

- The answer of **"how can models learn representations (good features)?"**
- Natural data lives in a low-dimensional(non-linear) manifold
- Because variables in nature data mutually dependent


An ideal (and unrealistic) feature extractor represents all the factors of variation (each of the muscles, lighting, etc.).

## Linear Transformations

- Rotation
    - when the matrix is **orthonormal**
- Scaling
    - when the matrix is **diagonal**
- Reflection
    - **when the determinant is negative.**

*(?) Note that translation alone is not linear since 0 will not always be mapped to 0, but it is an affine transformation.*

- Affine transformation은 선형변환에서 이동변환까지 포함함. 


## Singular Value Decomposition(SVD)

- decompose a matrix into 3 componenet matrices, each representing a different linear transformation

<사진>

- U, V^T : orthogonal. rotation and reflection transformation
- middle matrix : diagonal. scaling transformation

## Non-linear transformations

- `tanh`


<img src="https://atcold.github.io/pytorch-Deep-Learning/images/week01/01-3/tanh_lab1.png" width="200" height="200">

- (?) *The effect of this non-linearity is to bound points between -1 and +1, creating a square.*

- When `s` increases, more and more points are pushed to the edge of the square.
    - By forcing more points to the edge, we can attempt to **classify** them.

    ![스크린샷 2020-08-18 오후 4 07 03](https://user-images.githubusercontent.com/48315997/90481513-ddac5e00-e16c-11ea-8925-93c201f92ba6.png)
