* `model.cuda()`
  - 모든 model parameter를 GPU 버퍼에 옮기는 것
  - input batch, tensor, model에 쓰임
  - optimizer를 설정하기 전에 실행되어야 한다. 
  - pre-0.4 way (old version)
  - asynchronous
 
* `model.to(device)`
  - When loading a model on a GPU that was trained and saved on GPU, simply convert the initialized model to a CUDA optimized model using model.to(torch.device('cuda'))
  - **여러 개의 gpu에 올릴 수 있다.** (model.cuda()와 다른 점인 듯)
  - model.cuda() 보다 flexible함
  > Then if you’re running your code on a different machine that doesn’t have a GPU, you won’t need to make any changes. If you explicitly do x = x.cuda() or even x = x.to('cuda') then you’ll have to make changes for CPU-only machines.
  
* `model. eval(), train()`
  - eval(test), train 모드로 변경한다.
  - **dropout이나 batchnorm 쓰는 모델은 학습시킬 때와 평가할 때의 구조와 역할이 다르기때문에 반드시 명시한다.**

* `zero_grad()`
  - 모든 model parameter의 gradient를 0으로 설정한다.
