# Generative Adversarial Networks (2014)
---
<br>

## 1. Introduction
- deep learning이 Discriminative task에서 괄목할 만한 성과를 보이고 있음
- 그러나 generative models에서는 크게 다음과 같은 두 가지 이유때문에 deep learning의 적용이 어려움

    1) *difficulty of approximating many intractable probabilistic computations*
    2) *difficulty of leveraging the benefits of piecewise linear units* (while it contributes a lot in the discriminative context)
   
위의 두 가지 어려움을 극복한 generative model estimation procedure를 제안함. 
→ ***Adversarial Nets*** framework

- adversarial nets framework?
  - two-player game (two agents are each generator and discriminator)
  - 생성자 G (이하 G)는 데이터의 분포를 학습하고 샘플은 생성해고, 판별자 D (이하 D)는 해당 샘플이 dataset으로부터 온 것인지 아님 model이 생성해낸 결과인지를 판별해내는 과정이 지속적으로 이루어지는 구조.
  - 두 모델 간의 경쟁방식은 D가 진짜와 가짜를 구분할 수 없을 때까지 계속 D와 G의 성능이 향상되도록 학습을 이끌어나감.

- 논문에서는 G가 random noise를 입력으로 받아 샘플을 생성해내는 과정과 D가 이를 판별해내는 과정을 모두 multilayer perceptron으로 구현함. → *Adversarial Nets*
- 이를 통해 architecture의 학습 과정은 backpropagration과 dropout 알고리즘만을 활용하여 학습이 가능해지고, 이전의 approximate inference 혹은 markov chain 등의 과정은 필요하지 않게 됨.


## 2. Related Work (reference를 얼마나 공부해야될지?)
![generative_taxonomy](../figures/generative_models_taxonomy.png)
- undirected graphical models with latent models (such as Resricted Boltzmann machines, deep Boltzmann machines)
  - this quantity and its gradient are **intractable** for all but the most trivial instances
- hybrid models (such as Deep belief Networks)
  - single undirected layer + several directed layers
  - undirected와 directed model이 갖는 computational difficulties를 모두 갖게 됨.
- score matching and noise-contrastive estimation (NCE)
  - do not approximate or bound the log-likelihood
  - 위 종류의 모델에서는 학습한 확률 밀도를 정규화된 상수로 해석적으로 명시 가능한 형태로 표현해야 함. 그러나 대부분의 경우 잠재 변수로 구성된 여러 개의 layer를 쌓은 모델은 계산이 가능한 형태의 정규화되지 않은 확률 밀도를 유도해내는 것 자체가 불가능함.
  - **NCE** 같은 경우에는 **생성 모델을 fitting하는 데에 discriminative training criterion을 적용**. 하지만 별도의 discriminative model을 사용한 것이 아니라 generative model 내에서 판별을 수행한다는 점 & 고정된 noise distribution을 사용하기 때문에 학습이 굉장히 느리다는 점 (느리다는 게 주안점인 건지 아니면 학습이 이상한 방향으로 빠질 수 있다는 게 주안점인 건지는 모르겠음)
- implicit density model (such as GSN)
  - 확률 밀도를 명시적으로 정의하지 않고 생성 모델을 원하는 distribution에서 sampling 가능하도록 학습시키는 방식
  - 역전파가 가능하다는 점에서 good
  - parameterized Markov Chain
  - GAN은 sampling에 있어서 markov chain을 요구하지 않는다는 점에서 이점이 있음.
  