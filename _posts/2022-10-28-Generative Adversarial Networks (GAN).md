---
title:  <font size="5">Generative Adversarial Networks (GAN)</font>
excerpt: "Generative Adversarial Networks (GAN)"
toc: true
toc_sticky: true
use_math: true
categories:
  - Machin Learning
tags:
  - Machin Learning
  - GAN
last_modified_at: 2022-09-15T18:09:00-55:00
---

--------
**<font size="4">Generative Adversarial Nets </font>** 

>Paper : <https://arxiv.org/pdf/2006.10204.pdf>


<div markdown = "1">
Generative Adversarial Nets 논문 리뷰를 위해 의역하여 작성.
<br><br><br>





#### <center>Abstract</center>

이 논문에서는 Adversarial(적대적) Process를 통해 새로운 프레임워크인 Generative Model(생성 모델)을 추정한다. 이러한 추정을 하기위해서는 두개의 모델을 학습한다 : Generative Model **$G$**는 Data Distribution을 따라가고, Discriminative model **$D$**는 Sample이 **$G$**에서 온것인지 Training Data에서 온것인지 판별한다. $G$에 대한 Training 절차는 $D$가 잘못 예측 할 확률을 최대화 한다. 즉, $D$와 $G$라는 두명의 Player가 진행하는 게임과 같다. 임의의 함수 $G$, $D$ 공간에서 $G$는 Training Data Distribution을 복구하여 $D$를 헷갈리게 함으로써, 결국 $D$는 판별하기 어려워하고 반반의 확률($\frac{1}{2}$)을 갖게된다.<font size="2">(Unique Solution이 존재한다.)</font> 이러한 $G$, $D$는 MLP(Multi-Layer Perceptrons)으로써 정의된 경우, 전체 시스템은 Backpropagation으로 학습 될 수 있다. 그리고 기존 Generative Model은 Training 또는 샘플의 생성과정 동안에는 Markov Chain 또는 unrolled approximate inference networks가 필요하였지만, GAN은 Neual Network로만 되어있다.

> <font size="2"> 요약하면, 결국 Discriminative model(판별 모델)은 $D$라고 부르고, Generative Model(생성 모델)은 $G$라고 부르며, $D$와 $G$가 서로 경쟁해 학습을 하게된다. </font>


<br><br>
#### <center>1. Introduction</center>

지금까지 Deep Discriminative Model은 많은 성공적인 Model이 있었지만, Deep Generative Model은 Maximum Likelihood Estimation 및 관련 전략에서 확률론적 계산을 근사화하기 어렵고 Generative Context에서 선형 활성함수의 이점을 활용하기 어려웠기 때문에 영향력이 적었다. 그래서 이 어려운 문제를 피하기위해, 새로운 Generative Model Estimation 절차를 제안한다.
<br><br>

제안된 Adversarial Nets Framework에서 Generative Model은 상대와 경쟁한다 : Discriminative Model은 Sample이 G Model Distribution으로부터 왔는지 Training Dataset Distribution으로부터 왔는지 판별하는것을 배운다. Generative Model은 위조지폐를 탐지 되지않게 생성하는 위조범과 유사하다고 볼 수 있다. 반면에, Discriminative Model은 위조지폐를 탐지하려고 노력하는 경찰과 유사하다고 볼 수 있다. 이러한 경쟁은 진짜와 가짜를 구별하기 어려울때까지 두 Model을 개선한다.<br>
이 논문에서는 Generative Model이 MLP를 통해 Random Noise로 부터 Sample을 생성하는 특별한 경우를 설명하고, Discriminative Model또한 MLP이다. 이러한 방식을 **Adversarial Nets**이라고 부른다. 두 모델 모두 Backpropagation과 Dropout Algorithms을 사용하여 학습 할 수 있고, 오직 Forward Propagation을 사용하여 Generative Model로 부터 Sample을 추출 할 수 있다. Approximate Inference 또는 Markov Chains는 필요하지 않다.

> <font size="2"> 기존의 어려웠던 문제를 해결하기 위해, 새로운 Generative Model Estimation 절차를 제안한다. 바로 $G$와 $D$가 위조범과 경찰이 되어 서로 경쟁하며 개선하는 것이다. 기존의 필요했던, Approximate Inference 또는 Markov Chains는 필요하지 않고, 두 모델 모두 MLP을 사용하고 Backpropagation을 통해 학습이 가능하다. </font>



<br><br>
#### <center>2. Related work</center>

RBMs / DBMs / DBNs / NCE / GSN 등 다양한 모델들이 있다.<br>
Adversarial Nets은 feedback loop가 필요없기 떄문에 Markov Chain이 필요없고, 이로인해 Backpropagation의 성능이 향상된다.<br>
Backpropagation을 통한 Generative Machine 학습방식은 Auto-Encoding Variational Bayes / Stochastic Backpropagation이 있다.<br>



<br><br>
#### <center>3. Adversarial nets</center>

Adversarial Modeling 프레임워크는 모델이 모두 MLP일때 적용하기 가장 쉽다. Data $x$에 대한 Generator의 Distribution $p_g$를 배우기 위해, Input Noise Variables $p_z(z)$를 정의하고, Data 공간에 대한 mapping을 $G(z;\theta_g)$로 나타낸다. 여기서 $G$는 MLP parameter $\theta_g$를 갖는 Differentiable(미분가능한) Function이다. 또한 Single Scalar값을 갖는 두번째 MLP $D(x;\theta_d)$를 정의한다. $D(x)$는 $x$가 $p_g$보다 Data에서 왔을 확률을 나타낸다. 우리는 D를 Training Example과 $G$로부터의 sample 두개의 Label 모두 옳은 정답을 한 확률을 최대화하여 학습한다. <font size="2">(즉, Real은 Real / Fake는 Fake로 구별할수 있게 학습한다)</font> 동시에 $log(1 - D(G(z)))$를 최소화하여 $G$를 학습한다 : <br>
다시말해, $D$와 $G$는 Value Function $V(G, D)$를 가지고 minimax game을 진행한다 : <br>

$$ \begin{align} \underset{G}{min}\underset{D}{max}V(D,G)=\mathbb{E}_{x\sim p_{data}(x)}[logD(x)]+\mathbb{E}_{z\sim p_z(z)}[log(1-D(G(z)))] \end{align} $$


<br>
다음은 이러한 Adversarial Nets의 Training criterion은 $G$, $D$에 충분한 Capacity가 주어질 때, Data generating distribution을 잘 복구 할 수 있음을 이론적 분석으로 제시한다. Training의 inner loop 안에서 $D$의 최적화를 완료하는 것은 계산적으로 어렵고, 유한한 데이터셋에서는 overfitting을 일으킬 것이다. 그렇기 때문에, $k$ step동안 $D$를 최적화하고 한번의 step동안 $G$를 최적화하는 것을 번갈아 진행했다. 결과적으로 $G$가 충분히 천천히 변한다면, $D$는 optimal solution 근처에 유지할것이다. 이러한 절차는 ***Algorithm 1***에서 제시된다.
<br><br>

사실, ***Eq 1***은 $G$가 잘 학습되기 충분한 gradient를 제공하지 않을수도 있다. 왜냐하면, 학습 초기 G의 학습이 덜 되었을때는, sample이 training data로 부터 명확히 다르기 때문에 $D$는 높은 confidence로 sample을 가짜라고 판별 할 수 있다. 이 경우에 $log(1-D(G(z)))$가 Saturates된다. 이런 경우 $log(1-D(G(z)))$를 minimize를 하기위해 G를 training하는것 보다, $logD(G(z))$를 maximize하도록 train하면 학습초기에 강력한 gradient 제공하여, 학습을 빠르게 진행 할 수 있다. <font size = 2>($log(1-x)$ 보다 $log(x))$의 gradient가 더 크기 때문에.)</font>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-10-28-Generative Adversarial Networks (GAN)/Figure 1.png){: .align-center}

***Figure 1***: Generative Adversarial Networks는 Data generating distribution(black, dotted line) $p_x$, Generative distribution($G$, green, solid line) $p_g$로 부터의 sample들을 구별하기 위해 Discriminative distribution($D$, blue, dashed line)와 동시에 업되이트 되어 학습된다. 또한 수평선 아랫쪽은 z가 uniform하게 샘플링된 경우의 도메인이다. 윗쪽 수평선은 $x$의 도메인의 일부이다. 윗쪽으로 향하는 화살표는 $x=G(z)$가 transformed sample에 대한 non-uniform distribution $p_g$를 어떻게 부과하는지 보여준다. $G$는 $p_g$의 high density 영역에서 수축하고 low density 영역에서 팽창한다. <br><br>
***(a)*** 어느정도 학습된 분포에서는 두 분포는 가깝다: $p_g$는 $p_{data}$와 비슷하다. 그리고 $D$는 부분적으로 정확한 Classifier다.<br>
***(b)*** Algorithm의 inner loop에서 $D$는 sample과 data가 구별되도록 학습되고, $D^\*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$로 수렴한다.<br>
***(c)*** $D$의 gradient는 $G(z)$가 data로 분류될 가능성이 높은 부분으로 유도한다. <br>
***(d)*** 몇 step training 후에, 만약 $G$와 $D$가 충분한 capacity를 갖고있다면, $p_g=p_{data}$로 둘다 향상되지 않는 지점에 도달할 것이고, 이때 Discriminator는 $D(x)=\frac{1}{2}$에 가까워져 두 분포사이의 차이점을 구별할수 없을 것이다.

<font size="2"> <div markdown = "1">
> - Generator Distribution : $p_g$ / Input Noise Variables : $p_z(z)$ / Dataset Distribution : $p_{data}(x)$ <br>
 - 이때, ***Eq 1***의 $$\mathbb{E}_{x \sim p_{data}(x)}$$는 Dataset Distribution에서 $x$를 샘플링한다는 뜻이고, $$\mathbb{E}_{z \sim p_z(z)}$$는 Input Noise Variables에서 $z$를 샘플링한다는 뜻. $p_z(z)$는 보통 Uniform / Gaussian Distribution을 사용<br>
 - $D$는 최대가 되고 $G$는 최소가 되도록 학습. => $D$는 +가 붙어(증가하도록), $G$는 -가 붙어(감소하도록) Backpropagation 진행<br>
 - 이를 활용해서, Value Function $V(G,D)$ ***Eq 1***를 가지고 Optimal Solution을 구한다. 서로 경쟁을 하기때문에, $D$를 $k$ step, $G$를 1 step 이렇게 반복적으로 학습시킨다. <br>
 - ***Figure 1***은 Training Data는 유한하기 때문에 검은색 점으로 나타내고, (b)에서 $D$가 학습되고 $D^\*(x)$는 section 4.1에서 설명한다. (c)에서 $D$에 의해 초록색 선 $G$는 점점 Data distribution을 따라가고, (d)에서는 충분한 step이 지난 후 학습이 완료된 모습을 보인다.<br>
</div> </font>

<br><br>
#### <center>4. Theoretical Results</center>

generator $G$는 $z \sim p_z$일 때 얻은 sample $G(z)$의 distribution으로 probability distribution $p_g$를 간접적으로 정의한다. 그러므로, 만약 충분한 capacity 그리고 training time이 주어진다면, ***Algorithm 1***이 $p_{data}$의 좋은 estimator로 수렴하길 원한다. 이 section의 결과는 Nonparametric setting을 통해 수행된다.<br>

<font size="2"> <div markdown = "1">
> Parametric(모수적) : 데이터가 특정 확률분포를 따른다고 선험적으로 가정한 후 그 분포를 결정하는 모수를 추정한는 방법.<br>
Nonparametric(비모수적) : 특정 확률분포를 선험적으로 가정하지 않고 데이터에 따라 모델의 구조 및 모수의 개수가 유연하게 바뀌는 방법
</div> </font>

section 4.1에서는 minimax game이 $p_g = p_{data}$에 대해 global optimum을 갖는다는 것을 보여준다. 그런다음 section 4.2에서 ***Algorithm 1***이 **Eq 1**을 최적화하여 원하는 결과를 얻는다는 것을 보여준다. 먼저 ***Algorithm 1***에 대한 설명을 보면,
<br>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-10-28-Generative Adversarial Networks (GAN)/Algorithm 1.png){: .align-center}



|***Algorithm 1*** Generative Adversarial Networks의 Minibatch stochastic gradient descent 학습. Discriminator에 적용할 단계의 수($k$)는 hyperparameter다. 여기서 가장 저렴한 $k=1$를 사용한다.|

<font size = 2> <div markdown = "1">
_**for** number of training iterations **do**_ <br>
&#160;&#160;&#160;&#160; _**for** $k$ steps **do**_ <br>
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160; $\bullet$ noise prior $p_g(z)$로 부터 $m$개의 noise samples $\begin{Bmatrix} z^{(1)}, .... , z^{(m)} \end{Bmatrix}$ Minibatch Sample 생성 <br>
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160; $\bullet$ data generating distribution $p_{data}(x)$로 부터 $m$개의 examples $\begin{Bmatrix} x^{(1)}, .... , x^{(m)} \end{Bmatrix}$ Minibatch Sample 생성<br>
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160; $\bullet$ stochastic gradient를 이용하여 증가하도록 Discriminator 업데이트:
<center> $\bigtriangledown_{\theta_d} \frac{1}{m}\sum_{i=1}^{m}[log D(x^{(i)})+log(1-D(G(z^{(i)})))].$ </center>

&#160;&#160;&#160;&#160; _**end for**_ <br>
&#160;&#160;&#160;&#160; $\bullet$ noise prior $p_g(z)$로 부터 $m$개의 noise samples $\begin{Bmatrix} z^{(1)}, .... , z^{(m)} \end{Bmatrix}$ Minibatch Sample 생성 <br>
&#160;&#160;&#160;&#160; $\bullet$ stochastic gradient를 이용하여 감소하도록 Generator 업데이트:
<center> $\bigtriangledown_{\theta_g} \frac{1}{m}\sum_{i=1}^{m}log(1-D(G(z^{(i)}))).$ </center>

_**end for**_ <br>
여기서는 momentum을 이용한 optimizer 사용

--------------------

> training criterion에서 $k$ step의 Discriminator을 학습시킨다. 이때 $D$의 Value fuction은 증가하는 방향(+)으로 학습을 시키고, $G$는 Value fuction +의 왼쪽 항이 $G$와 관여되있지 않기 때문에 생략되어 업데이트되고, 감소하는 방향(-)으로 학습시킨다.

</div></font>



<br><br>

##### 4.1 Global Optimality of $p_g=p_{data}$ <br>
먼저 Generator $G$에 대한 최적의 Discriminator $D$를 가정한다.<br>
***Proposition 1.*** $G$가 fix될 경우, 최적의 Discriminator $D$는 <br><br>
$$ \begin{align} D_{G}^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)} \end{align} $$

_Proof._ Generator $G$가 주어질때, Discriminator $D$에 대한 training criterion은 $V(G,D)$가 Maximize되는 것이다.
$$ \begin{align} V(G,D)&=\int_{x}^{}p_{data}(x)log(D(x))dx+\int_{z}^{}p_z(z)log(1-D(g(z)))dz \nonumber \\
 &=\int_{x}^{}p_{data}(x)log(D(x))+p_g(x)log(1-D(x))dx  \end{align} $$

$ \begin{Bmatrix} 0, 0 \end{Bmatrix} $이 아닌 집합 $(a,b)\in \mathbb{R}$에 대해서, 함수 $y \to a log(y)+b log(1-y)$는 $[0,1]$에서 $\frac{a}{a+b}$인 maximum을 갖는다.

<font size="2"><div markdown = "1">
> 기댓값은 모든 사건에 대해서 확률값을 곱하고 더하여 계산한다.
연속확률변수의 기대값은 연속확률분포함수의 면적이 되므로, 적분을 통해 구할수 있게된다. => 연속확률변수는 Discrete하지 않고 Continuous하기 때문에 곱셈합이 아닌 곱셈적분으로 해야된다.<br>
그래서 식을 구하면 ***Eq 1***을 ***Eq 3***과 같이 나타낼수 있게된다.
또한 $z$ domain에서 sampling된 noise vector를 $G$에 넣어서 $x$를 만들어 낼 수 있고, 이러한 과정은 domain $z$에서 $x$로 매핑 될 수 있기 때문에, 하나의 적분식으로 나타낼수 있다.<br>
***Eq 2***는 $y \to a log(y)+b log(1-y)$  $(a = p_{data}(x), b = p_g(x), y = D(x))$ 형식으로 나타낼수 있고, 이러한 형태는 y에 대해 미분을 하여 계산해보면 $\frac{a}{a+b}$에서 maximum을 갖는것을 알수있다.
</div></font>
<br>

$D$에 대한 훈련 목표는 conditional probability(조건부 확률) $P(Y=y|x)$ 추정을 위한 log-likelihood를 maximizing함으로서 해석될수 있으며, 여기서 $Y$는 $x$가 $p_{data}$(with $y=1$)로 부터 왔는지 아니면 $p_g$(with $y=0$)로 부터 왔는지를 나타낸다. 그러므로 ***Eq 1***에서 minimax game은 아래와 같이 수정될 수 있다: <br><br>
$$ \begin{align} 
C(G)&=\underset{D}{max}V(G,D) \nonumber\\
&= \mathbb{E}_{x\sim p_{data}}[log D^*_G(x)]+\mathbb{E}_{z\sim p_z}[log(1-D^*_G(G(z)))] \nonumber\\
&= \mathbb{E}_{x\sim p_{data}}[log D^*_G(x)]+\mathbb{E}_{x\sim p_g}[log(1-D^*_G(x))] \nonumber\\
&= \mathbb{E}_{x\sim p_{data}}\left [log \frac{p_{data}(x)}{P_{data}(x)+p_g(x)}  \right ]+\mathbb{E}_{x\sim p_g}\left [log \frac{p_{g}(x)}{P_{data}(x)+p_g(x)}  \right ]
\end{align} $$

<br><br>

***Theorem 1.*** 가상의 training criterion C(G)의 global minimum이 만약 $p_g=p_{data}$를 만족시킨다면, 이때, $-log 4$를 얻는다.<br>
_Proof._ $p_g=p_{data}$에 대하여 $$D^*_G(x)=\frac{1}{2}$$이므로, ***Eq2***와 ***Eq4***에 따라, $$C(G)=log\frac{1}{2}+log\frac{1}{2}=-log4$$를 갖는다. 이것이 $p_g=p_{data}$에 대해서 $C(G)$가 도달가능한 가능한 최선의 값인지 확인하기 위해서 살펴보면,
<center>$$ \mathbb{E}_{x\sim p_{data}}[-log2]+\mathbb{E}_{x\sim p_g}[-log2]=-log4 $$</center>
그리고 ***Eq 4***에서의 $$C(G)=V(D^*_G,G)$$에 위식을 뺌으로써, 아래와 같이 얻어진다.<br><br>
$$ \begin{align} 
C(G)&= \mathbb{E}_{x\sim p_{data}}\left [log \frac{p_{data}(x)}{P_{data}(x)+p_g(x)}  \right ]+\mathbb{E}_{x\sim p_g}\left [log \frac{p_{g}(x)}{P_{data}(x)+p_g(x)}  \right ] \nonumber \\
&= \mathbb{E}_{x\sim p_{data}}\left [log \frac{2*p_{data}(x)}{P_{data}(x)+p_g(x)}  \right ]+\mathbb{E}_{x\sim p_g}\left [log \frac{2*p_{g}(x)}{P_{data}(x)+p_g(x)}  \right ]-log(4) \nonumber \\
&= KL\left (p_{data}||\frac{p_{data}(x)+p_g(x)}{2} \right )+KL\left (p_g||\frac{p_{data}(x)+p_g(x)}{2} \right )-log(4) \\
&= 2*JSD(p_{data}||p_g)-log(4)
\end{align} $$

여기서 **KL**은 Kullback-Leibler divergence를 뜻하며, ***Eq 5***와 같은 식은 Jensen-Shannon divergence(**JSD**)로 변환될수 있다.
Jensen-Shannon divergence는 두 Distribution 사이를 Distance로 나타낸다. 그러므로 음(-)의 값을 갖지 않으며, 두분포가 동등할때 0을 갖는다. 그러므로, 두 분포가 같은 global minimum일때, JSD는 0이 되고 $$C^*=-log(4)$$는 C(G)의 global minimum 임을 보여준다.
<font size = 2><div markdown = "1">
>KL divergence의 식은 아래와 같고, ***Eq 3***에 KL divergence 식을 이용하여 ***Eq 5***와 같이 나타낼수 있다. <br>
><center>$$ KL(p_{data}||p_g)=\int_{-\infty}^{\infty}p_{data}(x)log\left ( \frac{p_{data}(x)}{p_g(x)} \right )dx $$ </center>
>그리고 JSD의 식은 아래와 같고, ***Eq 5***에 JSD 식을 이용하여 ***Eq 6***과 같이 나타낼수 있다. <br>
><center>$$ JSD(p||q)=\frac{1}{2}KL\left (p||\frac{p+q}{2} \right )+\frac{1}{2}KL\left (q||\frac{p+q}{2} \right )$$ </center>
>이로써 global optimum이 존재한다는 것을 알았다.
</div></font>


<br><br>
##### 4.2 Convergence of Algorithm 1 <br>
***Proposition 2.*** 만약, $G$, $D$가 충분한 capacity를 가지고 있다면, ***Algorithm 1***의 각 step에서 $G$가 주어졌을때, Discriminator는 optimum에 도달할것이다. 그리고 $p_g$는 criterion을 개선하기 위해 업데이트 되면, $p_g$는 $p_data$로 수렴한다.
<center> $$ \mathbb{E}_{x\sim p_{data}}\left [log D^*_G(x)  \right ]+\mathbb{E}_{x\sim p_g}\left [log (1-D^*_G(x))  \right ] $$ </center>
<br>

_Proof._ $V(G,D)=U(p_g,D)$를 $p_g$의 function에서 가정한다. $U(p_g,D)$는 $p_g$에서 convex function이다. convex function의 supremum의 subderivative는 항상 극대값이 얻어지는 지점의 기울기를 포함한다. 다른말로, 만약 $f(x)=sup_{\alpha \in A}f_{\alpha}(x)$와 $f_{\alpha}(x)$가 모든 $\alpha$에 대해 $x$에서 convex function이라면, $\beta=argsup_{\alpha \in A}f_{\alpha}(x)$라면 $\partial f_{\beta}(x) \in \partial f$이다. 이것은 최적의 $D$와 상응하는 $G$가 있을때 $p_g$에 대한 gradient descent를 update를 계산하는 것과 같다. $sup_DU(p_g,G)$는 ***Thm 1***에서 증명된 것처럼 특별한 global optima를 갖은 $p_g$에서 convex function이다. 그러므로 $p_g$의 충분하게 작은 업데이트로, $p_g$는 $p_x$로 수렴한다.

<font size = 2><div markdown = "1">
> ***upper bound*** : 상계라 부르며, 전체집합 $U$의 원소 중에서 부분집합 $S$의 모든 원소보다 크거나 같은 값<br>
***supremum*** : 상한이라고 부르며, 상계중 가장 작은 값을 말함<br>
***subderivative*** : 미분을 일반화 하여 미분가능하지 않은 볼록 함수에 적용할 수 있도록 하는 방법<br>
쉽게 말해서 ***Eq 3***에서 $p_g$에 대해서 gradient를 업데이트 한다고 하면, $p_g$에 대해 미분되어 (+)왼쪽항은 상수취급되어 생략되고, (+)의 오른쪽 항은 $log(1-D(x))$ term에 대해서만 영향을 받게 된다. 그러므로 function $V$는 $p_g$에 대해서 convex function 역활을 하게되고, 또한 $log(1-D(x))$도 상수처럼 행동하기 때문에 $V$는 $p_g$의 선형함수로 볼 수 있고, 그러므로 $p_g$는 적은 업데이트로도 충분히 잘 수렴할수 있다. 
</div></font>



<br><br>
#### <center>5. Experiments</center>

Adversarial nets을 MNIST, Toronto Face Database(TFD), CIFAR-10에 대해서 학습시켰다. Generator nets은 rectifier linear activation, sigmoid activation을 섞어서 사용하였다. 반면에 Discriminator net은 maxout activation을 사용하였다. Dropout은 Discriminator nets을 학습할때 적용되었다. 반면에 이론상으로는 Generator의 중간 계층에서 Dropout 및 기타 noise를 사용할 수 있지만, noise를 Generator Network의 맨 아래 계층에서만 input으로 사용했다. 
<br><br>

$G$를 통해 생성된 sample을 Gaussian Parzen window로 fitting함으로써 이 분포에서의 log-likelihood를 점수내어 표시하고, $p_g$에서 testset data의 확률을 추정한다. Gaussian의 parameter $\sigma$는 validation set에서 cross validation함으로 얻는다. 이러한 과정은 _Breuleux<font size = 2>(Quickly generating representative samples from an RBM-derived process)</font>_에서 소개되었고, 정확한 likelihood는 다루기 어렵기 때문에 다양한 Generative model에서 사용된다. 이러한 결과는 ***Table 1.***에서 보여준다. 이러한 likelihood를 추정하는 방식은 variance가 다소 높고 high dimensional spaces에서 잘 작동하지 않는다. 그러나 우리가 알고있는 최선의 방식이다. 

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-10-28-Generative Adversarial Networks (GAN)/Table 1.png){: .align-center}
<font size = 2><div markdown = "1">
***Table 1:*** Parzen window-based log-likelihood를 추정한다. MNIST는 test set sample의 mean log-likelihood이고, 여러 example에 대한 평균 standard error가 계산된다. TFD에서는 dataset의 여러 fold를 통해 standard error가 계산되고, 각 fold의 validation set을 사용하여 서로 다른 $\sigma$를 선택한다. TFD의 $\sigma$는 각 flod에 대해 cross validation 되었고, 각 fold의 mean log-likelihood를 계산한다. MNIST는 dataset의 real-valued version인 다른 모델과 비교한다.


>최근에는 Generative Model을 평가하는 새로운 방식을 사용하고 있음. 여기서는 Average Log-likelihood를 사용하고 있지만, 최근에는 Inception Score같은 방식을 많이 사용하고 있음.
</div></font>
<br><br>

***Figure 2, 3***는 Generator Net가 training 후에 그린 sample을 보여준다. 이러한 생성된 sample이 기존의 방식에 의해 생성된 sample보다 더 낫다고 주장하지는 않는다. 이러한 smaples은 더 나은 Generate Model과 경쟁할수 있고 믿고, Adversarial framework의 잠재력을 강조한다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-10-28-Generative Adversarial Networks (GAN)/Figure 2.png){: .align-center}
<font size = 2><div markdown = "1">
***Figure 2:*** 모델로부터 생성된 sample의 시각화. 가장 오른쪽 column은 모델이 training set을 외우지 않았다는것을 증명하기 위해, 생성된 이미지와 가장 가까운 학습데이터 이미지를 보여준다. sample은 cherry-pick이 아니라 random하게 그렸다. 대부분의 Deep Generative models의 시각화와는 달리, 이러한 이미지는 hidden units의 조건부 평균이 아니라 model distribution으로 부터 실제 sample을 보여준다. 게다가 sampling 과정은 Markov chain mixing에 의존하지 않기 때문에 이러한 sample들은 uncorrelate하다. $a)$ ***MNIST***  $b)$ ***TFD*** $c)$ ***CIFAR-10(fully connected model)*** $d)$ ***CIFAR-10(convolutional discriminator and "deconvolutional" generator)***
</div></font>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-10-28-Generative Adversarial Networks (GAN)/Figure 3.png){: .align-center}

<center><font size = 2><div markdown = "1">
***Figure 3:*** 전체 모델의 $z$ 공간에서의 좌표들을 선형 보간하여 얻은 숫자
</div></font></center>


![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-10-28-Generative Adversarial Networks (GAN)/Table 2.png){: .align-center}
<font size = 2><div markdown = "1">
***Table 2:*** Generative Modeling에서의 도전: 모델을 포함하는 각 주요작업에 대해 Deep Generative modeling에 대한 다른 접근법. 이에 대해, 직면하는 어려움에 대한 요약
</div></font>



<br><br>
#### <center>6. Advantages and disadvantages</center>

이러한 새로운 framework는 이전 모델링 framework와 비교하여 장단점이 있다. 
<br><br>
***disadvantages :*** 
1. $p_g(x)$의 명시적 표현이 없다.
2. $D$와 $G$를 잘 동기화해야함 (특히, $G$는 $p_{data}$를 modeling하기에 충분한 다양성을 갖기 위해 $z$의 너무많은 값들을 $x$의 동일한 값으로 분해시키는 "Helvetica scenario"를 피하기 위해, $G$는 $D$의 update없이 너무 많이 train되면 안된다.)
<font size = 2><div markdown = "1">
> 또한, Real과 Fake를 잘 분별하여 $G$를 학습시키려고, $D$를 빡세게 학습시키고 $G$를 학습시키기 시작하면 $G$가 학습을 포기해버림.
</div></font>

***Advantages :*** 
1. 오직 backprop으로 gradient를 얻기 때문에, Markov chain이 필요하지 않는다.
2. 학습중에 inference가 필요하지 않고, 다양한 functions에 통합될수 있다.
3. Generator Network가 data examples로 직접 업데이트 되지 않고, 오직 Discriminator를 통해 흐르는 gradient만으로 업데이트를 갖으므로 통계적 이점을 얻는다. (이것은 input data의 요소가 Generator의 parameter에 직접적으로 copy되지 않는다는 것을 의미한다.)
4. Markov chains을 썼을때 보다 더 sharp하게 결과 이미지를 나타낼수 있다.



<br><br>
#### <center>7. Conclusions and future work</center>

1. _Conditional Generative_ Model $p(x \mid c)$는 $G$와 $D$ input에 $c$를 추가하여 얻을수 있다.
2. _Learned approximate inference_ 는 $x$가 주어졌을때 $z$를 예측하기 위한 auxiliary network를 훈련함으로써 수행될수 있다. 
3. parameter를 공유하는 conditional model군을 훈련시킴으로써, $S$가 $x$ index들의 부분 집합일 때, 모든 조건 $p(x_S \mid x_{\not{S}})$를 대략적으로 모델링 할수 있다. 결정론적 MP-DBM의 stochastic extension을 구현하기 위해 Adversarial Nets을 사용할 수 있다.
4. _Semi-supervised learning_ : Discriminator 또는 Inference Net의 feature는 제한된 레이블 데이터를 이용할수 있을때, classifier의 성능이 향상 시킬수 있다.
5. _Efficiency improvements_ : training중에 $G$와 $D$를 조정하거나 sample $z$에 대한 더 나은 distribution을 결정함으로써, 훈련을 크게 가속시킬 수 있다.



<br><br>





</div>

