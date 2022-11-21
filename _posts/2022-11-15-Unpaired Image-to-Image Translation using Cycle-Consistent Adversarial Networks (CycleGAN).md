---
title:  <font size="5">CycleGAN</font>
excerpt: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)"
toc: true
toc_sticky: true
use_math: true
categories:
  - Machin Learning
tags:
  - Machin Learning
  - GAN
  - CycleGAN
last_modified_at: 2022-11-15T18:09:00-55:00
---

--------
**<font size="4">CycleGAN 논문 리뷰</font>** 

>Paper : Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN) <https://arxiv.org/pdf/1703.10593v6.pdf>


<div markdown = "1">

#### <center>CycleGAN</center>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 1.png){: .align-center}
Image-to-Image translation은 aligned image pairs를 학습에 사용하여 input image에서 output image로 mapping 시키는것을 목적으로 한다. 
Pix2Pix에서 발견된 고질적인 문제는, "pair된 image만 사용이 가능하다."는 것이다. <br>
CycleGAN은 이러한 문제점을 해결하기 위해 unpaired image 상황에서, _source_ _domain_ $X$에서 _target_ _domain_ $Y$로 translation하는것을 학습을 목표로 하며, adversarial loss를 사용하여 $G(X)$의 image distribution이 $Y$의 distribution과 구별될수 없도록 mapping $G:X \to Y$를 학습한다. 왜냐하면, 이러한 mapping은 제약이 낮기 때문에, inverse mapping $F:Y->X$를 결합하고, $F(G(X)) \approx X$(또는 그 반대)가 되도록 cycle consistency loss를 추가한다.



<br><br><br>
#### <center>1. Introduction</center>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 2.png){: .align-center}

***Figure2***에서 왼쪽이미지는 paired training example이다. 이러한 paired training data는 얻기 어렵고 비용이 비싸다. 그래서 CycleGAN은 paired input-output example없이 domain사이의 translation을 배우는 algorithm을 제시한다. 비록 paired example보다는 성능이 떨어지지만, 불리한 조건에서 unpaired image를 학습하여 비교할 만한 성능을 내고있다.<br>

이러한 방식을 설명하면, domain $X$($x \in X$) 와 또 다른 domain $Y$($y \in Y$)에서, $\hat{y}$와 $y$를 구분하도록 적대적으로(adversary) 학습된 모델에 의해 $y \in Y$ 가 구분되지 않는 output $\hat{y}=G(x)$를 내보내도록 mapping $G:X \to Y$를 학습한다.

> image $x$를 넣어서 나온 output $\hat{y} = G(x)$과 $y$를 구분할수 있도록 학습된 discriminator에 의해 mapping $G:X \to Y$를 학습한다.

이론적으로 이러한 목표는 empirical distribution $p_{data}$와 매칭되는 output distribution $\hat{y}$를 유도한다. 따라서, optimal $G$는 $Y$와 동일하게 분포된 domain $\hat{Y}$로 domain $X$를 translation시킨다. 하지만 그러한 traslation이 의미있는 방식으로 $x$와 $\hat{y}$를 짝지는다고 보장되지는 않는다. $\hat{y}$과 동일한 distribution을 유도하는 수많은 mapping $G$가 있다. 즉, $G$는 $\hat{y}$이 $Y$와 비슷하게만 만들면 된다. 이러한 결과는, mode collapse가 일어나기 쉽다. 

> mode collapse는 비슷하게만 만들면 되니까, 다른 input image에 대해서도 매번 같은 결과만 나타내는것을 의미한다.

그러므로, "cycle consistent"를 이용한다. 예를들어, English to French로 번역하고 다시 French to English로 번역하면 다시 원래 문장으로 돌아가야 할 것이다. 이와 마찬가지로, 만약 translator $G:X \to Y$와 $F: Y \to X$가 있을때, $G$와 $F$는 inverse 관계여야한다. 그리고 두 mapping은 bijection되어야만 한다. $G$와 $F$는 동시에 학습되며, $F(G(x)) \approx x$, $G(F(y)) \approx y$가 되도록 ***cycle consistency loss***를 추가한다. 이러한 loss를 domain $X$와 $Y$의 adversarial losses와 결합하면 unpaired image-to-image translation을 위한 objective function이 완성된다.

> 1. $G:x \to y$ (일반적인 adversarial loss)
2. $F:y \to x$ (역방향의 adversarial loss)
3. $F(G(x)) \approx x$, $G(F(y)) \approx y$ (cycle consistency loss)
CycleGAN은 이 3가지 loss를 학습한다.



<br><br><br>
#### <center>2. Related work</center>

**Generative Adversarial Networks (GANs)** <br>
GAN의 핵심 아이디어인 adversarial loss가 translated image와 target domain에 있는 image들과 구분되지 않도록 학습

<br>
**Image-to-Image Translation** <br>
pix2pix와 같은 pair image를 학습하는 prior work와 다르게, paired training example없이 mapping을 배운다.

<br>
**Unpaired Image-to-Image Translation** <br>
CycleGAN의 방식은 이전의 input과 output사이의 task-specific, predefined similarity function, low-dimensional embedding space 방식과는 다르게 general-purpose solution을 제안한다. Section 5.1. 에서

<br>
**Cycle Consistency** <br>
이러한 방식은 이전에도 사용하였으며, CycleGAN에서는 이러한 similar loss를 추가한다.

<br>
**Neural Style Transfer** <br>
painting $\to$ photo와 같은 다양한 분야로 확장이 가능하다.



<br><br><br>
#### <center>3. Formulation</center>

CycleGAN은 training sample $\begin{Bmatrix} x_i  \end{Bmatrix}_{i=1}^N$이 주어질때, domains $X$와 $Y$사이의 mapping function을 학습하는것이 목표다.<br>
***Figure 3*** 왼쪽을 보면 $G:X \to Y$, $F:Y \to X$를 나타내고, 두개의 discriminator $D_X$와 $D_Y$를 추가로 도입한다. 여기서 $D_X$의 목표는 image $x$와 translated image $F(y)$를 구별하는것이다. $D_Y$도 마찬가지로 $y$와 $G(x)$를 구별한다. 전체적으로 두가지 loss를 결합하여 최종학습을 진행한다. 
1. generated images의 distribution을 target domain에서의 data distribution과 일치시키기 위한 _adversarial_ _losses_
2. 학습된 mapping $G$와 $F$가 서로 모순이 되는것을 방지하기 위한 _cycle_ _consistency_ _losses_

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 3.png){: .align-center}


<br>
##### 3.1. Adversarial Loss

Adversarial Loss는 기존의 GAN과 같이 있을법한 image를 만들어내는데 사용한다. CycleGAN에서는 역방향도 같이 진행해주어 translation된 이미지가 다시 original image로 translation된다.
mapping function $G:X \to Y$와 이에 해당하는 discriminator $D_Y$의 objective function은 아래와 같이 나타낸다.

<br>
$$ \begin{align} 
L_{GAN}(G,D_Y,X,Y)=\mathbb{E}_{y \sim p_{data}(y)}[\log D_Y(y)]+\mathbb{E}_{x \sim p_{data}(x)}[\log(1-D_Y(G(x)))]
\end{align}$$ 
<br>

$G$는 domain $Y$의 image와 비슷하게 보이는 image $G(x)$를 생성하려고 한다. 반면에 $D_Y$는 translated sample $G(x)$와 real sample $y$를 구별하려고 한다. 이러한 objective function을 $D$ maximize하려고 시도하고, 이에 반대로 $G$는 minimize하는것을 목표로 한다. (i.e., $\underset{G}{min}\underset{D_Y}{max} L_{GAN}(G,D_Y,X,Y)$)
mapping function $F:Y \to X$도 discriminator $D_X$에 대해서 똑같이 진행한다.(i.e., $\underset{F}{min}\underset{D_X}{max} L_{GAN}(F,D_X,Y,X)$)


<br>
##### 3.2. Cycle Consistency Loss

Adversarial Loss가 그럴듯한 image를 내보내도록 학습이 되었다면, Cycle Consistency Loss는 원본이미지와 같이지도록 하는 Loss와 같다.
***Figure 3***(b)에서 파란색 점 두개의 차이를 줄이는것이 Cycle Consistency Loss와 같은것이다. 즉, cycle translation 후 원본과 최대한 유사하게 만들려고 시도한다. ***Figure 3***(b)와 같이 각 image $x$는 $G$에 의해 translation되고 $F$에 의해 original image로 다시 translation될 수 있어야한다.(i.e., $x \to G(x) \to F(G(x)) \approx x$) 이것을 ***forward cycle consistency***라고 한다. ***Figure 3***(c)도 비슷하게 ***backward cycle consistency***를 진행한다. (i.e., $y \to F(y) \to G(F(y)) \approx y$)

<br>
$$ \begin{align} 
L_{cyc}(G,F)=\mathbb{E}_{x \sim p_{data}(x)}[\begin{Vmatrix} F(G(x))-x \end{Vmatrix}_1] + \mathbb{E}_{y \sim p_{data}(y)}[\begin{Vmatrix} G(F(y))-y \end{Vmatrix}_1]
\end{align}$$ 
<br>

***Figure 4***을 보면 reconstructed image $F(G(x))$는 original image와 매우 비슷하게 출력되었다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 4.png){: .align-center}


<br>
##### 3.3. Full Objective

최종 objective function은 다음과 같다.

<br>
$$ \begin{align} 
L(G,F,D_X,D_Y)=L_{GAN}(G,D_Y,X,Y)+L_{GAN}(F,D_X,Y,X)+\lambda L_{cyc}(G,F)
\end{align}$$ 
<br>

$\lambda$는 두 objective function의 상대적 중요도에 따라 control 된다.
이러한 최종적인 objective function은 아래와 같이 푸는것을 목적으로한다.

<br>
$$ \begin{align} 
G^{*},F^{*}=arg\underset{G,F}{min}\underset{D_X,D_Y}{max} L(G,F,D_X,D_Y)
\end{align}$$ 
<br>




<br><br><br>
#### <center>4. Implementation</center>

##### Network Architecture
CycleGAN에서는 neural style transfer and super-resolution에서 인상깊은 결과를 낸 architecture를 사용한다.
이러한 network는 두개의 stride-2 convolution, residual blocks, 두개의 stride $\frac{1}{2}$을 가진 fractionally-strided convolution를 포함한다. $128 \times 128$ image는 6개의 block을 사용하고 $256 \times 256$과 higher-resolution training image에는 9개의 block을 사용한다. 또한 Instance Normalization을 사용한다. discriminator에는 $70 \times 70$ PathGANs을 사용하고, $70 \times 70$ overlapping image patch가 real인지 fake인지 분류한다. 이러한 Patch-level discriminator architecture는 full-image discriminator보다 parameter수를 줄여주고, fully convolution에 의해 arbitrarily-sized images를 사용할수있다.

##### Training details
모델 학습에 안정성을 주기위해, 첫번째로, $L_{GAN}$을 negative log likelihood에서 least-squares loss로 대체한다. 이 loss는 좀 더 학습에 안정적이고, higher quality result로 생성해준다. 특히, GAN loss에서 <font size = 2>$L_{GAN}(G,D,X,Y)$</font>에서 <font size = 2>$\mathbb{E}_{x \sim p_{data}(x)}[(D(G(x))-1)^2]$</font>를 minimize하도록 $G$를 학습하고, <font size = 2>$\mathbb{E}_{y \sim p_{data}(y)}[(D(y)-1)^2] + \mathbb{E}_{x \sim p_{data}(x)}[D(G(x))^2]$</font>를 minimize하도록 $D$를 학습한다.<br>

두번째로, 이전에 사용한 generated image들을 discriminators에 update함으로써 model의 oscillation을 줄인다. 50개의 previously created image가 저장되도록 image replay buffer를 사용하여 유지한다. 아래의 그래프를 보면, 파란색 선을 기준으로 왼쪽 그래프는 대체로 진짜라고 판단하지만, random seed만 바꾼 오른쪽 그래프는 대체적으로 가짜라고 판단하고 있다. 이러한 문제는 discriminator를 여러개 생성하여 평균내는 방법이 있지만, 메모리를 너무 많이 잡아먹기 때문에, replay buffer를 사용하여 문제를 해결한다.<br> 
그리고, 경험적으로 ***Eq3***에서의 $\lambda = 10$으로 세팅했고, batch size 1, Adam optimizer를 사용한다. 모든 network는 learning rate 0.0002를 가지고 처음부터 학습된다. 처음 100 epoch까지는 동일하게 학습하고 100~200epoch까지는 0이 되도록 linearly decacy한다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/replay buffer.png){: .align-center}



<br><br><br>
#### <center>5. Results</center>

***Figure 6***은 위성이미지와 label된 이미지를 paired dataset으로 학습한 결과이다. ***Table 1,2,3***과 같이 여러 metric에 대해서도 CycleGAN은 이전 generation model에 비해 결과가 우수하다는것을 볼수있다. 
pix2pix의 경우는 이미 정답을 알고있는 paired image로 학습하여 CycleGAN보다 더 좋은 성능을 내지만, 정답이 없는 unpired image로 학습한 CycleGAN은 불리한 조건임에도 불구하고 그에 비교할만한 수준으로 성능이 나오고 있다.<br>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 6 Table 1,2,3.png){: .align-center}


***Figure 7***에서는 CycleGAN의 loss를 모두 사용했을때, 가장 좋다는것을 보여준다. ***Table 4***와 같이 GAN alone과 GAN+forward는 평가 지표상으로는 높을수 있으나 mode collapse 문제가 있고 모든 loss를 사용하였을때 original image와 가장 유사한 image가 나온다는것을 볼수있다. 그림상으로도 GAN alone과 GAN+forward에서 input은 다른데 output이 모두 똑같은 현상을 볼수있다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 7.png){: .align-center}
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Table 4,5.png){: .align-center}


또한, ***Figure 9***에서 보듯이, CycleGAN은 cycle translation 후 original image로 돌아와야하는데, 이때 그럴싸한 original image로 돌아오기만 하면 되기 때문에 색상 정보는 유지되지 않고 돌아온다. 만약 그림을 사진으로 변경할 때처럼 색상정보가 유지되어야 하는 task에서는 $L_{identity}$를 추가하여 학습을 진행하면, 색상정보가 유지된 결과를 얻을수 있게된다. i.e., <font size = 2> $ L_{identity}(G,F)=\mathbb{E}_{y \sim p_{data}(y)}[\begin{Vmatrix} G(y)-y \end{Vmatrix}_1]+\mathbb{E}_{x \sim p_{data}(x)}[\begin{Vmatrix} F(x)-x \end{Vmatrix}_1] $ </font>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 9.png){: .align-center}


CycleGAN은 ***Figure 10***과 같이 style transfer에도 사용될수 있다. 기존의 neural style transfer method와 다르게 선택된 하나의 style만이 아니라 전체 작품의 style을 모방하며, target domain과 유사한 결과 이미지를 생성하는것이 가능하다. 뿐만 아니라 ***Figure 13***과 같이 object transfigureation을 진행 할 수있고, 이외에도 season transfer, photo generation from paintings, photo enhancement 등 다양한 task를 수행할수 있다.
***Figure 14***와 같이 스마트폰으로 찍은 사진을 전문가가 찍은 DSLR 사진처럼 translation이 가능하다. 그림을 보면 photo enhancement를 진행하여 ouput focusing 기능을 CycleGAN을 통해 진행할 수 있다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 10.png){: .align-center}

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 13.png){: .align-center}

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 14.png){: .align-center}



<br><br><br>
#### <center>6. Limitations and Discussion</center>

CycleGAN의 Limitations도 존재하는데 ***Figure 17***과 같이 사과를 오렌지로 바꾸는 task의 경우 shape도 함께 바껴야한다. 하지만, CycleGAN의 경우 input image에서의 변화를 최소한으로 하려고 학습하기 때문에 shape를 변경하려는 task에 대해서는 좋지 않는 결과가 나오게된다. 그리고 오른쪽의 horse to zebra translation 같은경우에는 사람의 style 정보까지 zebra로 변하게 되었다. 이러한 training dataset에 포함되지 않은 object의 경우 잘못 translation 되는 경우도 있다.
이렇게 training dataset에 따라 학습 성능이 달라지며, shape 변경같은 content 정보까지 함께 변경이 필요한 task는 개선이 필요하다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-15-Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)/Figure 17.png){: .align-center}




</div>

