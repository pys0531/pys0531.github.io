---
title:  <font size="5">StyleGAN</font>
excerpt: "A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)"
toc: true
toc_sticky: true
use_math: true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - GAN
  - StyleGAN
last_modified_at: 2022-11-22T18:09:00-55:00
---

--------
**<font size="4">StyleGAN 논문 리뷰</font>** 

>Paper : A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN) <https://arxiv.org/pdf/1812.04948.pdf>


<div markdown = "1">

#### <center>StyleGAN</center>

StyleGAN은 새로운 generator architecture을 제안한다. 이러한 archtecture는 high-level attributes의 unsupervised seperation / generated image의 stochastic variation을 배운다.

<font size = 2><div markdown = "1">
> **high-level attributes의 unsupervised seperation** : 사람의 얼굴의 pose나 identity / 여자인지 남자인지 / 안경을 썼는지 벗었는지 같은 큼직막한 feature를 서로 다른 style로 구분이 되고, 적용할수 있도록 학습한다.<br>
**generated image의 stochastic variation** : 서로 다른 style이 seperation되도록 학습을 하면, 그 style들은 매번 확률적으로 바뀔수 있다.
</div></font>

새로운 generator는 기존의 distribution quality metrics(FID)의 향상을 보여주고, 여러 image에 interpolation을 진행하였을때 더 나은 image가 나왔다. 또한, latent vector가 disentanglement 되어있어 학습에 도움을주었다.

<font size = 2><div markdown = "1">
> **latent vector의 disentanglement** : 각각의 feature가 얽혀있지(entanglement) 않고, 즉 의존하지 않고있어, 각각의 attribute를 잘 나타낼수있다.
</div></font>



<br><br><br><br>
#### <center>1. Introduction</center>

generative adversarial network(GAN)은 최근 급격한 발전을 보이고 있지만, generator는 stochastic feature와 같은 것들이 여전히 black box로 수행되고있다. 또한 latent space interpolation을 잘 설명할수 있는 정량적인 지표를 제공하지 못한다. <br>
이러한 문제점을 해결하기 위해 image synthesis process를 제어하기 위한 새로운 generator architecture 제안한다. generator는 learned constant input을 가지고 latent code(noise)를 기반으로 각 convolution layer의 image style을 조정한다. 이러한 noise와 결합하면, high-level의 attribute가 stochastic variation을 가지면서 잘 분리될수 있다. 또한 scale-specific mixing과 interpolation을 수행할수 있다. 이러한 작업은 discriminator의 구조나 loss function을 수정하지 않고 generator의 수정을 중점으로하여 작업한다.<br>
StyleGAN의 generator는 latent code가 intermediate latent space를 거쳐 input으로 들어가게 된다. 그러한 이유는, 기존 input latent space는 training data의 probability density를 따라야한다. 그래서 이러한 probability density는 어느정도의 entanglement를 갖는다. intermediate latent space는 gaussian/uniform같은 특정 distribution을 꼭 따라야한다는 restriction으로 부터 자유로워서 어느정도 disentanglement를 갖도록한다. 이러한 disentanglement를 측정하기 위한 지표로 perceptual path length / linear separability metrics를 제시한다. 이러한 metric을 사용하여 traditional generator architecture보다 representation이 더 선형적이고, 덜 entanglement하다는 것을 보여준다.

<font size = 2><div markdown = "1">
> 기존 GAN의 input latent space는 gaussian, normal, uniform등 다양한 distribution 중 선택하여 입력으로 들어갔다. 그렇게 되면은 내재된 attribute들이 입력된 distribution으로 entanglement(얽힘)를 갖게된다. 그러한 이유는 내재된 attribute들의 distribution이 억지로 gaussian과 uniform같은 latent space distribution으로 맞춰지기 때문이다. 이러한 entanglement를 없애기 위해 intermediate latent space를 거쳐 학습을 하게된다. ***Section 4***에서 설명
</div></font>

또한, 고해상도 Face Dataset인 Flickr-Faces-HQ / FFHQ Dataset을 제공하여 직접 학습에 이용해 볼 수 있다.




<br><br><br><br>
#### <center>2. Style-based generator</center>

Traditionally latent code는 input layer를 통해 입력된다.**<font size = 2>(Figure 1a)</font>** 하지만 여기서 input layer를 제거하고 learned constant를 사용한다.**<font size = 2>(Figure 1b, right)</font>** input latent space $$\mathcal{Z}$$에서 latent code $z$가 주어질때, non-linear mapping network $$f:\mathcal{Z} \to \mathcal{W}$$는 먼저 $$w \in \mathcal{W}$$를 만든다.**<font size = 2>(Figure 1b, left)</font>**

<font size = 2><div markdown = "1">
> 다시말해, $$\mathcal{W}$$는 intermediate latent space를 말하고, 이는 latent code $z$를 disentanglement하기 위해 만들어진다.
</div></font>

$$f:\mathcal{Z} \to \mathcal{W}$$는 8-layer MLP사용하여 진행되며, 두개의 latent vector $z$와 $w$는 512 dimension 갖고 $$f$$로 매핑된다. <font size = 2>(Section 4.1에서 설명)</font> 학습된 affine transformation***("A")***은, adaptive instance normalizetion**(AdaIN)**을 통해 style을 제어하는 parameter가 되고, 그 parameter는 $$y=(y_s,y_b)$$를 갖고 $$w$$를 transformation한다. AdaIN은 synthesis network $g$의 각 convolution layer 이후에 적용되고, 아래와 같이 정의된다.
<br>
$$ \begin{align} 
\textrm{AdaIN}(x_i,y)=y_{s,i} \frac{x_i-\mu(x_i)}{\sigma(x_i)}+y_{b,i}
\end{align}$$ 
<br>

<font size = 2><div markdown = "1">
> AdaIN은 Batch Normalization에서 scale과 bias를 학습하는 affine transformation과 비슷한 방식으로 작동한다. Instance Normalization에서 정규화 후 $y_{s,i}$, $y_{b,i}$를 통해 scale과 bias를 적용시켜 style을 제어하도록 학습이 진행된다.
</div></font>

여기서 각 feature map $x_i$가 정규화된 후, style $y$에 해당하는 scalar components를 사용하여 scale과 bias를 적용시킨다.

마지막으로 noise input***("B")***을 이용하여 stochastic detail을 생성해낸다. 이러한 noise는 image형태로 convolution의 size와 비례하는 형태로 각 layer마다 들어간다. 이러한 결과는 ***Section 3.2 / 3.3***에서 논의된다.



![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 1.png){: .align-center}

<font size = 2><div markdown = "1">
> ***Left :*** baseline인 PGGAN의 architecture를 나타내고있음. latent vector $z$가 input layer로 바로들어감 <br>
***right :*** StyleGAN architecture를 나타냄. latent vector $z$가 Mapping network를 거쳐서 input layer로 들어감. 변형된 intermediate latent vector $w$는 별도의 affine trasformation(***"A"***)을 거쳐 convolution network에 들어간다. 이때 latent vector $z$와 $w$의 크기는 같다. 
또한, Synthesis network $g$는 convolution block을 거칠수록 resolution이 두배가 되므로 $$4 \times 4 \to 1024 \times 1024$$가 되기 위해서는 9개의 block이 필요하다. 각 block들마다 2번의 style vector(intermediate latent vector($w$))가 들어오므로 총 18개의 latent vector가 필요하고, latent vector의 총 크기는 $$18 \times 512$$크기의 vector가 된다.<br>
***"B"***는 stochastic variation과 같은 다양성을 주기위해 noise를 추가하여 넣어준다.<br><br>
정리하면,
1. learned constant input을 통해 
2. stochastic variation을 위한 noise(***"B"***)를 넣어주고 
3. AdaIN에서 statistics한 style을 변경해주기 위해 affine transformation(***"A"***) 된 latent vector를 넣어주고 
4. convolution을 거쳐 output을 내게 된다. 그리고 network의 마지막 layer에는 RGB 이미지가 나오도록 $1 \times 1$ convolution을 통해 3 dimension으로 만들어준다.
</div></font>


##### 2.1. Quality of generated images

***Table 1***에서는 논문에서 제시하는 다양한 method를 사용하였을때, Baseline Network에 비해 성능이 높아진다는 것을 볼수있다. ***Table 1***에서는 Frechet inception distances(FID)를 통해 CELEB A-HQ와 FFHQ의 성능을 비교하여 보여준다. <br>
(A): Baseline architecture인 Progressive GAN과 비교한 성능이고 원본 논문과 유사한 parameter를 사용 <br>
(B): bilinear up/down sampling operation / 더 긴 학습 / hyperparameter tunning을 통해 성능을 올림 <br>
(C): mapping network / AdaIN operation을 추가하여 성능을 올림 <br>
(D): $$4 \times 4 \times 512$$ learned constant tensor을 input으로 사용하여 성능을 올림. 각 layer 마다 noise(latent vector)가 추가되므로, 굳이 input이 noise(latent vector)가 아니여도 됨 <br>
(E): noise input을 추가하여 결과를 향상시키고 <br>
(F): mixing regularization을 통해 이웃 style을 decorrelation시키고 생성된 이미지를 좀더 세밀하게 제어할수 있게함 <br>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Table 1.png){: .align-center}

이러한 method는 CELEB A-HQ의 경우 WGAN-GP loss를 사용하여 얻어진 결과이며, FFHQ는 (A)에서 WGAN-GP loss를 사용하고 (B)-(F)까지는 $R_1$ regularization을 사용한 결과이다. style-based generator (E)는 traditional generator (B)에 비해 FID가 20% 정도 향상된 것을 볼 수 있다. ***Figure 2***는 StyleGAN generator에서 FFHQ dataset을 사용하여 생성한 uncurated image(선별되지 않은 이미지)를 나타낸다. 잘 나온것을 뽑은것이 아닌 평균적으로 나오는 이미지들이다. 심지어 썬글라스나 모자와 같은 악세사리들고 성공적으로 합성되었다. 그리고 truncation trick이라고 불리는 기법을 사용하여 $\mathcal{W}$가 distribution의 극단적인 부분에서 sampling되는것을 막았다. truncation trick은 high resolution에서는 영향을 받지 않도록 low resolution에서만 적용할수도 있고, FID에서는 truncation trick을 사용하지 않았다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 2.png){: .align-center}

##### 2.2. Prior art

이전의 논문들은 discriminator를 중심으로 연구를 진행하였다. 하지만 몇몇 논문은 multiple generator layer에 latent code를 넣는 generator를 중심으로 진행하기도 하였다. 또한 AdaIN을 사용하여 연구한 논문도 있었지만, StyleGAN과 같이 intermedia latent space나 noise input을 진행한 연구는 없었다.





<br><br><br><br>
#### <center>3. Properties of the style-based generator</center>

StyleGAN의 generator architecture는 scale-specific modification을 통해 style을 변경한다. 이러한 style은 training dataset의 수집된 style을 기반으로 새로운 image를 생성하게 된다. 즉, layer의 특정 부분에 대한 style을 바꿈으로써, 특정한 aspect를 바꿀수있다. 이러한 특정 부분의 style을 바꾸는것을 localization(지역화)이라한다. AdaIN operation은 먼저 각 channel normalization하고 scale과 bias를 적용하여 style을 바꾼다. 매번 convolution operation을 지나서 normalization을 진행하기 때문에 original statistics에 의존하지 않고, 그러므로 각 style은 하나의 convolution에 의해 적용되고 AdaIN operation에 의해 덮어씌어진다.



##### 3.1. Style mixing

Style mixing은 stlye이 더 localization되도록 하기위해 제안되었다. mixing regularization(Style mixing)은 두개의 random latent vector가 있을때, 하나의 latent vector만 사용하는것이 아닌, 두개의 latent vector를 섞어서 학습에 사용하는것을 의미한다. 구체적으로, mapping network를 통해 두개의 latent code $z_1$, $z_2$를 실행하고, 대응하는 $w_1$, $w_2$에서 crossover point 이전에는 $w_1$을 사용하고 이후에는 $w_2$를 사용하여 style을 제어할수 있게한다. 이러한 regularization technique는 인접한 style이 연관되어 있다고 추정하는것을 막는다. (=> 인접한 style을 연관되어 추정하지 않기 때문에 각 style이 잘 구분이 되게 추정하도록 한다.)

<font size = 2><div markdown = "1">
> <font size = 2>다른 layer에는 관여하지 않도록 하기위해서(=> localization 되도록) style mixing을 사용함</font>
</div></font>

***Table 2***는 training 중에 mixing regularization하는것이 FID에 어떠한 영향을 끼치는지 나타낸다. 표를 보면 1~4개의 latent space를 가지고 crossover point를 기점으로 나눠서 사용한 결과를 나타내고 있다. mixing regularization을 많이 사용할수록 adverse operations에 내성이 생겨 높은 성능을 보이고있다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Table 2.png){: .align-center}
<br><br>


***Figure 3***의 경우는 여러개의 image사이에서 style mixing을 사용한 결과이다. interpolation이 아닌 두개의 latent code를 다양한 scale에서 mixing한 합성 image이며, 각 subset에 대하여 의미있는 high level attribute의 style control을 하고있다.<br>
첫번째 행과 열은, 각각의 latent code(source A and B)에 따라 생성된 image이고, 나머지 가운데 image들은 source B style의 subset을 가져오고, source A style의 나머지 subset을 합쳐 생성된 image이다. AdaIN에서 $18 \times 512$의 $w$ latent vector가 들어오고 resolution에 해당하는 style마다 각각의 특징을 가지고 있다.
1. B로 부터 coarse spatial resolution($4^2 - 8^2$)의 style은 source B로 부터 pose / general hair style / face shape / eyeglasses와 같은 high-level의 aspect를 가져오고, all colors(eyes, hair, lighting) / finer facial features는 source A로 부터 가져온다. <font size = 2>(18개의 latent vector중에 0-3과 network의 앞쪽 0-1 layer와 대응)</font>
2. B로 부터 middle resolution($16^2 - 32^2$)를 가져온다면, B로부터 hair style / eyes open/closed와 같은 더작은 scale의 facial feature를 가져오고, pose / general face shape / eyeglasses같은 style을 A로 부터 유지한다. <font size = 2>(18개의 latent vector중에 4-7과 network의 중간 2-3 layer와 대응)</font>
3. B로 부터 fine style (64^2-1024^2)를 가져온다면, color scheme / microstructure style을 주로 가져온다. <font size = 2>(18개의 latent vector중에 8-17과 network의 뒷쪽 4-8 layer와 대응)</font>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 3.png){: .align-center}


##### 3.2. Stochastic variation

human facial은 stochastic variation이 있다. 예를들어 머리카락의 위치 / 수염 / 주근깨 / 피부모공과 같은 것들은 언제나 바뀔수 있다. <br><br>
***Figure 4***를 보면,<br>
(a) : Network에 의해 생성된 image를 보여준다. <br>
(b) : style vector가 같은 상태에서 noise를 주어 머리카락 같은 미세한 변화를 준다. 즉, 확률적으로 미세한 정보들이 바뀌는것이다.<br>
(c) : 100개 이상의 서로다른 Stochastic variation의 Standard deviation을 보여준다. 그림을 보면 머리카락과 같은 미세한 정보들이 많이 바뀌는것을 볼수있다. 또한, noise vector는 매 실행마다 바뀌므로, 같은 style이라도 실행할때 마다 결과가 미세하게 바뀐다.<br>

***Figure 5***같은 경우에는, <br>
(a) : 모든 layer에 noise를 적용하여 생성된 image이다. (original StyleGAN)<br>
(b) : noise를 적용하지 않는 결과, 머리카락과 같은 스타일에 디테일이 떨어지는것을 보인다.<br>
(c) : fine layer에 style을 적용한 결과 => 세밀한 헤어스타일 컬링 / background detail feature / skin pores를 야기한다.<br>
(d) : corse layer에 style을 적용한 결과 => large-scale의 헤어스타일 컬링 / 더 큰 background feature를 야기한다. <br>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 4,5.png){: .align-center}


##### 3.3. Separation of global effects from stochasticity

이렇게 detail한 stochastic variation은 noise에 대한 영향을 받지만, high-level aspect는 style에 의해 control이 가능하고 noise로는 control이 불가능하다.




<br><br><br><br>
#### <center>4. Disentanglement studies</center>

disentanglement는 latent space가 linear subspace갖도록 하는것을 말한다. linear subspace를 갖고있다면, latent space에 존재하는 두개의 vector에 대해 interpolation을 수행했을때, 의도했던 feature만 바뀔 가능성이 높다. 즉, 개별적인 control이 특정한 factor만 바꿀수 있다면, 여러개의 특징들이 잘 분리되어 있다고 말할수 있다.<br>
***Figure 6***에서 보듯이, <br>
(a) : training set의 distribution을 나타낸다. distribution의 가로축이 남자 / 여자, 세로축이 머리가 긴 / 짧은 으로 나타낸다면, 비어있는 왼쪽 윗부분은 머리가 긴 남자의 영역이 될것이다. 즉, 머리가 긴 남자의 feature가 없는 distribution을 나타낸다.<br>
(b) : (a)의 distribution을 typical input latent space로 mapping을 하면 (b)와 같이 feature들이 왜곡이된다. typical input latent distribution은 보통 gaussian / uniform과 같은 distribution을 갖으므로, 이러한 distribution에 강제로 mapping한다면 다양한 feature들이 왜곡되고, 각각의 feature에 대해 interpolation을 수행하면 image의 sementic한 정보가 많이 바뀔수있다. 즉, typical input latent distribution는 entanglement하여 interpolation을 수행하였을때, 왜곡된 distribution에 영향을 받아 의도한 featrue 외에도 다른 feature에 영향을 받을수 있다.<br>
(c) : 하지만, intermediate latent space $$\mathcal{W}$$를 사용하여 고정된 distribution에 따른 sampling을 하지않고, learned mapping $f(z)$를 사용하므로써 $$\mathcal{W}$$에 대한 "unwarp"(왜곡되지 않은, 왜곡됨/비틀어짐을 바로 잡는다.)을 적용한다면, 여러 feature들이 disentanglement해져, 다양한 feature들이 linear한 subspace를 갖게한다. <br>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 6.png){: .align-center}

이러한 intermediate latent space $$\mathcal{W}$$가 실제로 disentanglement한지 알아보기 위해 성과지표를 제공한다. 기존의 latent vector가 input으로 들어가는 metric의 경우 StyleGAN에서 사용할수 없으므로 두가지 새로운 metric을 제안한다. 



##### 4.1. Perceptual path length

위에서 말한것과 같이 latent space가 왜곡되어 mapping된다면, linear하게 interpolation되기 힘들다. 즉, latent space는 entangle되어있고, 다양한 factor들이 linear하게 구분될수 있지 않다는 것이다. 이러한것을 정량적으로 나타내기 위해, image를 interpolation할때 drastic한 변화를 측정한다. latent space가 덜 curve되있다면, interpolation이 더 smooth하게 translation 될것이다. <br>
방식은 두개의 VGG16 분류 networks를 가지고 feature값을 비교하여 계산한다. feature값은 인간이 인식하기 비슷한정도로 조절하여 측정한다. 즉, 두개의 latent space를 VGG16에 넣어 embedding 시키고, 두 embedding vector간의 차이를 계산한다. 방식은 어떠한 지점 $t$에서의 interpolation과 아주 작은 subdivision인 $\epsilon$만큼의 지점에서의 sampling된 perceptual path length를 계산하여 두 이미지가 얼마나 feature상에서 변화가 있었는지 판단할수 있다. 이러한 계산은 아래와 같이 진행되며, latent space $\mathcal{Z}$와 intermediate latent space $\mathcal{W}$은 약간 다르게 계산된다.<br>

$\mathcal{Z}$의 경우 gaussian distribution을 따르기 때문에 slerp(spherical interpolation operation)을 진행한다.
<br>
$$ \begin{align} 
l_{\mathcal{Z}}=\mathbb{E}\begin{bmatrix}
\frac{1}{\epsilon^2}d \begin{pmatrix} G(\textrm{slerp}(z_1,z_2;t)),G(\textrm{slerp}(z_1,z_2;t+\epsilon)) \end{pmatrix}
\end{bmatrix}
\end{align}$$ 
<br>
여기서 $z_1,z_2 \sim P(z)$, $t \sim U(0,1)$, $G$는 style-based network $g \circ f$ generator, $$d(\cdot,\cdot)$$는 결과 이미지 사이의 perceptual distance이다.

$\mathcal{W}$의 경우 유사한 방식으로 진행된다. $\mathcal{Z}$와 계산상 차이점은 $\mathcal{W}$는 mapping fuction을 거쳐 normalize된 상태가아니고, 각 factor들이 선형적으로 분리되었다는 가정이 있기 때문에 slerp가 아닌 lerp를 사용하여 계산한다.
<br>
$$ \begin{align} 
l_{\mathcal{W}}=\mathbb{E}\begin{bmatrix}
\frac{1}{\epsilon^2}d \begin{pmatrix} g(\textrm{lerp}(f(z_1),f(z_2);t)),g(\textrm{lerp}(f(z_1),f(z_2);t+\epsilon)) \end{pmatrix}
\end{bmatrix}
\end{align}$$ 
<br>


![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Table 3.png){: .align-center}

***Table 3***에서 보는것과 같이, latent space $\mathcal{W}$가 $\mathcal{Z}$보다 더 perceptual적으로 linear하다는것을 볼수있다. 사실 이러한 측정은 latent space $\mathcal{Z}$에 유리하도록 약간 bias되어있다. 만약 $\mathcal{W}$가 정말로 disentangle되어있고 "flattened" mapping이라면, 이것은 아래의 그림과 같이 input manifold에 없는 공간을 포함할것이다. 이러한 결과는 generator에 의해 매우 나쁜 이미지가 생성될것이다. 반면에, latent space $\mathcal{Z}$는 gaussian으로 mapping되기 때문에 그러한 영역이 없다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/interpolation.png){: .align-center}

그러므로 endpoints 지점을 제한하여 측정한다면(i.e., $t \in \begin{Bmatrix} 0,1 \end{Bmatrix}$), 더 작은 $l_{\mathcal{W}}$를 얻어야하고, ***Table 3***에서 보는것과 같이 더 작은 결과를 얻었다.<br>

<font size = 2><div markdown = "1">
> ***Table 3***에서 $\mathcal{W}$가 $\mathcal{Z}$보다 더 낮은 값을 얻어 image의 급격한 변화가 적다는것을 확인했고, 이러한 급격한 변화가 적다는것은 latent space가 왜곡되어 mapping되지 않고, interpolation시에 더 linear한 수행을 했다고 볼수있다. <br>
또한, style mixing을 사용할경우 각각의 layer의 style 정보들을 uncorrelate 되도록 함으로, perceptual path length 측면에서는 왜곡을 일으키는 요인으로 볼수있다. 그래서 style mixing을 사용할 경우 더 낮은 성능을 보이게 된다.
</div></font>


![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Table 4.png){: .align-center}
***Table 4***에서는 path length가 mapping network에 어떠한 영향을 받는지 보여준다. mapping network의 depth가 많을수록 성능이 높다는것을 볼수있다. 또한, $\mathcal{W}$가 전반적으로 $\mathcal{Z}$보다 path length와 FID 모두 좋다는것을 보여준다. 즉, $\mathcal{W}$는 개선되는 반면, $\mathcal{Z}$는 상당히 악화되어 GAN에서 input latent space는 entangled되어있다고 주장할수있다.
 

##### 4.2. Linear separability

만약, latent space가 충분히 disentangle하다면, 각각의 variation factor에 해당하는 direction vector를 찾을수 있어야한다. 때문에 latent space point가 linear hyperplane에 의해 두개의 distinct set로 얼마나 잘 분리될수있는지 측정할수 있는 또 다른 metric을 제안한다.<br>
이를위해 CELEB A-HQ dataset을 가지고 binary attribute를 갖는 auxiliary classification networks를 학습한다. CELEB A-HQ dataset은 40개의 attribution를 갖는다. <font size = 2>(남자인지 여자인지, 웃고있는지 안웃고있는지 같은 binary attribute)</font> 그 후, $z \sim P(z)$를 sampling하는 generator를 이용해 200,000개의 image를 생성하고 학습했던 auxiliary classification network를 통해 분류를 진행한다. 그런다음 가장 낮은 confidence를 갖는 100,000개의 sample을 제거하고, 높은 confidence를 갖는 이미지만 남겨놓는다.<br>

<font size = 2><div markdown = "1">
> latent space가 얼마나 잘 disentangle 되었는지, latent space의 linear classification network를 학습하는것이 목표이기 때문에 confidence가 가장 높은 sample만 남겨놓고 이를 새로운 dataset으로 활용하여 linear classification network를 학습시킨다.
</div></font>

각 attribute에 대해, latent space point를 학습할 수 있는 linear SVM을 학습한다. 이를 통해 측정지표인 conditional entropy $H(Y \mid X)$를 구한다. 여기서 $X$는 SVM에 의해 예측된 class이고 $Y$는 pre-trained classifier에 의해 측정된 class이다. 이것은 sample을 true class로 결정하는데 얼마나 많은 추가적인 정보가 필요한지 말해준다. 즉, $H(Y \mid X)$는 entropy 값으로 높으면 latent vector를 잘 분류할수 없고, linear하게 분포해있지 않다는 것이다. entropy가 낮으면 일관된 latent space direction을 제시한다.

<font size = 2><div markdown = "1">
> entropy는 불확실성에 대한 척도이다. 그러므로 entropy가 높다는것은 불확실성이 높다는것이고, 위에서 봤듯이 conditional entropy가 높다는것은 latent vector를 선형적으로 잘 분류하지 못한다는것이다. 
</div></font>

최종 separability score는 $$\exp \begin{pmatrix} \sum_{i} H(Y_i \mid X_i) \end{pmatrix}$$로 나타낸다. 여기서 $i$는 40개의 atribute를 나다낸다.<br>
***Table 3,4***에서 보듯이 $\mathcal{W}$는 $\mathcal{Z}$보다 덜 entangle되어 매우 일관되게 분리된다는 것을 볼 수 있다. separability를 봐도 $\mathcal{W}$가 성능이 더 좋은것을 볼수있다. 이는 interpolation과정에서 중간 image들이 그럴싸하다는 것을 뜻한다. 또한, traditional generator and style base generator 둘다 mapping network를 사용하였을때 훨씬 더 linear한 latent의 subspace를 갖을수 있도록 한다.




<br><br><br><br>
#### <center>5. Conclusion</center>

traditional GAN architecture는 모든면에서 style-based design 보다 낮은 성능을 보이고 있다. 또한, high-level attribute들이 잘 분리되고 stochastic variation 제어가 가능하고 또한 intermediate latent space에서 linearity는 GAN synthesis를 이해하고 제어하는데 큰 도움이 될것이라고 생각한다.<br>
average path length metric은 학습 중에 regularizer로 사용될수 있고, 아마 linear separability metric 또한 가능할것이다.



<br><br><br><br>
#### <center>6. Appendix</center>

<br>
##### A. The FFHQ dataset

1. 1024 resolution를 갖는 70,000장의 high-quality image dataset인 FFHQ를 제공함.
2. CELEB A-HQ 보다 나이와 지역성면에서 더 많은 variation을 갖고 있으며, 선글라스 유무/성별 등의 특징요소 범위도 더 많다.
3. 이러한 이미지들은 Flickr website에서 크롤링하여 라이선스가 허용된 image에서 자동적으로 align하고 crop함
4. <https://github.com/NVlabs/ffhq-dataset>에서 FFHQ dataset을 이용가능함.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 7.png){: .align-center}


<br>
##### B. Truncation trick in $\mathcal{W}$

training dataset distribution에서 low density 부분에서 sampling을 진행하면 덜 그럴싸한 image들이 생성되므로, 평균 image quality가 향상되는 방향으로 latent space $\mathcal{W}$를 truncation한다. 즉, $$\overline{w}=\mathbb{E}_{z \sim P(z)}[f(z)]$$로 $\mathcal{W}$의 중간값을 구한뒤, $${w}' = \overline{w}+\psi(w-\overline{w})$$로 $w$의 중심으로부터 deviation의 scale을 조정할 수 있다. (여기서 $\psi<1$.)

<font size = 2><div markdown = "1">
> 즉, low density 부분은 덜 그럴싸한 image들이 나오므로, $w$의 중간값을 구한 뒤 일정범위만큼 잘라내겠다는 것이다.
$$\overline{w}=\mathbb{E}_{z \sim P(z)}[f(z)]$$는 $z \sim P(z)$로 sampling한 기댓값을 구해 중간지점을 찾아내고
$${w}' = \overline{w}+\psi(w-\overline{w})$$에서 $\psi$값을 조절해 truncation을 진행해 준다.
</div></font>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 8.png){: .align-center}

***Figure 8***에서 보듯이 $\psi=0$으로 truncation을 진행하면 평균값에서 sampling된 동일한 평균적인 이미지만 나오게되고, $\psi=1$로 진행하게 되면 truncation을 진행하지 않는다는 의미로 더 다양한 image가 생성될수있다.


<br>
##### C. Hyperparameters and training details

1. 여기서는 detail한 hyperparam과 training 방법을 설명하고, Style GANs networks는 Progressive GANs의 training detail과 거의 비슷하게 작동

2. Progressive GANs과 같은 discriminator architecture / resolution에 따른 minibatch size / Adam hyperparam / exponential moving average를 사용

3. CelebA-HQ, FFHQ에서는 mirror augmentation을 진행하지만, LSUN에서는 사용하지 않는다. 또한, Tesla V100 GPU를 사용하여 일주일 정도 학습하였다. 

<font size = 2><div markdown = "1">
> ***A in Table 1***와 같음
</div></font>

4. bilinear sampling 대신에 nearest-neighbor up/downsampling으로 대체하고, 각 upsampling layer 후에 그리고 downsampling layer 전에 separable 2차 binomial filter로 activation을 lowpass filtering하여 실행
5. progressive growing 학습에서는 $4^2$ 대신 $8^2$부터 시작
6. FFHQ dataset에는 WGAN-GP loss로 변경하고 $\gamma=10$을 갖는 $R_1$ regularization 함께사용하여 non-saturating loss를 갖게함 
7. $R_1$은 WGAN-GP보다 FID score를 상당히 오랫동안 감소한다는것을 발견했고, 따라서 12M에서 25M image로 늘려 training 시간을 늘림 
8. Progressive GAN과 동일한 learning rate, CelebA-HQ $512^2$ and $1024^3$에서는 0.003대신 0.002를 사용하는것이 더 안정적

<font size = 2><div markdown = "1">
> ***B in Table 1***와 같음
</div></font>

9. style-based generator는 leaky ReLU($\alpha = 0.2$) / 모든 layer에 Progressive GANs에서 사용된 equalized learning rate 사용
10. mapping network는 동일한 8 fully-connected layers / latent vector $z$와 $w$는 모두 동일한 512 dimension을 갖음
11. mapping network의 dimension이 증가할수록 high learning rate에 대해 학습이 불안정해진다. 그러므로 mapping network의 learning rate를 $10^2$정도 줄임 (i.e., $\lambda' = 0.01 \cdot \lambda$)
12. convolutional, fully-connected, affine transform layers의 모든 weights를 $N(0,1)$로 초기화 / synthsis network의 constant input은 ones(1)으로 초기화 / biases and noise scaling factors는 zeros(0)으로 초기화
13. 모든 network에서는 batch normalization / spectral normalization / attention mechanisms / dropout / pixelwise feature vector normalization를 사용하지 않는다.

<font size = 2><div markdown = "1">
> ***F in Table 1***와 같음
</div></font>



<br>
##### D. Training convergence

***Figure 9***는 ***Table 1***의 (B), (F)가 얼마나 성능이 차이나는지 FID와 perceptual path length metric를 통해 보여준다. 그림에서 보듯이 $1024^2$ resolution에 도달했을때, 더 학습을 하게되면 path length는 천천히 증가하고 FID는 개선된다. 즉, FID가 개선됨으로 더 그럴싸한 image들이 나오지만, path length가 커짐으로 linear sperable한 특징들은 더 entangle하게 된다. 다시말해 FID와 path length가 trade off 관계를 보여주고 있다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 9.png){: .align-center}



<br>
##### E. Other datasets

또한 다른 데이터셋에 대해서도 비슷한 효과를 얻을수 있다. 예를들어 LSUN bedroom dataset의 경우 corse style을 변경하면 카메라의 구도가 바뀌고, middle style을 변경하면 특정 가구들이 바뀌고, fine stlye을 변경하면 색생과 같은 detail한 정보들이 변경된다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-22-A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)/Figure 10,11,12.png){: .align-center}



</div>

