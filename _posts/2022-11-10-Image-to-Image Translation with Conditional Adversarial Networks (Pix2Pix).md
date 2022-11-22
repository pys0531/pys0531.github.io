---
title:  <font size="5">Pix2Pix</font>
excerpt: "Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)"
toc: true
toc_sticky: true
use_math: true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - GAN
  - Pix2Pix
last_modified_at: 2022-11-10T18:09:00-55:00
---

--------
**<font size="4">Pix2Pix 논문 리뷰</font>** 

>Paper : Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix) <https://arxiv.org/pdf/1611.07004.pdf>


<div markdown = "1">

#### <center>Pix2Pix</center>

Pix2Pix는 Conditional Adversarial Networks를 통한 image-to-image translation을 제안한다. 이러한 networks는 input image에서 output image로 mapping하는것 뿐만이 아니라, 이 mapping을 학습시키는 loss function도 학습한다. 


<br><br><br>
#### <center>1. Introduction</center>

Pix2Pix는 language translation처럼 image도 다른 image로 traslation될 수 있다고하며, 이를 _image-to-image translation_ 이라고 정의한다. 이러한 image translation을 위해 일반적인 framework를 제안한다.<br>
convolution neural networks(CNNs)은 image prediction 분야에서 많이 사용되고 있다. loss function을 최소화하는 방향으로 학습하며, 자동적으로 진행될지라도, loss를 효과적으로 디자인하는데 많은 노력이 필요하다. 다시말해, CNN에게 무엇을 최소화 할것인지 알려줘야한다. 
만약에 predict와 ground truth사이의 예측을 pixel Euclidean distance로 minimize 한다면, 이것은 blurry한 결과를 생성할 경향이 있을것이다. 왜냐하면, 있을법한 output을 평균하여 minimize하기 때문이다.
따라서, "현실과 구별할수 없게 한다"와 같은 높은 수준의 목표를 명시해주고, 이 목표를 달성하도록 적합한 loss function을 자동적으로 학습한다.  이러한 목표와 정확하게 부합하는것이 Generative Adversarial Networks (GANs)이고, 이를 통해 real과 fake를 분류하는 loss를 배우고, 동시에 generator loss를 minimize한다. 
<font size = 2><div markdown = "1">
> Euclidean distance 같이 의미가 명확하지 않은 loss function을 사용하여 blurry한 이미지가 나타났다면, "현실과 구별할수 없게한다"와 같은 높은 수준의 학습 목표를 주어 학습을 진행하면 blurry한 이미지는 나타나지 않을것이다.
</div></font>

GAN은 data에 맞게 loss를 학습하기 때문에, 다양한 task에도 적용될수 있다. 논문에서는 input image에 조건을 주어 대응하는 output이미지를 만들어내는 cGAN이 image-to-image translation task에 적합하기 때문에 cGAN을 사용한다. 여기서 주된 contribution은 매우 다양한 문제에서 conditional GAN이 합리적인 결과를 만들어낸다는 것을 설명하는것이다. 두번째 contribution은 좋은 결과를 내기에 충분한 simple framework를 제안하고, 몇 가지 중요한 architecture 선택의 효과를 분석하는 것이다. 

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Figure 1.png){: .align-center}
> application-specific algorithms 때문에 image translation에 많은 문제가 있다. cGAN은 위의 결과처럼 application-specific하지 않고 다양한 문제에 잘 작동한다. 


<br><br><br>
#### <center>2. Related work</center>

image-to-image translation problem은 per-pixel clasification or regression으로 표현된다. 이러한 형태는 각 pixel간에 서로 영향을 끼치지 않는 independent로 간주되고, 이를 output space가 "unstructured"하다고 한다. cGAN은 pixel간의 dependent를 고려하여 "structured loss"를 다룬다. (pixel간의 연결관계를 파악한다.) <br>
또한, architecture부분에서는 Generator는 "U-Net"-based architecture를 사용하고, Discriminator에서는 convolutional "PatchGAN" classifier을 사용한다. "PatchGAN"은 image patch에 대해 진위 여부를 파악하는 모델이고, patch size에 따른 영향을 알아본다.




<br><br><br>
#### <center>3. Method</center>
GANs은 random noise vector $z$에서 output image $y$로 mapping($G:z \to y$)하는 것을 배우는 generative model이다.  Conditional GANs은 observed image $x$와 random noise vector $z$에서 $y$로 mapping($G: \begin{Bmatrix} x,z \end{Bmatrix} \to y$)하는 것을 배운다. generator $G$는 생성된 output이 discriminator로부터 "real" image로 구별도록 학습되고, discriminator $D$는 generated image를 "fake"라고 구별하도록 학습된다. 이러한 training procedure은 아래의 그림과 같다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Figure 2.png){: .align-center}

##### 3.1. Objective

conditional GAN의 objective function는 다음과 같이 나타낸다.
<br><br>
$$ \begin{align} 
L_{cGAN}(G, D)=\mathbb{E}_{x,y} \begin{bmatrix} \log D(x,y) \end{bmatrix} + \mathbb{E}_{x,z} \begin{bmatrix} \log (1-D(x,G(x,z))) \end{bmatrix}
\end{align} $$
<br>
여기서 $G$는 적대적 $D$에 대해서 objective function를 minimize하려고 시도하고, $D$는 maximize하려고 시도한다. i.e. $$G^*=arg \underset{G}{min}\underset{D}{max} L_{cGAN}(G,D)$$ <br>
또한, GAN은 $L2$ distance와 같은 traditional loss와 함께 사용하면 좋다. Disctiminator에는 적용하지 않고, Generator가 ground truth에 가까와 지도록 추가 적용한다. $L1$과 $L2$ distance를 비교하였는데, $L1$ loss가 blurring에 더 효과가 있어서 $L1$을 사용하였다.
$$ \begin{align} 
L_{L1}(G)=\mathbb{E}_{x,y,z}[\begin{Vmatrix} y-G(x,y) \end{Vmatrix}_1]
\end{align} $$

그래서 최종 objective function은 아래와 같다. <br><br>
$$ \begin{align} 
G^*=arg \underset{G}{min}\underset{D}{max} L_{cGAN}(G,D)+ \lambda L_{L1}(G)
\end{align} $$
<br>

$z$없이 network는 mapping $x \to y$를 학습할수 있다. 그러나 delta function 이외의 다른 distribution에서는 match되지 않을것이다. noise $z$가 없기 때문에 deterministic한 output만 나온 것이다. <font size = 2>(즉, 다양성이 없어지는것이다, $z$를 같이 학습해도 $z$가 무시되는 쪽으로 학습이 됨.)</font> 최종 모델에서는 dropout을 train,test layer에 사용하여 noise를 주었지만, 조금의 stochasitic만 생겼다.
하지만 pix2pix에서는 image translation을 메인으로 다루기 때문에 있을법한 이미지를 만들어 내는데 의의를 두고, 이러한 deterministic한 문제는 future works로 남겨두었다.

##### 3.2 Network architectures

Generator와 Discriminator 둘 다 Convolution-BatchNorm-ReLu module을 사용한다. 아래에서 자세한 내용을 설명한다.

###### 3.2.1 Generator with skips

image-to-image translation problem은 high resolution input에서 high resolution output으로 매핑된다는 것이다. 또한, input과 output의 surface는 다르지만, 모양은 비슷해야 된다. 그러므로 이러한 문제를 고려하여 Generator architecture를 디자인한다.<br>

이전의 많은 문제들은 encoder-decoder network를 사용해왔다. 이러한 network에서, input은 downsampling을 통해 점진적으로 이미지 resolution을 줄여 feature를 뽑아내고, bottleneck layer에 도달하면, 반대의 절차로 이미지를 점진적으로 upsampling을 진행한다. 이와 같은 구조인 "U-Net"을 채택하여 사용한다.

> 또한, image translation 문제에서 input의 prominent edge는 output과 비슷해야한다. CNN의 low level info는 shape를 중점으로 학습하므로, 이를 skip connection을 통해 encoder와 decoder에 공유를 하면 성능이 올라간다. "U-Net"이 이와 같은 구조를 갖고있으며, 이 논문에서는 "U-Net"을 채택하여 사용한다.
low level에서는 receptive field가 좁기 때문에 local(디테일)한 정보를 담고 있고, 이러한 local한 정보를 output에 넘겨줌으로써 PatchGAN을 잘 활용할수 있고, high frequency(shape) 정보도 잘 학습할수 있게된다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Figure 3.png){: .align-center}


###### 3.2.2 Markovian discriminator (PatchGAN)

GAN의 경우 기본적으로 다른 generative model보다 blurry한 결과가 나오는 문제가 있다. $L1$을 사용했을때, Low frequency를 잘 잡아내므로 $L1$을 사용하여 새로운 framework를 제안한다. <font size = 2>($L2$보다 $L1$이 덜 blurry함.)</font> $L1$이 low frequency를 잘 잡아내기 때문에 high frequency만 신경쓰면 된다. pix2pix에서는 Discriminator에 _PatchGAN_을 통해 이를 해결한다. PatchGAN은 Full image에 대해서 real/fake를 판별하는것이 아닌 $N \times N$ patch 마다 real/fake를 판별한다. 이러한 이점은, 더 적은 parameter / 더 빠른 실행 / 더 큰 이미지에 적용할 수 있다. 

> Markovian discriminator에서 Markovian이라고 지은 이유는, Markovian은 이웃한 성분끼리의 영향은 있지만, 멀리 떨어진 성분과는 독립적이다. 그래서 PatchGAN에서 멀리 떨어진 Patch는 독립적이라 생각하여 Markovian discriminator라고 지은것 같다.

아래의 그림에서 봤을때, patchGAN에서 patch size가 $1 \times 1$는 매우 blurry하고 $70 \times 70$일 때 가장 성능이 좋아보인다. $286 \times 286$이 더 artifact가 심해보이는 이유는 학습 난이도가 더 높기 때문이라고 한다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Figure 6.png){: .align-center}

실제로 FCN-score를 확인해 보면 $70 \times 70$일 때 성능이 가장 높은것을 볼수있다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Table 3.png){: .align-center}

##### 3.3 Optimization and inference

pix2pix는 $D$와 $G$를 한번씩 gradient를 업데이트하고, original GAN paper의 objective function인 $log(1-D(x,G(x,z)))$를 minimize하는것이 아닌 $log(D(x,G(x,z)))$를 maximize하는 방향으로 학습을 진행한다. 또한, $D$를 $G$에 비해 학습 속도를 낮추기 위해, $D$를 optimization하는 동안 objective fuction을 2로 나눈다. minibatch SGD와 Adam optimization를 사용하고 learning rate는 0.0002, momentum paramerters $\beta_1=0.5$, $\beta_2 = 0.9999$를 사용한다.<br>

generator inference시에 training 방식과 정확히 같은 방식을 사용한다. test 할때에도 dropout을 적용시키며, batchNorm도 test batch의 statistics을 사용한다.




<br><br><br>
#### <center>4. Experiments</center>

아래의 그림은 기존의 Encoder-decoder 형식과 U-Net 구조를 사용했을때와 $L1$ loss와 $L1 + cGAN$을 사용했을때 차이를 보여준다.
$L1$만 사용했을때는 blurring이 심했고 $L1+cGAN$을 U-Net과 함께 사용했을때 가장 성능이 좋았다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Figure 5.png){: .align-center}

<br>

정량적인 평가를 위해 FCN-score를 사용하여 측정해보면, U-Net ($L1+cGAN$)이 가장 성능이 높게 측정된다.  
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Table 2.png){: .align-center}

<br>

또한 아래의 그림을 보면 확연한 차이를 보인다. $L1$일때는 눈에띄는 blurring이 존재하고, cGAN의 경우 shape하지만 visual artifact한 영향이 있다. 두개를 적절히 섞어서 사용한 경우에는 artifact가 적으면서도 완성도 높은 결과를 얻을수 있음을 보여주고있다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Figure 4.png){: .align-center}

<br>

Objective function에 대한 FCN-score를 확인해보면, 성능상으로도 $L1+cGAN$이 가장 성능이 뛰어난것을 볼수있다. 
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)
/Table 1.png){: .align-center}



<br><br><br>
#### <center>5. Conclusion</center>

pix2pix는 conditional adversarial networks가 image-to-image translation task에 유망한 접근법이라고 말하고 있다. 실제로 현재에도 많은 image-to-image translation의 baseline이 되고있고, 다양한 분야에서 적용될수 있다.

<br><br><br>

</div>

