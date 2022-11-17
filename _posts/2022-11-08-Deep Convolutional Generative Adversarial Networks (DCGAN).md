---
title:  <font size="5">DCGAN</font>
excerpt: "Deep Convolutional Generative Adversarial Networks (DCGAN)"
toc: true
toc_sticky: true
use_math: true
categories:
  - Machine Learning
tags:
  - Machine Learning
  - GAN
  - DCGAN
last_modified_at: 2022-11-08T18:09:00-55:00
---

--------
**<font size="4">DCGAN 논문 리뷰</font>** 

>Paper : Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN) <https://arxiv.org/pdf/1511.06434.pdf> <br>

<div markdown = "1">

#### <center>DCGAN</center>

DCGAN은 Deep Convolutional Generative Adversarial Networks의 약자로, 기존의 GAN에 Convolutional Network를 적용한 형태이다.<br>
기존의 GAN은 unstable하고 학습한 내용과 중간계층을 시각화하기에 제한적이였다. 그래서 DCGAN에서는 아래와 같은 내용을 포함한다.<br>
- Convolutional GAN의 architectural topology의 제약을 제안하고 평가하여 학습에 안정성을 준다. 이러한 architecture를 Deep Convolutional GANs (DCGAN)이라고 부른다.
- image classification task가 학습된 Discriminator는 다른 unsupervised algorithms과 대등한 성능을 보인다.
- GAN에 의해 학습된 필터를 시각화하고 특정 filter가 특정 object를 그리는 방법을 배운다는것을 확인했다.
- Generator는 generated sample의 많은 semantic qualities를 쉽게 조작 할수있는 vector arithmetic 특성을 갖는다. 

<br><br>
##### 1. Model Architecture

DCGAN 이전에도 CNN에 대한 GAN 적용이 몇몇이었지만 성공적으로 이끌지 못했다. 하지만 DCGAN에서는 그당시에 최근 입증된 CNN architecture 특징 3가지를 적용하여 문제점을 해결했다.

###### 1.1. Strided Convolution
먼저 CNN에서는 미분 불가능한 maxpooling을 strided convolution으로 변경 함으로써, network가 spatial downsampling을 학습할수 있도록 한다. 이러한 기법을 Generator에 적용하여 spatial upsampling / Discriminator를 학습할수 있도록 한다. 즉, Generator에서는 Upsample이나 Transposed Convolution를 사용하여 학습한다.

###### 1.2. Eliminate Fully-Connected Layers
두번째는, 트렌드에 맞게 Fully-Connected Layer를 제거하고 convolution으로 교체하였다.
global average pooling는 모델를 안정적으로 만들지만, 수렴속도를 헤친다. 그래서 Generator와 Discriminator의 input과 output에 global average pooling를 사용하지 않고 convolution으로만 연결하여 학습한다.
Uniform noise distribution $Z$는 단순한 행렬 곱이기 때문에 Fully Connected라고 불릴수 있지만, 4차원 tensor로 reshape되어 convolution과 연산하여 사용된다. Discriminator 모델은 last convolution layer를 flatten하여 single sigmoid output으로 한다.


###### 1.3. Batch Normalization
Batch Normalization은 각 unit에 대한 input을 평균을 0, 분산을 1로 nomalization하여 안정적으로 학습하게 한다. 이것은 최적화되지 않은 initialization과 깊은 model에서 gradient가 잘 흐르도록 도와준다. 또한 기존 GAN에서 공통적으로 관찰되는 문제인, 모든 sample들이 single point로 붕괴하는 것을 예방해준다. 하지만 Batchnorm을 모든 layer에 적용하면, sample oscillation 과 model instability를 초래했다. 그래서 Generator의 ouput 부분과 Discriminator의 input 부분에는 Batchnorm을 적용시키지 않았다.<br>

또한, Activation function을 변경하였는데, Generator의 ouput layer에서는 Tanh를 사용하고 나머지는 ReLU activation을 사용하였고, Discriminator에서는 LeakyReLU를 사용하였다.
<font size = 3><div class="notice" markdown="1">
***안정된 Deep Convolutional GANs을 위한 Architecture guidelines***<br>
- 모든 pooling layer를 strided convolution(discriminator), fractional-strided convolution(generator)로 교체한다.
- Generator와 Discriminator에 batchnorm을 사용한다.
- 더 깊은 architecture를 위해 fully connected hidden layer를 제거한다.
- Generator는 ouput layer에는 Tanh를 사용하고 나머지는 ReLU activation function을 사용한다.
- Discriminator의 모든 layer는 LeakyReLU activation function을 사용한다.
</div></font>


<br><br>
##### 2. DETAILS OF ADVERSARIAL TRAINING

- Tanh activation function 스케일인 [-1, 1]로 scaling하는것 이외에는 pre-processing이 적용되지 않음
- mini-batch stochastic gradient descent(SGD) / mini-batch 128로 학습
- zero-centered / std = 0.02 인 Normal distribution으로 weight 초기화
- LeakyReLU의 slope는 0.2
- Adam Optimizer
- learning rate = 0.0002 <font size = 2>(기존 GAN의 lr = 0.001은 너무 높다고 판단)</font>
- momentum term $\beta_{1}$에 제안된 값 0.9는 training oscillation과 instability를 야기하기 때문에, 학습의 stability를 돕기위해 0.5로 줄임

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-08-Deep Convolutional Generative Adversarial Networks (DCGAN)
/Figure 1.png){: .align-center}
<font size = 2><div markdown = "1">
***Figure 1:*** 100 dimensional uniform distribution $Z$는 작은 spatial extent convolutional representation으로 projection된다. 4개의 fractionally-strided convolutions(= deconvolution이 아니라 transposed convolution임)은 $64 \times 64$ pixel image로 변환된다. 특히, fully connected or pooling layer는 사용하지 않는다.

> deconvolution은 convolution을 inverse하여 구하는것과 같다. 즉 ouput과 kernel을 할때 input을 구하는것과 같다. 하지만 transposed convolution은 학습을 통해 kernel을 구해나간다.

</div></font>


<br><br>
##### 3. INVESTIGATING AND VISUALIZING THE INTERNALS OF THE NETWORKS

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-08-Deep Convolutional Generative Adversarial Networks (DCGAN)
/Figure 4.png){: .align-center}
만약 $Z$(noise)가 천천히 변함 따라 급격히 변한다면 memorization이 일어났다고 볼수있지만, 제대로 학습된 Genertator라면, 모델의 $Z$(noise)를 천천히 변화시키면 Data Distribution의 Generation image가 천천히 변하는것을 관찰가능할 것이다. ***Figure 4***에서는 $Z$에 따라 천천히 변하는 이미지들을 보여주고 있다. <br>


![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-08-Deep Convolutional Generative Adversarial Networks (DCGAN)
/Figure 5.png){: .align-center}
또한 ***Figure 5***에서는 각각의 filter들이 침대나 창문과 같이 침실의 일부를 학습하였고, 필터 시각화를 통해 기존의 모델들이 Black Box였던 문제점을 조금이나마 해소하였다. <br>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-08-Deep Convolutional Generative Adversarial Networks (DCGAN)
/Figure 6.png){: .align-center}
***Figure 6***에서의 윗부분 이미지들은 아무 수정 없이 Generator가 만들어낸 이미지들이고 아랫부분의 이미지들은 학습된 filter들에서 특정 object인 창문을 나타내는 filter를 삭제하고 Generator가 이미지를 만들어낸 것이다. 결과를 보면 창문대신에 문이나 거울 같은 다른 object를 만들어 내는것을 볼수있다. <br>



<br><br>
##### 4. Vector Arithmetic

위에서 DCGAN은 semantic qualities를 쉽게 조작 할수있는 vector arithmetic 특성을 갖는다고 했다. 이러한 특성은 자연어 처리 분야의 word2vec과 같은 특성을 지니는데, 쉽게 말하면, $"King"(vector) - "Man"(vector) + "Woman"(vector) = "Queen"(vector)$과 같은 연산 가능한 벡터의 특성을 가지고 있다. ***Figure 7***에서 보는것과 같이 "man with glasses"의 벡터 $Z$의 평균과 "man without glasses"의 벡터 $Z$의 평균을 빼고 "woman without glasses"의 벡터 $Z$의 평균을 더하면 "woman with glasses"의 벡터 $Z$가 만들어 진다는 것이다. <font size = 2>(single sample의 경우 불안정하여 3개의 벡터 $Z$를 평균으로 하여 사용했다.) </font>

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-08-Deep Convolutional Generative Adversarial Networks (DCGAN)
/Figure 7.png){: .align-center}

<br>

또한 ***Figure 8***과 같이 왼쪽을 보고있는 얼굴의 벡터 $Z$의 평균과 오른쪽을 보고 있는 얼굴의 벡터 $Z$의 평균을 interpolation하여 Generator에 input으로 넣어보았더니, 천천히 회전하는 얼굴이 나오는것을 볼수 있었다.
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-08-Deep Convolutional Generative Adversarial Networks (DCGAN)
/Figure 8.png){: .align-center}


</div>