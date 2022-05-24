---
title:  <font size="5">Kullback-Leibler (KL) Divergence & Jensen-Shannon Divergence</font>
excerpt: "Kullback-Leibler (KL) Divergence & Jensen-Shannon Divergence"
toc: false
toc_sticky: false
use_math: true
categories:
  - Statistics
tags:
  - Machine Learning
  - Statistics
last_modified_at: 2022-04-04T15:30:00-16:10:00
---

<br>
**<font size="4">Kullback-Leibler (KL) Divergence</font>** : <font size="3">두 분포가 얼마나 닮았는지 / 근사치</font> <br>
**<font size="4">Jensen-Shannon Divergence</font>** : <font size="3">두 분포가 얼마나 닮았는지 / Distance</font>
<br>

<font size="3"><div markdown = "1">
**Kullback-Leibler Divergence**<br>
우선 수식으로 확인해보면 아래와 같다.
<br>
<center>$D_{KL}(p||q)=E[log(p_i)-log(q_i)]=\sum_{i}p_ilog\frac{p_i}{q_i}$<br>
<font size="2">($p: $ 원본확률분포 $q: $ 근사된 분포 $i: $ i번째 item이 가진 정보량)</font></center>
<br>
수식에서 확인해보면 두 정보량간의 차이의 기댓값이다.<br><font size="2">=> $-log(q_i) - (-log(p_i))$ => $log(p_i) - log(q_i)$ </font>
<br><br>
간단히 말하면 KL-Divergence는 근사시 발생하는 정보 손실량의 기댓값이다. 두 분포의 차이를 이용해 얼마나 유사한지 근사치를 나타낸다.
이를 이용해서 보통 DeepLearning에서는 Cross-Entropy Loss를 구하곤한다.

<br><br>
**Jensen-Shannon Divergence**<br>
KL-Divergence는 단순히 두 분포가 얼마나 닮았는지 정도로 측정하는 척도로 이용이 가능하다.<br>
하지만 Jensen-Shannon Divergence는 KL-Divergence를 Symmetric하게 개량하여 두 확률 분포사이의 Distance로서의 역할을 할 수 있게 된다.
<center>$JSD(p,q)=\frac{1}{2}D_{KL}(p||\frac{p+q}{2})+\frac{1}{2}D_{KL}(q||\frac{p+q}{2})$</center><br>
q와 p에 대해서 KLD를 구하게 되어 Symmetric하게 만든다. Symmetric하게 되면 $JSD(p,q)=JSD(q,p)$가 만족되어 Distance 역할을 할 수 있게 된다.

<br>
<br>


기계학습에서는 복잡한 함수나 분포를 단순화하여 하나의 간단한 함수로 나타내서 비교적 적은 파라미터로 동일한 성능을 내도록 많은 노력을 한다. <br>
간단히 나타낸 함수가 원본분포와 약간의 오차는 있어도 얼마나 유사한지 차이를 나타내는 척도를 알아야하기 때문에 KL-Divergence를 이용하여 계산을 하곤한다.



</div>
</font>