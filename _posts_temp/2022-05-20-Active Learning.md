---
title:  <font size="5">Active Learning</font>
excerpt: "Active Learning"
toc: true
toc_sticky: true
use_math: true
categories:
  - Machine Learning
tags:
  - Statistics
last_modified_at: 2022-04-08T22:39:00-55:00
---

**<font size="4">Active Learning</font>** : <font size="3">모델이 어려워하는 데이터를 판별하여, 판별된 데이터를 유저가 효율적으로 학습할 수 있도록 도와주는 기법</font>
<br><br><br>


<font size="3">
<div markdown = "1">
앞서 알아본 Entropy는 불확실성에 대한 척도라고 알아보았다.<br><br>
예를들면, 완전히 반듯한 동전이 있다고 가정할때, 동전의 앞면과 뒷면이 나올확률은 50:50으로 각각 같다.
이때, 앞면이 나올지 뒷면이 나올지 확실하지 않으므로 불확실성이 높다고한다.<br>
반대로, 동전이 크게 휘어서 앞면과 뒷면이 나올확률이 90:10이라고 하면은 불확실성이 낮다.

이렇게 불확실성은 정답을 예측할때 얼마나 확신할수 있는지에 대한 값이다.
이를 섀넌의 엔트로피 수식으로 계산하여 알아보았었다.<br>

<center>$H=\sum_i^np_iI(s_i) = -\sum_i^np_ilog(p_i)$</center>

<br>
<br>

DL에서 많이 쓰는 Binary Cross Entropy(BCE) 역시 Entropy 수식을 사용하여 계산한다.
높은 불확실성을 갖은 모델을 낮을 불확실성을 갖도록 하기위해 아래와 같은 수식을 이용해 Optimize 한다.
Binary이므로 $q=1-p$가 성립되어 다음과 같이 나타낸다.
<center>$H(p,q)=-\sum_{i=1}^{N} p_i*log(q_i)=-[p*log(1-p)+(1-p)*log(1-p)]$</center>
<font size="1"><div markdown = "1">
**<center>p=True probability distribution, q=Predicted probability distribution</center>**





</div></font>

</div>
</font>

