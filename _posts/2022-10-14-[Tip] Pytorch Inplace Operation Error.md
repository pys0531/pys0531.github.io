---
title:  <font size="5">[Tip] Pytorch Inplace Operation Error</font>
excerpt: "[Tip] Pytorch Inplace Operation Error"
toc: true
toc_sticky: true
use_math: true
categories:
  - Machin Learning
tags:
  - Machin Learning
  - Pytorch
  - Inplace Operation
  - Error
  - Tip
last_modified_at: 2022-09-15T18:09:00-55:00
---

RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: ~~~
라는 오류가 발생할 때 이다.<br>


이때는 inplace operation이 있어 gradient를 계산하는 데 문제가 생겼다는 것이다. 새로운 주소값을 갖는 텐서를 만들어주지 않고 값 자체를 바꾸는 연산에서 일어나곤 한다.<br>
저 같은 경우는 Resnet 모듈에서 += , -= 같은 연산자를 사용하여 파라미터를 업데이트 하여 생겼다.
>x += i 대신에 x = x + i로 풀어써야함.<br>

만약 Inplace Operation을 정말 사용해야할 경우 .clone()을 붙혀 사용하면 된다.

또한 Underbar(_)가 붙어있는 연산은 바로 값에 저장되기 때문에 Inplace Operation 연산이다.
