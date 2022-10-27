---
title:  <font size="5">[Tip] Pytorch CrossEntropy, BCELoss 주의사항</font>
excerpt: "[Tip] Pytorch CrossEntropy, BCELoss 주의사항"
toc: true
toc_sticky: true
use_math: true
categories:
  - Machin Learning
tags:
  - Machin Learning
  - Pytorch
  - CrossEntropy
  - BCELoss
  - Error
  - Tip
last_modified_at: 2022-09-15T18:09:00-55:00
---

CrossEntropy Loss를 많이 사용하는데 Pytorch의 경우 Class가 2개일 경우(즉, Class가 0, 1) BCELoss 또는 BCEWithLogitsLoss를 사용해야된다. <br><br>

Pytorch의 CrossEntropy Loss는 Class가 3이상에서만 돌아가기 때문이다. (즉, 마지막 레이어 노드수가 2개 이상이어야함.)

