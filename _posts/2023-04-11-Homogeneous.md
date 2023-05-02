---
title:  <font size="5">Homogeneous</font>
excerpt: "Homogeneous"
toc: true
toc_sticky: true
use_math: true
categories:
  - Linear Algebra
tags:
  - Linear Algebra
  - Vision
  - Geometry
last_modified_at: 2023-04-03T11:10:00-55:00
---

--------


> Homogeneous란, $f(x) = a_n\frac{\mathrm{d^n} y}{\mathrm{d} x^n} + a_{n-1}\frac{\mathrm{d^{n-1}} y}{\mathrm{d} x^{n-1}} + \cdots + a_1\frac{\mathrm{d} y}{\mathrm{d} x} + a_0y$ n계 선형 미분방정식에서 아래와 같이 정의 <br><br>
> $f(x) = 0$이면 '동차(homogeneous)' <br>
> $f(x) \neq 0$이면 '비동차(inhomogeneous)' <br> 

<br><br>
<div markdown = "1">
#### <center>Homogeneous</center>
Homogeneous의 뜻은 **"동차"**이다. 즉, 차수가 동일하다는 뜻이다.<br>

위에서 정의한 것과 같이 $f(x) = 0$이므로, <br>
$\Rightarrow f(\alpha x) = \alpha f(x) = 0$이고,<br>
$\Rightarrow f(\alpha x) = \alpha^2 f(x) = \alpha^n f(x)$이다.<br>
딱 보면, "$\alpha$의 차수를 확장해도 동질성이 유지되는 애를 **'Homogeneous하다'** 라고 하구나." 라고 알 수 있다.

여기서 보면은, $x$ 하나에 대해서 사용하기 때문에 $\alpha$가 그대로 나오지만, 예를들어 $f(x,y) = x^2y^3 + x^3y^2$이라면 아래와 같이 진행된다.<br>
$$ \begin{align*} 
f(\alpha x, \alpha y) &= \left (\alpha x \right)^2 \left (\alpha y \right)^3 + \left (\alpha x \right)^3 \left (\alpha y \right)^2
\\ &= \alpha^5 \left (x^2y^3 + x^3y^2\right)
\\ &= \alpha^5 \left (f(x,y) \right)
\end{align*}$$ 

$x, y$에 대해 각각 $\alpha = 2$배를 하면 $f(x)$는 $\alpha^5 = 2^5$ 만큼 scale이 변한다는 뜻이다.<br>
이러한 방식은 Vision에서 Projective Geometry를 다룰때 사용된다. 예를들어, 2D Euclidean space에서는 $\left(x, y \right)$로 나타내었던 좌표를 Projective Geometry에서는 $\left(x, y, 1 \right)$**(Homogeneous Coordinate)**로 나타내여, $\left(kx, ky, k \right)$와 같은 scale에 대한 동질성을 다룰수 있게된다.



</div>


