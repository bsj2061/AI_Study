---
description: Linear Span(선형생성)에 대하여 알아보는 페이지입니다.
---

# Linear Span

## 1) Linear Span(선형 생성)

벡터공간 $$\boldsymbol{V}$$의 공집합이 아닌 부분집합 $$S=\{\mathrm{v}_1,\mathrm{v}_2,\,...\,,\mathrm{v}_n\}$$내의 벡터들의 가능한 모든 선형결합으로 이루어진 $$\boldsymbol{V}$$의 부분벡터공간을 $$S$$의 (선형)생성 $$span(S)$$이라 한다. 즉,

$$span(S)=\left\{\underset{i=1}{\overset{n}{\sum}}k_i\mathrm{v}_i|k_i\in\boldsymbol{F},\,\mathrm{v}_i\in S\right\}$$

이때 '$$S$$가 $$span(S)$$을 (선형)생성한다'라고한다.

## 2) Linear Dependent(선형 독립)

벡터공간 $$\boldsymbol{V}$$의 공집합이 아닌 부분집합 $$S=\{\mathrm{v}_1,\mathrm{v}_2,\,...\,,\mathrm{v}_n\}$$에 대하여

$$k_1\mathrm{v}_1+k_2\mathrm{v}_2+\,...\,+k_n\mathrm{v}_n = \vec{0}\\\Rightarrow k_1=k_2=\,...\,=k_n=0$$

이면 $$S$$가 선형 독립(linearly indenpendent)이라고 한다. 만약 $$k_1=k_2=\,...\,=k_n=0$$외의 다른 해가 존재하면 $$S$$가 선형 종속(linearly dependent)이라고 한다.
