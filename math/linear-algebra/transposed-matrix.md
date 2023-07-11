---
description: Transposed Matrix(전치행렬)에 대하여 알아보는 페이지입니다.
---

# Transposed Matrix

## 1)Transposed Matrix란?

전치행렬이란 행과 열을 교환하여 얻는 행렬을 말한다. 주대각선을 축으로 하여 반사 대칭을 하면 전치행렬이 나온다. 여기서 주대각선이란 행과 열의 지표가 같은 행렬을 말한다. 전치행렬을 수학적으로 쓰면 다음과 같다.&#x20;

$$M_{ij}^T = M_{ji}$$

예를 들어서 행렬 $$M$$이 Eq.1과 같을 때 행렬 $$M$$의 전치행렬은 Eq.2와 같다.

$$M$$ = $$\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}$$            Eq.1

$$M^T=\begin{bmatrix}1&4&7\\2&5&8\\3&6&9\end{bmatrix}$$          Eq.2

## 2)Transposed Matrix의 연산법칙

1. $$(M+N)^T = M^T + N^T$$
2. $$(cM)^T=cM^T$$
3. $$M^{TT}=M$$
4. $$(MN)^T = N^TM^T$$
5. rank $$M^T$$=rank $$M$$
6. tr $$M^T$$=tr $$M$$
7. det $$M^T$$=det $$M$$
8. $$(M^T)^{-1} = (M^{-1})^T$$

## 4) 참고문헌

[https://ko.wikipedia.org/wiki/%EC%A0%84%EC%B9%98%ED%96%89%EB%A0%AC](https://ko.wikipedia.org/wiki/%EC%A0%84%EC%B9%98%ED%96%89%EB%A0%AC)
