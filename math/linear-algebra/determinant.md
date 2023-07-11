---
description: Determinant(행렬식)에 대해 알아보는 페이지입니다.
---

# Determinant

## 1) Determinant(행렬식)

정사각행렬 $$A$$를 하나의 수로써 대응시키는 특별한 함수. $$det(A)=|A|$$

행렬 $$A$$에서 $$i$$​번째 행과 $$j$$번째 열을 제외한 나머지 행렬을 $$M_{ij}$$​라고 하고 행렬 $$A$$의 $$i$$행,$$j$$열에 위치한 요소를 $$a_{ij}$$라 할 때,  $$det(A)$$는 라플라스 전개를 통해 다음과 같이 재귀적으로 정의된다.

&#x20;                                                                  $$det()=0$$

&#x20;                                              $$det(A)=\underset{j=1}{\overset{n}{\sum}}(-1)^{i+j}a_{ij}det(M_{ij})$$

이는 모든 행 $$i\in \{1,2,3,\,...\,,n\}$$에 대하여 같은 함수를 정의한다. 마찬가지로 $$j\in \{1,2,3,\,...\,,n\}$$에 대한 라플라스 전개를 통해 정의할 수 있다.​

## 2) 크래머 공식(Cramer's Rule)

연립일차방정식 $$AX=B$$​에서, $$A$$가 행렬식이 ​0이 아닌 정사각행렬일 때,

$$x_j=\frac{det A_j}{det A}=\frac{\begin{vmatrix} a_{11} \cdots b_1\cdots a_{1n}\\a_{21}\cdots b_2 \cdots a_{2n} \\ \vdots \;\;\;\;\;\;\;\;\;\vdots\;\;\;\;\;\;\;\;\;\vdots\\a_{n1}\cdots b_n\cdots a_{nn}\end{vmatrix}}{\begin{vmatrix} a_{11} \cdots a_{1j}\cdots a_{1n}\\a_{21}\cdots a_{2j} \cdots a_{2n} \\ \vdots \;\;\;\;\;\;\;\;\;\vdots\;\;\;\;\;\;\;\;\;\vdots\\a_{n1}\cdots a_{nj}\cdots a_{nn}\end{vmatrix}}$$              ($$j = 1,2,3,\cdots,n)$$

여기서 $$A_j$$는 $$A$$의 $$j$$번째 열을 $$B$$로 바꾼 행렬이다.

이를 통해 연립일차방정식의 해를 구할 수 있다.
