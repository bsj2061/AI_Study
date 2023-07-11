---
description: 서포트 벡터 머신(SVM,Suppor Vector Machine)의 개념에 대해 알아보는 페이지입니다.
---

# Concept

## 1) 서포트 벡터와 마진

&#x20;훈련세트 $$D=\{(x_1,y_1),(x_2,y_2), ...  ,(x_m,y_m)\}, y_i \in \{-1,+1\}$$가 주어졌을 때 이 훈련세트를 분류하는 가장 기본적인 아이디어는 훈련 세트 $$D$$의 샘플 공간에서 한 분할 초평면을 찾는 것이다. 그렇다면 다음과 같은 데이터가 주어졌을 때 어떤 분할 초평면을 선택하는 것이 가장 합리적일까?

![](<../../.gitbook/assets/image (15).png>)

직관적으로 1이라는 것을 알 수 있을 것이다. 예를 들어 훈련 세트에 노이즈나 어떤 영향으로 인해 새로운 샘플이 분류 경계에 가까이 가게 된다면 2, 3에서는 오류가 생기게 된다. 그렇기 때문에 가장 합리적인 분할 초평면은 데이터셋에 생기는 에러로부터 영향을 크게 받지 않는 분할 초평면이라고 할 수 있다. 즉, 훈련 샘플 공간에서 다른 클래스를 가진 훈련 샘플 포인트들로부터의 거리가 먼 분할 초평면이라고 할 수 있다. 그렇다면 그러한 분할 초평면을 어떻게 결정할 수 있을까? 이때 필요한 것이 서포트 벡터와 마진이다.

![](<../../.gitbook/assets/image (8).png>)

샘플 공간에서 분할 초평면은 다음 선형 방정식 Eq.1을 통해 나타나진다.

&#x20;                                                                $$\boldsymbol{w}^T\boldsymbol{x}+b = 0$$&#x20;                     Eq.1

여기서 $$\boldsymbol{w}=(w_1;w_2;...;w_d)$$는 법선 벡터이고 초평면의 방향을 결정한다. 또한 $$b$$는 변위 항으로 초평면과 원점사이의 거리를 결정한다. 그리고 샘플 공간에서의 임의점 $$x$$에서 초평면 $$(\boldsymbol{w},b)$$까지의 거리는 Eq.2로 나타나진다.

&#x20;                                                                $$r = \frac{|\boldsymbol{w}^T\boldsymbol{x}+b|}{||\boldsymbol{w}||}$$                          Eq.2

초평면 $$(\boldsymbol{w},b)$$가 훈련 샘플을 정확히 분류할 수 있다면 $$(\boldsymbol{x}_i,y_i)\in D$$에서 Eq.3과 같다.

&#x20;                                            $$\begin{cases}\boldsymbol{w}^T\boldsymbol{x}_i+b \geq + 1, &\mbox{}y_i=+1;\\\boldsymbol{w}^T\boldsymbol{x}_i+b \leq - 1, &\mbox{}y_i =-1\end{cases}$$             Eq.3

Eq.3 등호에 해당하는 초평면에 가장 가까운 몇 개의 샘플 포인트를 서포트 벡터(support vector)라고 부른다. 또 두 개의 서로 다른 클래스의 서포트 벡터에서 초평면에 달하는 거리의 합을 마진(margin)이라고 부르고, Eq.4로 나타낼 수 있다.

&#x20;                                                                  $$\gamma = \frac{2}{||\boldsymbol{w}||}$$ ​                              Eq.4



최대 마진(maximum margin)을 가지는 분할 초평면을 가지고 싶다면 Eq.3의 제약 조건을 만족하는 파라미터 $$\boldsymbol{w}$$와 $$b$$를 찾아서 $$\gamma$$를 최대화해야 한다.

&#x20;             $$\underset{\boldsymbol{w},b}{max} \, \frac{2}{||\boldsymbol{w}||}\;\;\;\;s.t.\;\;\;y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\,\ge\,1,\;\;\;i=1,2,...,m.$$    Eq.5

Eq.5를 통해 최대 마진을 구하기 위해서는 $$||\boldsymbol{w}||^{-1}$$​만 최대화하면 되고, 이는 $$||\boldsymbol{w}||^{2}$$​를 최소화하는 것과 같다. 따라서 Eq.5는 Eq.6으로 다시 작성된다.

&#x20;                                                      $$f(x) = \boldsymbol{w}^T\boldsymbol{x}+b$$                      Eq.6

이것이 서포트 벡터 머신(SVM)의 기본 모델이다.​

## 2) 쌍대문제

Eq.6를 통해 최대 마진 분할 초평면에 대응하는 모델을 효과적으로 구하기 위해 Eq.6에 라그랑주 승수법을 써서 쌍대문제(dual problem)을 얻을 수 있다. Eq.6의 각 제약 조건에 라그랑주 승수 $$\alpha_i\,\,\ge\,\,0$$을 추가하면 해당 문제의 라그랑주 함수는 다음과 같다.

&#x20;                $$L(\boldsymbol{w},b,\boldsymbol{\alpha}) = \frac{1}{2}||\boldsymbol{w}||^2+\underset{i=1}{\overset{m}{\sum}}\alpha_i(1-y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b))$$              Eq.7

여기에서 $$\boldsymbol{\alpha}=(\alpha_1;\alpha_2;\,...\,;\alpha_m)$$이다. $$\boldsymbol{w},b$$에 편도함수에 대한 $$L(\boldsymbol{w},b,\boldsymbol{\alpha})$$​를 0으로 두면, Eq.8과 Eq.9를 얻을 수 있다.

&#x20;                                                                $$\boldsymbol{w}=\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i\boldsymbol{x}_i$$                    Eq.8

&#x20;                                                                  $$0=\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i$$                        Eq.8

Eq.7에 Eq.8을 대입하면, ​$$L(\boldsymbol{w},b,\boldsymbol{\alpha})$$에서 $$\boldsymbol{w}$$​와 $$b$$를 없앨 수 있고, 다시 Eq.8의 제약조건을 고려하면 Eq.6의 쌍대문제를 얻을 수 있다.

&#x20;                                  $$\underset{\boldsymbol{\alpha}}{\max}\;\underset{i=1}{\overset{m}{\sum}}\alpha_i-\frac{1}{2}\underset{i=1}{\overset{m}{\sum}}\underset{j=1}{\overset{m}{\sum}}\alpha_i\alpha_jy_iy_j\boldsymbol{x_i}^T\boldsymbol{x_j}\;\\\;\;\;\;s.t.\;\;\; \underset{i=1}{\overset{m}{\sum}}\alpha_iy_i=0,\alpha_i\ge0,i=1,2,\,...\,,m$$          Eq.9

$$\boldsymbol{\alpha}$$의 해를 구한 후 $$\boldsymbol{w}$$와 $$b$$를 구하면 다음 모델을 얻는다.

&#x20;                                                $$f(x) = \boldsymbol{w}^T\boldsymbol{x}+b=\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i\boldsymbol{x}_i^T\boldsymbol{x}+b$$          Eq.10

Eq.9의 쌍대문제에서 나온 $$\boldsymbol{\alpha}_i$$​는 Eq.7의 라그랑주 승수이고, 훈련 샘플 $$(\boldsymbol{x}_i,y_i)$$​에 대응한다. 또한 Eq.6에서 부등식 제약 조건이 있으므로 이 과정들은 KKT(Karush-Kuhn-Tucker) 조건을 만족해야 한다. 따라서 Eq.11과 같이 나타난다.

&#x20;                                                        $$\begin{cases}\alpha_i\ge0; \\ y_if(\boldsymbol{x}_i)-1\ge0;\\\alpha_i(y_if(\boldsymbol{x}_i)-1)=0; \end{cases}$$                 Eq.11 ​

Eq.11을 통해 모든 훈련 샘플$$(\boldsymbol{x}_i,y_i)$$이 $$\boldsymbol{\alpha}_i=0$$​이나 $$y_if(\boldsymbol{x}_i)=1$$을 만족한다는 것을 알 수 있다. 만약 $$\boldsymbol{\alpha}_i=0$$이라면 해당 샘플은 Eq.10의 합 부분에 존재하지 않을 것이고, $$f(\boldsymbol{x})$$​에 영향이 없을 것이다. 만약 $$\boldsymbol{\alpha}_i>0$$이라면 $$y_if(\boldsymbol{x}_i)=1$$은 무조건 존재할 것이고, 이에 대응하는 샘플 포인트는 최대 마진 경계상에 위치하게 될 것이다. 즉 하나의 서포트 벡터라는 뜻이다. 이것들을 통해서 훈련이 완료된 후 최종 모델은 오직 서포트 벡터에만 관련이 있다는 것을 알 수 있다.

Eq.9의 해를 효율적으로 구하기 위하여 많은 방법들 중 SMO(Sequential Minimal Optimization,순차적 최소 최적화)를 이용한다.

SMO의 기본적인 사상은 먼저 $$\boldsymbol{\alpha_i}$$ 외의 모든 파라미터를 고정하고 $$\boldsymbol{\alpha_i}$$에서의 극한값을 구하는 것이다. 제약 조건 $$\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i=0$$이 존재하므로 $$\boldsymbol{\alpha_i}$$외에 기타 변수를 고정한다면 $$\boldsymbol{\alpha_i}$$는 기타 변수에 의해 도출 가능하다. 따라서 SMO는 매번 두 개의 변수 $$\boldsymbol{\alpha_i}$$와 $$\boldsymbol{\alpha_j}$$​를 선택하고 기타 파라미터들을 고정시킨다. 이렇게 하면 파라미터를 초기화한 후 SMO는 아래의 과정을 수렴할 때까지 반복한다.

&#x20;      업데이트가 필요한 변수 $$\boldsymbol{\alpha_i}$$와 $$\boldsymbol{\alpha_j}$$ 한 쌍을 선택한다.

&#x20;       $$\boldsymbol{\alpha_i}$$와 $$\boldsymbol{\alpha_j}$$​ 이외의 파라미터를 고정시키고 Eq.9의 해를 구해 새로운 $$\boldsymbol{\alpha_i}$$와 $$\boldsymbol{\alpha_j}$$​를 구한다.

만약 $$\boldsymbol{\alpha_i}$$나 $$\boldsymbol{\alpha_j}$$​가 KKT 조건(Eq.11)을 만족하지 못한다면, 목표 함수는 반복적으로 증가한다. 직관적으로 KKT 조건을 위배한 정도가 크면 클수록 변수 업데이트 이후 목표함수가 증가하는 정도가 커지게 되므로 SMO는 KKT 조건 위배 정도가 가장 큰 변수를 선택한다. 각 변수가 대응하는 목표 함수값의 증가 정도를 파악하는 것은 복잡도가 매우 높기 때문에 SMO는 휴리스틱한 방법을 사용한다. 즉, 선택한 두 변수가 대응하는 샘플의 마진값이 가장 큰 것을 선택한다. 이러한 두 변수가 큰 차이를 가질 때 목표함숫값을 더 크게 변화시킨다.

Eq.9의 제약 조건에서 $$\boldsymbol{\alpha_i}$$와 $$\boldsymbol{\alpha_j}$$​만을 고려하면 Eq.12와 같이 다시 써진다.

&#x20;                                $$\alpha_iy_i + \alpha_jy_j = c, \;\;\alpha_i\geq0,\;\;\alpha_j\ge0$$                Eq.12

여기서,&#x20;

&#x20;                                                           $$c = -\underset{k\ne i,j}{{\sum}}\alpha_ky_k$$                        Eq.13

식 Eq.13은 $$\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i=0$$을 성립하게 하는 상수이다.

Eq.12를 사용하여 Eq.9의 변수 $$\alpha_j$$를 소거하면 $$\alpha_i$$의 단변량 이차 프로그래밍 문제를 얻게 된다.

여기에 존재하는 제약 조건은 $$\alpha_i\ge0$$​뿐이다. 따라서 이러한 프로그래밍 문제는 닫힌 형식의 해가 존재하므로, 최적화 알고리즘의 별다른 조정 없이 효율적으로 업데이트 후의 $$\alpha_i$$와 $$\alpha_j$$​를 계산할 수 있다.

$$b$$를 얻기 위해서 우선 모든 서포트 벡터$$(\boldsymbol{x}_s,y_s)$$에 대해서 ​$$y_sf(x_s)=1$$​이 존재하기 때문에 Eq.14가 된다.

&#x20;                                          $$y_s\left(\underset{i\in S}{\sum}\alpha_iy_i\boldsymbol{x}_i^T\boldsymbol{x}_s+b\right)=1$$                     Eq.14​

여기서 $$S = \{i|\boldsymbol{\alpha}_i>0,i=1,2,\,...\,,m\}$$는 모든 서포트 벡터의 하위 인덱스 셋이다. 이론적으로 모든 서포트 벡터를 선택할 수 있고, Eq.14를 통해 $$b$$를 얻을 수 있다. 하지만 현실에서 자주 사용하는 로버스트(robust)한 방법은 모든 서포트 벡터 해의 평균값을 사용하는 것이다.

&#x20;                                         $$b = \frac{1}{|S|}\underset{s\in S}{\sum}\left(\frac{1}{y_s}-\underset{i\in S}{\sum}\alpha_iy_i\boldsymbol{x}_i^T\boldsymbol{x}_s\right)$$                    Eq.15

## 3) 커널함수

앞서서 한 논의에서는 하나의 분할 초평면이 훈련 샘플들을 한번에 정확하게 선형 분리가능하다고 가정했었다. 그러나 현실에서는 샘플 공간 내에 한 번에 모든 클래스를 정확히 분류할 수 있는 초평면이 없을 수 있다. 그렇기 때문에 샘플을 원시 공간에서 더 높은 차원의 특성 공간으로 투영하여 특성 공간 내에서 선형 분리 가능하게 만들 수 있다. 만약 원시 공간이 유한한 차원을 가졌다면, 반드시 고차원 특성 공간에서 샘플을 분할할 수 있기 때문에 원시 공간에서 하나의 초평면으로 분리할 수 없다고 해도 고차원으로 투영시키면 분리가능하다.

$$\phi(\boldsymbol{x})$$를 $$\boldsymbol{x}$$를 특성 공으로 투영시킨 후의 고유 벡터라고 한다면, 특성 공간에서 분할 포평면에 대응하는 모델은 Eq.16 같다.

&#x20;                                                     $$f(\boldsymbol{x})=\boldsymbol{w}^T\phi(\boldsymbol{x})+b$$              Eq.16

여기서 $$\boldsymbol{w}$$와 $$b$$​는 모델 파라미터이고 Eq.6과 비슷하게 Eq.17을 갖는다.

&#x20;                                   $$\underset{\boldsymbol{w},b}{min}\frac{1}{2}||\boldsymbol{w}||^2 \\ s.t. \;\;y_i(\boldsymbol{w}^T\phi(\boldsymbol{x}_i)+b) \ge 1,\;\; i=1,2,\cdots,m$$        Eq.17

이 식의 쌍대문제는 Eq.18이다.

&#x20;                                     $$\underset{\boldsymbol{\alpha}}{max}\underset{i=1}{\overset{m}{\sum}}\alpha_i-\frac{1}{2}\underset{i=1}{\overset{m}{\sum}}\underset{j=1}{\overset{m}{\sum}}\alpha_i\alpha_jy_iy_j\phi(\boldsymbol{x}_i)^T\phi(\boldsymbol{x}_j)\\s.t.\;\;\underset{i=1}{\overset{m}{\sum}}\alpha_iyi=0,\;\;\alpha_i\geq0,\;\;i=1,2,\cdots,m$$          Eq.18

Eq.18을 계산하기 위해서 $$\phi(\boldsymbol{x}_i)^T\phi(\boldsymbol{x}_j)$$를 계산해야 한다. 이것은 샘플 $$\boldsymbol{x}_i$$와 $$\boldsymbol{x}_j$$​가 특성 공간에 투영된 후의 내적이다. 특성 공간 차원수가 매우 높을 수 있기 때문에 바로 $$\phi(\boldsymbol{x}_i)^T\phi(\boldsymbol{x}_j)$$를 계산하기는 매우 힘들다. 따라서 이러한 문제를 해결하기 위해 Eq.19와 같은 함수를 가정한다.

&#x20;                                    $$\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)=<\phi(\boldsymbol{x}_i),\phi(\boldsymbol{x}_j)>=\phi(\boldsymbol{x}_i)^T\phi(\boldsymbol{x}_j)$$         Eq.19

즉, $$\boldsymbol{x}_i$$와 $$\boldsymbol{x}_j$$의 특성 공간에서의 내적은 그들의 원시샘플 공간에서 $$\kappa(\cdot,\cdot)$$를 통해 계산된 결과라는 것이다. 이러한 함수를 가정하면 굳이 고차원의 특성 공간에서 내적을 계산할 필요가 없으므로 Eq.18는 Eq.20으로 다시 써진다.

&#x20;                                    $$\underset{\boldsymbol{\alpha}}{max}\underset{i=1}{\overset{m}{\sum}}\alpha_i-\frac{1}{2}\underset{i=1}{\overset{m}{\sum}}\underset{j=1}{\overset{m}{\sum}}\alpha_i\alpha_jy_iy_j\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)\\s.t.\;\;\underset{i=1}{\overset{m}{\sum}}\alpha_iyi=0,\;\;\alpha_i\geq0,\;\;i=1,2,\cdots,m$$             Eq.20

해를 구한 후 다음을 얻는다.

$$f(\boldsymbol{x})=\boldsymbol{w}^T\phi(\boldsymbol{x})+b=\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i\phi(\boldsymbol{x}_i)^T\phi(\boldsymbol{x})+b=\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i\kappa(\boldsymbol{x},\boldsymbol{x}_i)+b$$      Eq.21

여기서 함수 $$\kappa(\cdot,\cdot)$$가 바로 커널함수(kernel function)이다. Eq.21은 모델의 최적해는 훈련샘플의 커널 함수를 전개하여 얻을 수 있다는 것을 보여주고 이러한 식을 서포트 벡터 전개(support vector expansion)이라고 한다.

만약 적절한 투영 $$\phi(\cdot)$$의 구체적인 형식을 알 수 있다면 커널함수 $$\kappa(\cdot,\cdot)$$을 특정할 수 있다. 그러나 현실에서는 $$\phi(\cdot)$$의 형식을 알 수 없다. 그렇다면 커널함수의 존재성에 관한 질문을 할 수 있을 것이다. 이러한 질문의 답은 다음의 정리와 같다.

#### Thm.1  커널 함수: $$\chi$$가 입력 공간을 나타내고, $$\kappa(\cdot,\cdot)$$가 $$\chi\times\chi$$에서의 대칭 함수라면, $$\kappa$$​는 모든 데이터 $$D=\{\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_m\}$$에 대한 커널 함수이다. 커널 행렬 ​$$\boldsymbol{\Kappa}$$​는 항상 양의 준정부호 행렬(positive semi-definite matrix)이다.

&#x20;                            $$\boldsymbol{\Kappa}=\begin{bmatrix}\kappa(\boldsymbol{x}_1,\boldsymbol{x}_1)&\cdots&\kappa(\boldsymbol{x}_1,\boldsymbol{x}_j)&\cdots&\kappa(\boldsymbol{x}_1,\boldsymbol{x}_m)\\ \vdots&\ddots&\vdots&\ddots&\vdots\\ \kappa(\boldsymbol{x}_i,\boldsymbol{x}_1)&\cdots& \kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)&\cdots&\kappa(\boldsymbol{x}_i,\boldsymbol{x}_m)\\ \vdots&\ddots&\vdots&\ddots&\vdots\\ \kappa(\boldsymbol{x}_m,\boldsymbol{x}_1)&\cdots& \kappa(\boldsymbol{x}_m,\boldsymbol{x}_j)&\cdots&\kappa(\boldsymbol{x}_m,\boldsymbol{x}_m) \end{bmatrix}$$

Thm.1 처럼 하나의 대칭함수에 대응하는 커널 행렬이 양의 준정부호 행렬이라면 이를 커널 함수로 사용할 수 있다. 사실상 하나의 양의 준정부호 커널 행렬에 대해 항상 대응하는 투영 $$\phi$$을 찾을 수 있다. 다시 말해서, 모든 커널 함수는 은연중에 재생 커널 힐베르트 공간(Reproducing Kernel Hilbert Space,RKHS)이라고 부르는 특성 공간을 정의하고 있다는 것이다.​

위에서 확인한 것처럼 우리는 샘플이 특성 공간 내에서 선형 분리될 수 있기를 바란다. 따라서 특성 공간의 좋고 나쁨은 SVM의 성능에 큰 영향을 미친다. 우리는 특정 투영의 형식을 모를 때 어떤 커널 함수가 가장 적합한지 알지 못한다. 그리고 커널 함수도 단순히 은연중에 이러한 특성 공간을 정의하고 있는 것 뿐이기 때문에 커널 함수의 선택은 SVM의 최대 변수가 된다. 만약 커널 함수의 선택이 적절하지 못하다면 샘플을 부적절한 특성 공간에 투영시킨다는 뜻이되고 이는 학습기의 성능에 큰 영향을 미친다.

다음 표 자주 사용하는 커널 함수들이다.&#x20;

|          |                                                                                                                           |                          |
| :------: | :-----------------------------------------------------------------------------------------------------------------------: | :----------------------: |
|     명    |                                                             표현                                                            |           파라미터           |
|   선형 커널  |                      $$\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)=\boldsymbol{x}_i^T\boldsymbol{x}_j$$                     |                          |
|  다항식 커널  |                    $$\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)=(\boldsymbol{x}_i^T\boldsymbol{x}_j)^d$$                   |    다항식의 차수는 $$d\ge1$$​   |
|  가우스 커널  | $$\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)=\exp\left(-\frac{||\boldsymbol{x}_i-\boldsymbol{x}_j||^2}{2\sigma^2}\right)$$ | 가우스 커널의 넓이는 $$\sigma>0$$ |
|  라플라스 커널 |    $$\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)=\exp\left(-\frac{||\boldsymbol{x}_i-\boldsymbol{x}_j||}{\sigma}\right)$$   |       $$\sigma>0$$       |
| 시그모이드 커널 |            $$\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)=\tanh(\beta\boldsymbol{x}_i^T\boldsymbol{x}_j+\theta)$$            |   $$\beta>0,\theta>0$$   |

텍스트 데이터에 대해서는 일반적으로 선형 커널을 사용하고, 잘 모를 때는 먼저 가우스 커널을 사용하는 것이 일반적이다. 또 다항식 커널은 $$d=1$$​일 때 선형 커널이 되고, 가우스 커널은 RBF(Radial Basis Function) 커널이라고도 한다.

이외에도 함수 조합을 통해 얻을 수 있다.

* 만약 $$\kappa_1$$​과 $$\kappa_2$$​가 커널 함수라면, 임의의 정수  $$\gamma_1$$​,$$\gamma_2$$에 대한 선형 조합 ​

&#x20;                                                        $$\kappa_1\gamma_1+\kappa_2\gamma_2$$                         Eq.22

* 만약 $$\kappa_1$$과 $$\kappa_2$$가 커널 함수라면, 커널 함수의 직접곱(direct product)

&#x20;                                   $$\kappa_1\bigotimes\kappa_2(\boldsymbol{x},\boldsymbol{z})=\kappa_1(\boldsymbol{x},\boldsymbol{z})\kappa_2(\boldsymbol{x},\boldsymbol{z})$$​             Eq.23

* 만약 $$\kappa_1$$이 커널 함수라면 모든 함$$g(x)$$​에 대해

&#x20;                                         $$\kappa(\boldsymbol{x},\boldsymbol{z})=g(\boldsymbol{x})\kappa(\boldsymbol{x},\boldsymbol{z})g(\boldsymbol{z})$$                 Eq.24

이때, Eq.22부터 Eq.24까지는 모두 커널 함수이다.​

## 4)소프트 마진과 정규화

앞선 논의에서 서로 다른 클래스의 샘플을 완전히 분리 가능헌 초평면이 존재한다고 가정하였다. 그러나 현실에서는 특성 공간에서 훈련 샘플을 선형 분리 가능한 적절한 커널 함수를 알지 못한다. 만약 적절한 커널 함를 찾았다고 하더라도 이 선형 분리 가능한 결과가 과적합에 의한 것인지 쉽게 판단할 수 없다.

이러한 문제를 완화해 주는 방법은 서포트 벡터 머신에게 약간의 오류를 허용해 주는 것이다. 이러한 방법은 소프트 마진(soft margin)이라는 개념으로 이어진다.



앞서 소개한 서포트 벡터 머신은 모든 샘플이 제약 조건 Eq.3을 만족해야 한다. 즉, 모든 샘플들이 정확하게 분류되어야 한다. 이러한 방식을 하드 마진(hard margin)이라 하고, 소프트 마진은 이와 반대로 일정의 샘플들에 대해 제약 조건을 요구하지 않는다.

&#x20;                                                       $$y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\geq1$$​                Eq.25

당연하게도 마진을 최대화하는 동시에 제약 조건을 만족시키지 못하는 샘플은 최대한 적게 둔다. 따라서 최적화 목표는 다음과 같이 쓸 수 있다.

&#x20;                             $$\underset{\boldsymbol{w},b}{\min}\frac{1}{2}||\boldsymbol{w}||^2+C\underset{i=1}{\overset{m}{\sum}}l_{0/1}(y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)-1)$$​             Eq.26

이 식에서 $$C>0$$​은 하나의 상수이고 $$l_{0/1}$$​는 0/1 손실함수이다.

&#x20;                                                 $$l_{0/1}(z)=\begin{cases}1,\;\;\;if\;z<0;\\0,\;\;\;otherwise.\end{cases}$$              Eq.27

만약 C가 무한히 커지면 Eq.26는 모든 샘플에 대해 조건을 만족시킬 것을 요구한다. 따라서 Eq.26는 Eq.3과 같아진다. C가 유한한 값을 취하면 Eq.26은 일정 샘플이 조건을 만족시키지 못해도 괜찮다.

그러나 $$l_{0/1}$$은 non-convex,즉 비연속적이라는 수학적 특징이 있다. 따라서 Eq.26은 쉽게 해를 구할 수 없다. 그렇기 때문에 $$l_{0/1}$$​을 다른 손실함수로 대체한다. 이 대체된 손실함수들은 보통 convex하고 연속적이다.&#x20;

#### &#x20;                                힌지 손실 (honge loss): $$l_{hinge}(z)=\max(0,1,-z);$$ ​

#### &#x20;                                지수 손(exponential loss): $$l_{exp}(z)=\exp(-z);$$

#### ​                                  로지스틱 손실(logistic loss): $$l_{log}(z)=\log(1+\exp(-z)).$$

만약 힌지 손실을 사용한다면 Eq.​26는 Eq.28과 같이 바뀐다.

&#x20;                            $$\underset{\boldsymbol{w},b}{\min}\frac{1}{2}||\boldsymbol{w}||^2+C\underset{i=1}{\overset{m}{\sum}}\max(0,1,-y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b))$$​                   Eq.28

여유 변수(Slack Variables) $$\xi_i\geq0$$​을 가져오면, Eq.28은 다음과 같이 바꿔 사용할 수 있다.

&#x20;                                               $$\underset{\boldsymbol{w},b,\xi_i}{\min}\frac{1}{2}||\boldsymbol{w}||^2+C\underset{i=1}{\overset{m}{\sum}}\xi_i$$​                       Eq.29

&#x20;                                         $$s.t.\;\;\;y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b)\geq1-\xi_i\\\;\;\;\;\;\;\;\;\xi_i\geq0,i=1,2,\cdots,m$$              &#x20;

이것이 바로 자주 사용되는 '소프트 마진 벡터 머신' 입니다.

Eq.29에서 각 샘플은 모두 대응하는 여유 변수가 있고, 이는 해당 샘플이 제약 조건 Eq.25을 만족하지 못하는 정도에 대한 표현으로 사용한다. 하지만 Eq.6처럼 이차 프로그래밍 문제이므로 Eq.7처럼 라그랑주 승수를 통해 Eq.29의 라그랑주 함수를 얻을 수 있다.

$$L(\boldsymbol{w},b,\boldsymbol{\alpha},\boldsymbol{\xi},\boldsymbol{\mu})=\frac{1}{2}||\boldsymbol{w}||^2+C\underset{i=1}{\overset{m}{\sum}}\xi_i+\underset{i=1}{\overset{m}{\sum}}\alpha_i(1-\xi_i-y_i(\boldsymbol{w}^T\boldsymbol{x}_i+b))-\underset{i=1}{\overset{m}{\sum}}\mu_i\xi_i$$     Eq.30

여기서 $$\alpha_i\geq0,\mu_i\geq0$$은 라그랑주 승수이다.

$$\boldsymbol{w},b,\xi_i$$에 대한 $$L(\boldsymbol{w},b,\boldsymbol{\alpha},\boldsymbol{\xi},\boldsymbol{\mu})$$의 편도 함수를 0으로 두면 다음을 얻는다.

&#x20;                                                                $$\boldsymbol{w}=\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i\boldsymbol{x}_i$$                         Eq.31

&#x20;                                                                    $$0=\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i$$                           Eq.32

&#x20;                                                                   $$C=\alpha_i+\mu_i$$                          Eq.33​

Eq.30을 Eq.31\~Eq.33에 대하면 Eq.29의 쌍대 문제를 얻을 수 있다.

&#x20;                       $$\underset{\boldsymbol{\alpha}}{\max}\underset{i=1}{\overset{m}{\sum}}\alpha_i-\frac{1}{2}\underset{i=1}{\overset{m}{\sum}}\underset{j=1}{\overset{m}{\sum}}\alpha_i\alpha_jy_iy_j\boldsymbol{x}_i^T\boldsymbol{x}_j\\s.t.\;\;\underset{i=1}{\overset{m}{\sum}}\alpha_iy_i=0\;\;,\;\;0\leq\alpha_i\leq C\;\;,\;\;i=1,2,\cdots,m$$      Eq.34

Eq.34와 하드 마진의 쌍대 문제(Eq.9)와 비교하면 쌍대 변수의 제약 조건이 전자는 $$0\leq\alpha_i\leq C$$ 이고 후자는 $$\alpha_i\geq 0$$로 서로 다르다는 것을 알 수 있다. 따라서 동일한 알고리즘을 사용하여 Eq.34의 해를 구할 수 있다. 커널 함수를 도입하여 Eq.21과 동일한 서포트 벡터 전개식을 얻을 수 있다.

Eq.11과 같이 소프트 마진 서포트 벡터 머신에 대한 KKT조건은 Eq.35를 요구한다.

&#x20;                                        $$\begin{cases}\alpha_i\geq0,\;\;\;\;\mu_i\geq0,\\y_if(\boldsymbol{x}_i)-1+\xi_i\geq0,\\\alpha_i(y_if(\boldsymbol{x}_i)-1+\xi_i)=0,\\\xi_i\geq0,\;\;\;\mu_i\xi_i=0.\end{cases}$$           Eq.35

따라서 모든 훈련 샘플 $$(\boldsymbol{x}_i,y_i)$$에 대해 항상 $$\boldsymbol{\alpha}_i=0$$ 혹은 $$y_if(\boldsymbol{x}_i)=1-\xi_i$$ 를 갖는다. 만약 $$\boldsymbol{\alpha}_i=0$$일 경우 해당 샘플은 $$f(\boldsymbol{x})$$에 아무런 영향을 주지 않는다. 만약 $$\boldsymbol{\alpha}_i>0$$라면 $$y_if(\boldsymbol{x}_i)$$의 값은 $$1-\xi_i$$이 되고 해당 샘플은 서포트 벡터가 된다. Eq.33을 통해 만약 $$\boldsymbol{\alpha}_i<C$$라면 $$\mu_i>0$$​이고 $$\xi_i\leq0$$​이 된다.즉 해당 샘플은 최대 마진 경계상에 놓여있게 된다. 만약 $$\boldsymbol{\alpha}_i=C$$라면 $$\mu_i=0$$이고 이때 $$\xi_i\leq1$$이면 해당 샘플은 최대 마진 내에 위치하게 된다. 만약 $$\xi_i>1$$이면 해당 샘플은 잘못 분류된 것이다. 이를 통해 소프트 마진 서포트 벡터 머신의 최종 모델은 오직 서포트 벡터와 연관이 있는 것을 알 수 있다. 즉, 힌지 손실 함수를 사용하여 희소성을 유지한다.

&#x20;그렇다면 Eq.26에 대하여 다른 손실 함수를 사용할 수 있을까?

Eq.26의 0/1 손실 함수를 다른 손실함수로 바꾸어 다른 학습 모델을 얻을 수 있다. 이런 모델의 성질은 사용된 손실 함수와 직접적인 관계가 있다. 하지만 이들은 하나의 공통된 성질이 있다. 바로 최적화 목표의 첫번째 항이 분할 초평면의 마진 크기를 설명하고 있다는 것이다. 다른 항 $$\underset{i=1}{\overset{m}{\sum}}l(f(\boldsymbol{x}_i),y_i)$$​은 훈련 세트에서의 오차를 나타낸다. 더 일반적인 형식으로는 Eq.36과 같다.

&#x20;                                         $$\underset{f}{\min}\,\Omega(f)+C\underset{i=1}{\overset{m}{\sum}}l(f(\boldsymbol{x}_i),y_i)$$​                      Eq.36

$$\Omega(f)$$는 구조적 위험(structual risk)이라고 부르며 모델 $$f$$의 어떠한 성질을 설명한다. 여기서 구조적 위험이란 전체 리스크 중에서 모델의 구조적인 요인에 대응하는 위험을 뜻한다.

두 번째 항 $$C\underset{i=1}{\overset{m}{\sum}}l(f(\boldsymbol{x}_i),y_i)$$은 경험적 위험(emprical risk)이고 모델과 훈련 데이터 간의 부합 정도를 나타낸다. $$C$$는 둘에 대한 절충을 진행하며, 경험적 리스크 최소화 관점에서 볼 때 $$\Omega(f)$$는 우리가 어떠한 특성 성질을 가진 모델을 원한다는 것을 나타낸다.(ex. 복잡도가 비교적 적은 특성을 가진 모델 등). 이는 도메인 지식과 사용자 편의를 위한 수단을 제시한다. 다른 한편으로 해당 정보는 가설 공간을 줄여주는 것을 도와준다. 이에 따라 최소화 훈련 오차의 과적합 위험을 낮춰준다. 이런 시각에서 본다면 Eq.36은 정규화 문제라고도 볼 수 있다. 정규화는 일종의 '벌점'이라고 생각하면 쉽ㄴ다. 즉, 기대하지 않은 결과에 대해서 벌점을 주어 최적화 과정이 기대하는 목표로 갈 수 있게 만드는 것이다.  $$\Omega(f)$$는 정규화 항이라고 부르고 $$C$$​는 정규화 상수가 된다. $$L_p$$노름은 자주 사용되는 정규화 항이다. 그중에서 $$L_2$$노름 $$||\boldsymbol{w}||_2$$는 $$\boldsymbol{w}$$의 가중치를 최대한 균형 있게 만드는 경향을 보이고, $$L_0$$노름 $$||\boldsymbol{w}||_0$$과 $$L_1$$노름 $$||\boldsymbol{w}||_1$$​은 $$\boldsymbol{w}$$​의 가중치를 희소 있게 만드는 경향을 보인다.



## 5) 서포트 벡터  회귀

## 6) 커널 트릭
