---
description: MDP의 개념을 알아보는 페이지입니다.
---

# Concept

MDP(Markov Decision Process)는 $$S$$(상태의 집합), $$P$$(상태전이확률행렬), $$A$$(액션의 집합), $$R$$(보상함수), $$\gamma$$(감쇠인자)로 이루어져있다. MDP에 대해 본격적으로 알아보기 전에 $$S$$(상태의 집합), $$P$$(상태전이확률행렬)로만 이루어진 가장 간단한 MP(Markov Process)부터 여기에 $$R$$(보상함수)과 $$\gamma$$(감쇠인자)가 더해진 MRP(Markov Reward Process)를 알아보도록 하자.

## 1) MP(Markov Process)

MP는 앞서 말했듯이 $$S$$(상태집합)와 $$P$$(상태전이확률행렬)로 이루어져있다. 그렇다면 상태집합과 상태전이확률행렬이란 무엇일까?

상태집합은 상태로 이루어진 집합이다. 그리고 상태란 자신의 상황에 대한 관찰을 뜻한다. 또한 상태전이확률이란 어떤 상태 $$s$$에서 다른 상태 $$s'$$으로 넘어갈 확률을 의미하고 $$p_{ss'}$$​이라고 쓴다.

예를 들어서 공부에 집하는 과정을 MP로 모델링해보자. 그럼 다음과 같은 모델이 나온다. ​

![](<../../.gitbook/assets/image (3).png>)

기에서 각각 $$s_0,s_1,s_2,s_3,s_4$$를 상태라 하며 이 상태들을 모아 놓은 집합을 $$S$$(상태 집합)이라고 하고 이 경우에서는 Eq.1과 같이 정의한다.

&#x20;                                      $$S = \{s_0,s_1,s_2,s_3,s_4\}$$                 Eq.1​

상태는 총 5가지로 각각의 상태에서 10분 간 머물고 다른 상태로 전이한다. 그리고 시작 상태는 $$s_0$$​(책상에 앉아만 있는 상태)이고 종료 상태는 $$s_4$$(공부에 집중하는 상태)이다. 또한 다른 상태로 전이할 때에 필요한 것이 바로 전이 확률이며 다음과 같이 정의한다.

&#x20;                                  $$P_{ss'}=ℙ[S_{t+1}=s'|S_t=s]$$​              Eq.2

그리고 앞서 정의한 마르코프 프로세스 모인 공부에 집중하는 과정에서 상태전이확률행렬을 표로 나타내 다음과 같다.

<table data-header-hidden><thead><tr><th width="150" align="center"></th><th width="150" align="center"></th><th width="150" align="center"></th><th width="150" align="center"></th><th width="150" align="center">Text</th><th width="150" align="center"></th></tr></thead><tbody><tr><td align="center"></td><td align="center"><span class="math">s_0</span></td><td align="center"><span class="math">s_1</span></td><td align="center"><span class="math">s_2</span></td><td align="center"><span class="math">s_3</span>​</td><td align="center"><span class="math">s_4</span>​</td></tr><tr><td align="center"><span class="math">s_0</span></td><td align="center"></td><td align="center">0.5</td><td align="center">0.2</td><td align="center">0.3</td><td align="center"></td></tr><tr><td align="center"><span class="math">s_1</span></td><td align="center">0.2</td><td align="center">0.8</td><td align="center"></td><td align="center"></td><td align="center"></td></tr><tr><td align="center"><span class="math">s_2</span></td><td align="center">0.2</td><td align="center">0.3</td><td align="center"></td><td align="center">0.5</td><td align="center"></td></tr><tr><td align="center"><span class="math">s_3</span></td><td align="center"></td><td align="center"></td><td align="center">0.2</td><td align="center"></td><td align="center">0.8</td></tr><tr><td align="center"><span class="math">s_4</span></td><td align="center"></td><td align="center"></td><td align="center"></td><td align="center"></td><td align="center"></td></tr></tbody></table>

MP는 앞서 설명한 $$S$$(상태 집합)와 $$P$$(상태전이확률행렬)로 이루어져 있으며 Eq.3과 같이 표현할 수 있다.

&#x20;                                                        $$MP\equiv(S,P)$$                Eq.3​

앞서 설명한 MP(Markov Process)의 이름에 Markov가 들어가는 이유는 이 프로세스가 마르코프한 성질을 가지고 있기 때문이다. 그렇다면 마르코프 성질이란 무엇일까?

마르코프 성질은 "미래는 오직 현재에 의해서만 결정된다." 라는 것이다. 앞서 정의했던 공부에 집중하는 과정을 살펴보면, 학생이 책상을 정리하는 상태인 $$s_2$$에 있었다고 했을 때 다음 상태인 공부를 시작하는 상태 ​$$s_3$$​로 전이할 확률은 50%이다. 학생이 유튜브를 보는 상태에서 앉아 있는 상태로 전이한 후 책상을 정리하는 상태로 전이했든지 아님 앉아있는 상태에서 책상을 정리하는 상태로 전이했든지 간에 상관없이 공부를 시작하는 상태로 전이할 확률은 50%이다. 바로 이것이 마르코프 성질이다. 이렇듯 마르코프한 상황이 있는가 하면 그렇지 않은 상황도 존재한다. 그렇기 때문에 상황을 마르코프한 상황으로 모델링하는 것은 강화학습에 있어서 중요하다.

## 2) MRP(Markov Reward Process)

MRP(Markov Reward Process) $$S$$(상태 집합)와 $$P$$(상태전이확률행렬)로 이루어진 MP에 $$R$$(보상 함수)​과 $$\gamma$$(감쇠 인자)가 추가된 형태로 Eq.4와 같이 정의된다.

&#x20;                                       $$MRP\equiv(S,P,R,\gamma)$$                            Eq.4

&#x20;MRP에서의 $$S$$​와 $$P$$​는 MP에서와 같다. 그렇다면 $$R$$​(보상 함수)와 $$\gamma$$​(감쇠 인자)는 무엇일까?

우선 $$R$$​(보상 함수)는 Eq.5와 같이 정의된다.

&#x20;                                          $$R = \mathbb{E}[R_t|S_t=s]$$​                               Eq.5

여기에서 $$\mathbb{E}[X]$$​는 기댓값을 의미한다. 여기에서 기댓값이 등장한 이유는 매번 받는 보상이 다를 수 있기 때문이다. 예를 들어 동전을 뒤집어서 앞면이 나오면 100원을 갖고 뒷면이 나오면 갖지 못한다고 할 때 보상은 앞면 뒷면에 따라 달라지지만 기댓값은 50원으로 일정하다. 이러한 이유로 기댓값을 사용한다. 그렇다면 $$\gamma$$(감쇠인자)는 무엇일까?

$$\gamma$$​(감쇠인자)는 0과 1 사이의 수로 미래에 얻을 수에 곱해짐으로써 그 값을 작게 만드는 파라미터이다. $$\gamma$$​(감쇠인자)가 쓰이는 이유는 우선 효율적인 경로를 찾기 위해서이다. $$\gamma$$​를 곱하여 경로가 길어질수록 보상을 점점 더 작게 만들어 더 효율적인 경로로 이끈다. 그리고 그 다음 이유로는 현재의 보상과 미래의 보상을 다르게 반영하도록 한다. $$\gamma$$​가 1에 가깝다면 미래의 보상과 현재의 보상의 차이가 거의 없으므로 미의 보상을 더 많이 반영하게 된다. 반면 $$\gamma$$가 0에 가깝다면 ​미래의 보상이 거의 0에 가까워 질 것이므로 현재의 보상을 더 많이 반영하게 된다. 그리고 $$\gamma$$​가 0에 가까워 눈앞의 이득만 챙기는 근시안적인 에이전트를 탐욕적(greedy)한 에이전트라고 한다. 또 다른 이유로는 $$\gamma$$​를 0과 1사이의 값으로 만들어 리턴이 무한대가 되는 것을 막기 위해서이다. 그렇다면 리턴이란 무엇일까?

리턴이란 감쇠된 보상의 총 합이다. 상태 $$s_0$$​에서 보상 $$R_0$$​를 받고 시작하여 종료 상태인 $$s_T$$​에 도달할 때 보상 $$R_T$$​를 받으면서 끝이 난다. 이와 같은 하나의 여정을 에피소드(Episode)라고 한다. 또한 이런 표기법을 통해서 리턴(Return) $$G_T$$ 정의할 수 있다. 리턴(Return)은 앞서 말했듯이 시점 T에서부터 미래에 받을 감쇠된 보상의 합으로 Eq.6와 같이 정의된다.

&#x20;                        $$G_T = R_{T+1}+\gamma R_{T+2}+\gamma^2R_{T+3}+\cdots$$​                   Eq.6

Eq.6에서 확인할 수 있듯이 더 미래에 발생한 보상일수록 $$\gamma$$​가 여러 번 곱해진다. 그리고 $$\gamma$$​가 0과 1사이의 값이기 때문에 $$\gamma$$​가 여러 번 곱해질수록 그 값은 0에 점점 더 가까워지게 된다. 따라서 미래에 얻게 될 보상에 비해 현재 얻는 보상에 가중치를 줄 수 있다.

MRP를 더 잘 이해하기 위해 공부에 집중하는 과정을 예를 들자면 앞선 MP에서의 과정에 $$R$$과 $$\gamma$$​를 추가하면 다음과 같다.

![](<../../.gitbook/assets/image (6).png>)

여기에서 감쇠인자 $$\gamma$$​의 값은 0.7이라고 하자.

## 3)MDP(Markov Decision Process)

MDP(Markov Decision Process)는 앞서 설명한 MRP(Markov Reward Function)에 $$A$$(액션 집합)가 추가된 것이다. 앞서 설명한 MP와 MRP는 오로지 환경에 의해서만 상태의 전이가 결정되었다. 하지만 MDP에서는 자신의 의사를 가지 의사결정을 하는 에이전트가 등장한다. 그리고 MDP는 Eq.7과 같이 정의된다.

&#x20;                              $$MDP\equiv(S,A,P,R,\gamma)$$                    Eq.7

MDP에서의 $$S$$​(상태의 집합)는 MP, MRP에서의 $$S$$​와 같다. $$A$$​는 에이전트가 취할 수 있는 액션들을 모아놓은 집합으로 에이전트는 각각의 스텝마다 $$A$$​에서 하나의 액션을 골라 액션을 취하며 그에 따라 다음에 도착하게 될 상태가 달라진다. MDP에서의 $$P$$​(상태전이확률행렬)는 에이전트가 추가되었기 때문에 "현재상태가 $$s$$이고 에이전트가 액션 $$a$$​를 취했을 때 다음 상태 $$s'$$​이 될 확률의 행렬"로 재정의해야 하고 Eq.8로 정의된다.

&#x20;                    $$P^a_{ss'}=ℙ[S_{t+1}=s'|S_t = s,A_t=a]$$                  Eq.8​

MDP에서의 $$R$$(보상 함수)은 마찬가지로 에이전트가 추가되었기 때문에 에이전트가 취하는 액션에 따라 보상이 달라진다. 따라서 $$R$$​은 Eq.9 같이 정의된다.

&#x20;                       $$R^a_s=\mathbb{E}[R_{t+1}|S_t=s,A_t=a]$$                        Eq.9​

감쇠인자 $$\gamma$$​는 MRP에서와 같다.

![](<../../.gitbook/assets/image (9).png>)
