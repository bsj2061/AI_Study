---
description: 로지스틱 회귀(Logistic Regression)의 간단한 예제에 대해 알아보는 페이지입니다.
---

# Example

optimizer = torch.optim.SGD(\[W,b], lr=0.001)

loss = nn.BCELoss()우선 필요한 라이브러리를 임포트해준다.

```python
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
```

이후 로지스틱 회귀에 필요한 함수들을 정의해준다. 그리고 데이터를 생성한다.  train셋과 test셋을 구분해서 데이터를 생성하였다. &#x20;

```python
def sigmoid(x):            #시그모이드 함수 
    return 1/(1+torch.exp(-x))

def z(W, x, b):
    return W*x + b

def gendata(n, W, b):       #데이터 생성 함수
    x = np.random.uniform(-25, 25, (n, 2))
    x2_hat = z(W,b, x[:,0])
    y = x[:,1] > x2_hat
    return x, y.astype(int)

x, y = gendata(100, 1., 0.5)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float().unsqueeze(1)
x_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

```

이렇게 생성된 데이터는 다음과 같다.

![](<../../.gitbook/assets/Figure 2022-05-12 212954.png>)

선형분류 문제에서는 데이터를 두개의 그룹으로 잘 나눌 수 있는 a \* x + b를 데이터 기반으로 구하는 문제이다. 이때 데이터의 레이블 즉 y값은 0이나 1값이 될것이다. (빨간 점선은 $$y=x+0.5$$이다. )

```python
W = torch.zeros((2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
```

&#x20;이후 optimizer와 loss function을 설정해준다. optimizer는 SGD(확률적 경사하강법)를 사용하였고, loss function은 crossentropy 중에서도 이진분류에 관한 crossentropy인 BCELoss() fuction을 사용하였다. &#x20;

\*BCELoss() document: [https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

```python
optimizer = torch.optim.SGD([W,b], lr=0.001)
loss = nn.BCELoss()
```

그리고 epochs를 2000으로 두고 학습을 시작하였다. Hypothesis는 Logistic Regression의 Concept부분에 있는 Eq.3과 같다.

```python
epochs = 2000 
for epoch in range(1,epochs + 1):

    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{}'.format(epoch, epochs))

```

Train data set에 대한 accuracy를 구하기 위해 다음과 같은 코드를 써줬다. Hypothesis가 0.5보다 작으면 False(0), 0.5보다 크거나 같으면 True(1)이 되기 때문에 이를 구한 후 y\_train과 비교하여 accuracy를 구하였다.

```python
prediction = hypothesis >= torch.FloatTensor([0.5])
correct_prediction = prediction.float() == y_train
accuracy = correct_prediction.sum().item() / len(correct_prediction)
print('The model has an accuracy of {:2.2f}% for the training set.'.format(accuracy * 100))
```

실행결과는 다음과 같다.

![](<../../.gitbook/assets/image (16).png>)

Train data set에 대하여 99.01%의 정확도를 가지고 있음을 확인할 수 있다.

## 참고문헌

[https://ratsgo.github.io/deep%20learning/2017/09/24/loss/](https://ratsgo.github.io/deep%20learning/2017/09/24/loss/)

[https://yamalab.tistory.com/94](https://yamalab.tistory.com/94)

&#x20; &#x20;
