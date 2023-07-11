---
description: Linear Regression의 간단한 예제를 살펴보는 페이지입니다.
---

# Example

## 1) 단순선형회귀

### Example\_1

우선 필요한 라이브러리들을 import해준다.

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt    
```

그리고 선형회귀 할 데이터를 생성한다. 함수 y = 2x + 7에 약간의 오차를 주어 데이터를 생성하였다.   &#x20;

```python
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 2*x + 7 + torch.rand(x.size())-torch.rand(x.size())
```

생성된 데이터는 다음과 같다.

![](<../../.gitbook/assets/Figure 2022-04-28 130359.png>)

가중치와 편향을 우선 0으로 설정하고 바뀔 수 있는 값으로 설정한다. 그리고 optimizer는 StepSize가 0.01인 SGD(확률적 경사하강법)으로 설정해 준다. 확률적 경사하강법은 데이터셋에서 무작위로 샘플을 뽑아서 그 샘플에 대해서만 기울기를 계산한다.&#x20;

학습횟수를 1000번으로 설정하고 학습을 진행한다.  &#x20;

```python
epochs = 1000

for epoch in range(1,epochs+1):
    h = x*W + b
    
    cost = torch.mean((h-y)**2)
    
    if(epoch%100==0):
        print("Epoch : {:4d}, y = {:.4f}x+{:.4f} Cost {:.6f}".format(epoch, W.item(),b.item(), cost.item()))
	
	
    optimzer.zero_grad()
    cost.backward()
    optimzer.step()
```

결과는 다음과 같다.

![](<../../.gitbook/assets/image (7).png>)

y=2x+7과  매우 비슷하게 나온 것을 알 수 있다. &#x20;



### Example2

이번에는 일차함수 대신 선형결합으로 이루어진 이차함수에 대해서 회귀분석을 할 것이다.

아까와 마찬가지로 필요한 라이브러리를 임포트해준다.

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt    
```

그리고 선형회귀 할 데이터를 생성한다. 함수 $$y=2x^2+3x+7$$에 약간의 오차를 주어 데이터를 생성하였다.&#x20;

```python
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 2*x*x+ 3*x + 7 + torch.rand(x.size())-torch.rand(x.size())
```

생성된 데이터는 다음과 같다.

&#x20;&#x20;

![](<../../.gitbook/assets/Figure 2022-05-12 144549.png>)

이번에는 학습횟수를 2000회로 놓고 학습을 진행한다.

```python
W1 = torch.zeros(1,requires_grad=True)
W2 = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimzer = optim.SGD([W1,W2,b],lr = 0.005)

epochs = 2000

for epoch in range(1,epochs+1):
    h = x*x*W1 + x*W2 + b
    
    cost = torch.mean((h-y)**2)
    
    if(epoch%100==0):
        print("Epoch : {:4d}, y = {:.3f}x^2 + {:.3f}x + {:.3f} Cost {:.6f}".format(epoch, W1.item(),W2.item(),b.item(), cost.item()))
	
    optimzer.zero_grad()
    cost.backward()
    optimzer.step()
```

&#x20; 그 결과는 다음과 같다.

&#x20;

![](<../../.gitbook/assets/image (5).png>)

원래 설정했던 함수인 $$y=2x^2+3x+7$$과 매우 비슷하게 나온 것을 알 수 있다.  &#x20;
