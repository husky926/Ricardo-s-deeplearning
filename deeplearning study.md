# Task2 study

## torch中的一些函数

```python
torch.arange(z,x,y)#生成z个x行y列的张量
torch.tensor([X,Y])
torch.zeros()
torch.zeros_like(A)#创建一个与A相同的全零tensor
torch.ones()
torch.abs(input,out = None)#计算张量的每个元素绝对值
torch.acos(input,out =None)#返回新张量，包括输入张量每个元素的反余弦,out(Tensor,可选)
torch.asin(input，out=None)#正弦
torch.add(input,value,out=None)#每个值加上value
torch.div(input,other,out=None)#input对应除以other
torch.from_numpy(A)#numpy转换成tensor
```

**梯度计算**

```python
#将其属性.requires_grad设置为True，则将开始追踪在其上的所有操作，完成后，可以调用.backward()来完成所有梯度的计算.此Tensor的梯度将累积到.grad属性中
import torch

def my_grad():
	x = torch.ones(2,2,requires_grad = True)
    y = x + 2
    z = y * y *3
    z_mean = z.mean()
    z_mean.backward()
```

.backward(),如果当前的tensor是标量，则不需要为backward传入参数

否则需要一个与当前tensor同行的tensor。

grad在反向传播的过程中是累加的，意味着每一次运行反向传播，梯度都会累加之前的梯度，所以需要清零

```python
x.grad.data_zero_()
```

**详解.backward()**

浅析

```python

x = torch.tensor([2, 3, 4], dtype=torch.float, requires_grad=True)
print(x)
y = x * 2
while y.norm() < 1000:
    y = y * 2
print(y)

y.backward(torch.ones_like(y))
print(x.grad)

```

结果：

```python
tensor([2., 3., 4.], requires_grad=True)
tensor([ 512.,  768., 1024.], grad_fn=<MulBackward0>)
tensor([256., 256., 256.])
```

参数：**grad_tensors**(查看需要记录梯度的参数的梯度值)





讨论两种情况

```python
x = torch.tensor([2,2])

#Case 1
y = x * x
y.sum().backward()

#Case 2
y = torch.dot(x,x)
y.backward()
```

这两种情况求出的x的梯度是一致的，区别在于，第一种情况相当于

X = [x0,x1] ,Y =X + 2 = [x0 * x1, x1  * x0]

如果直接对X求导相当于矩阵对矩阵求导，无法计算

所以需要y.sum() 变成 Y = x0 * x1 + x1*x0(其实此处改为加法更加直观)



