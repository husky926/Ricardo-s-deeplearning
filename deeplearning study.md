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
torch.mm(a,b)#矩阵ab相乘
tensor.view() = tensor.reshape#view方法进行形状重构需要保证原始tensor在内存空间中分布连续,reshape不需要
```

### **梯度计算**

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

.backward()

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

```python
torch.detach()
```

分离，返回一个新的tensor



# task3

## softmax回归

```python
x.sum(0,keepdim=True)
```

 如果keepdims这个参数为True，被删去的维度在结果矩阵中就被设置为一。 

```PYTHON
torch.exp(x)
```

相当于e的x次方，如果是张量，结果就是e次方的张量

```python
torch.shape(0/1)
```

0是行数，1是列数

```python
#训练模型
model.train()
#计算预测正确的数量
model.eval()
#建立神经网络
torch.nn.Module
#比较类型
isinstance(a,b)
```

建立神经网络



# task4

## 感知机

常见激活函数

**ReLU函数**

 ![../_images/output_mlp_76f463_21_0.svg](https://zh-v2.d2l.ai/_images/output_mlp_76f463_21_0.svg) 

易知，x大于零的时候，函数的导数为一，小于零的时候导数为零

使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题 

**sigmoid函数**

 ![../_images/output_mlp_76f463_51_0.svg](https://zh-v2.d2l.ai/_images/output_mlp_76f463_51_0.svg) 

**tanh函数**

 与sigmoid函数类似， tanh（双曲正切）函数也能将其输入压缩转换到区间（-1， 1)上 

 ![../_images/output_mlp_76f463_81_0.svg](https://zh-v2.d2l.ai/_images/output_mlp_76f463_81_0.svg) 

## 多层感知机的简易实现

```python
torch.nn.Linear(in_features,#输入的神经元个数
               out_fearutes,#输出的神经元个数
               bias=True)#是否包含偏置
```

定义了一个神经网络的线性层，实则就是对输入的X(n*i)执行了一个**线性变换**
$$
Y 
n×o
​
 =X 
n×i
​
 W 
i×o
​
 +b
$$
**损失函数**(交叉熵损失函数)

用于解决多分类问题，也可以用于二分类

```python
nn.CrossEntropyLoss(x,lable)
```

- 第一个参数：x为输入也是网络的**最后一层的输出**，其shape为[batchsize,class]（函数要求第一个参数，也就是最后一层的输出为二维数据，每个向量中的值为不同种类的概率值）
- 第二个参数：是传入的标签，也就是某个类别的索引值，在上面公式没有参与计算。batch_size如果是1，那么就只有一个数字，0，1或者2，表示的就是此时这个样本对应的真实类别，如果为batch_size是2，那么就有两个数字，例如（0，1），分别表示这两条样本对应的真实类别。

$$
loss(x,class)=−log( 
∑ 
j
​
 exp(x[j])
exp(x[class])
​
 )=−x[class]+log( 
j
∑
​
 exp(x[j]))
$$

也可以加上权重
$$
loss(x,class)=weight[class](−x[class]+log( 
j
∑
​
 exp(x[j])))
$$
 **例如**，输入为x=[[4,8,3]]，shape=(1,3),即batchsize=1，class=3 

```python
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()

      	self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), 
            nn.ReLU(True)，
            nn.Linear(n_hidden_1, n_hidden_2)，
            nn.ReLU(True)，
            # 最后一层不需要添加激活函数
            nn.Linear(n_hidden_2, out_dim)
             )

  	def forward(self, x):
      	x = self.layer(x)
      	return x


```

 一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。 相当于搭建神经网络

### 优化器**

```
torch.optimizer()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)
```

主要介绍加速神经网络训练

#### SGD

 SGD是最基础的优化方法，普通的训练方法, 需要重复不断的把整套数据放入神经网络NN中训练, 这样消耗的计算资源会很大.当我们使用SGD会把数据拆分后再分批不断放入 NN 中计算. 每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了 NN 的训练过程, 而且也不会丢失太多准确率. 

我们把这个人从平地上放到了一个斜坡上, 只要他往下坡的方向走一点点, 由于向下的惯性, 他不自觉地就一直往下走, 走的弯路也变少了. 这就是 Momentum 参数更新。 

#### Momentum

 Momentum 传统的参数 W 的更新是把原始的 W 累加上一个负的学习率(learning rate) 乘以校正值 (dx). 此方法比较曲折。 

#### RMSProp

 RMSProp 有了 momentum 的惯性原则 , 加上 adagrad 的对错误方向的阻力, 我们就能合并成这样. 让 RMSProp同时具备他们两种方法的优势. 不过细心的同学们肯定看出来了, 似乎在 RMSProp 中少了些什么. 原来是我们还没把 Momentum合并完全, RMSProp 还缺少了 momentum 中的 这一部分. 所以, 我们在 Adam 方法中补上了这种想法. 

## Adam

Adam 计算m 时有 momentum 下坡的属性, 计算 v 时有 adagrad 阻力的属性, 然后再更新参数时 把 m 和 V 都考虑进去. 实验证明, 大多数时候, 使用 adam 都能又快又好的达到目标, 迅速收敛. 所以说, 在加速神经网络训练的时候, 一个下坡, 一双破鞋子, 功不可没.

```python
# SGD 就是随机梯度下降
opt_SGD= torch.optim.SGD(net_SGD.parameters(), lr=LR)
# momentum 动量加速,在SGD函数里指定momentum的值即可
opt_Momentum=torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
# RMSprop 指定参数alpha
opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
# Adam 参数betas=(0.9, 0.99)
opt_Adam=torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
```



## AdaGrad 优化学习率

AdaGrad 优化学习率，使得每一个参数更新都会有自己与众不同的学习率。与momentum类似，不过不是给喝醉酒的人安排另一个下坡, 而是给他一双不好走路的鞋子, 使得他一摇晃着走路就脚疼, 鞋子成为了走弯路的阻力, 逼着他往前直着走.

**随机梯度下降**

```python
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

- params：要训练的参数，一般我们传入的都是`model.parameters()`。

- lr：learning_rate学习率，会梯度下降的应该都知道学习率吧，也就是步长。

-  weight_decay(权重衰退）和learning_rate（学习率）的区别 

  ```
  lr是我们所熟知的更新权重的方式，假设学习参数为θ，学习率由γ表示，梯度均值为g，θ(t) = θ(t-1) - γg(t)
  weight_decay是在L2正则化理论中出现的概念。
  L2范数也被称为“权重衰减”和“岭回归”，L2的主要作用是解决过拟合，L2范数是所有权重的平方开方和，最小化L2范数，使得权重趋近于0，但是不会为0。那么为什么参数的值小，就能避免过拟合。模型的参数比较小，说明模型简单，泛化能力强。参数比较小，说明某些多项式分支的作用很小，降低了模型的复杂程度。其次参数很大，一个小的变动，都会产生很大的不同。所以采用L2正则化可以很好地解决过拟合的问题。
  ```

   在损失函数中加入L2正则化项后（大致）变为 

$$
J(θ)= 
1/
2m
​
 [ 
i=1
∑
m
​
 (h 
θ
​
 (x 
(i)
 )−y 
(i)
 ) 
2
 +λ 
j=1
∑
n
​
 θ 
j
2
​
 ]
$$

后面一项就是正则化项，λ为weight_decay

也就是说正则化的实现是在原梯度的基础上加上一个系数乘以可学习参数值作为新梯度 

![1680002241955](C:\Users\Ricardo\AppData\Roaming\Typora\typora-user-images\1680002241955.png)

SGD有一个缺点，起更新方向完全依赖于当前的batch，因此其更新十分不稳定

## 模型选择

### 权重衰减

在训练参数化机器学习模型时， *权重衰减*（weight decay）是最广泛使用的正则化的技术之一 

也称为正则化， 这项技术通过函数与零的距离来衡量函数的复杂度 

 **最常用方法是将其范数作为惩罚项加到最小化损失的问题中** 

![1680006288271](C:\Users\Ricardo\AppData\Roaming\Typora\typora-user-images\1680006288271.png)

增加惩罚权重后的

![1680006429144](C:\Users\Ricardo\AppData\Roaming\Typora\typora-user-images\1680006429144.png)

参数的更新公式

![1680006535818](C:\Users\Ricardo\AppData\Roaming\Typora\typora-user-images\1680006535818.png)

### 暂退法