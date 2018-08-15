---
layout: post
title: 梯度下降法总结
author: Pekary
date: 2018-08-15-15:00:00 +0800
tags: 优化方法	
---

&emsp;&emsp;这篇文章主要讲一下深度学习中使用的梯度下降法及一些该方法的变种，包括批梯度下降法（Batch Gradient Descent, BGD）、小批量梯度下降法（Mini-Batch Gradient Descent）、带Momentum的梯度下降法（Gradient Descent with Momentum）、RMSprop以及Adam方法。阅读本文需要熟悉前向传播算法和反向传播算法，或者至少知道这两个方法是干什么的。文中的伪代码会利用一些python语言。

&emsp;&emsp;在高等数学中提到过，梯度方向是函数增长最快的方向，而在优化方法中，通常需要求目标函数的最小值的一个近似（直接求最小值通常比较困难），因此沿着梯度的反方向即可以使得目标函数的值减小的最快。所有的梯度下降法都是以这个原理为基础，其他的改进方法都是为了使目标函数能够更快的收敛（严格来说，Mini-Batch Gradient Descent更重要的作用是解决了内存或显存不够的问题）。

&emsp;&emsp;介绍一下本文中使用的变量，parameters是一个存储目标函数参数的字典；grads是存储目标函数参数梯度的字典；$X$为训练数据（$n \times m$的一个矩阵），数据维度是$n$，训练数据大小为$m$；$Y$为$X$的真实标签（$1\times m$的一个矩阵）；epoch_nums为训练的轮数，$m$个训练数据全都被计算过一次梯度为一轮；目标函数为$ J$。

## 批梯度下降法

&emsp;&emsp;批梯度下降法一次迭代使用所有训练数据（一次迭代即参数更新一次），该方法的伪代码如下：

```python
epoch_nums = N
初始化训练参数parameters
for epoch in range(epoch_nums):
    使用X, Y和paramters进行前向传播, 计算损失函数J
    反向传播计算梯度grads
    for key in parameters.keys():
        parameters[key] = parameters[key] - learning_rate * grads[key]
```

这里的learning_rate为称为学习率，它决定了一次下降的步长，通常不会太大，毕竟梯度是一种极限，只决定了一个小领域内这个方向是下降最快的，一步走太大，很有可能走偏。如果不使用dropout策略，批梯度下降法得到的损失函数随迭代次数变化的曲线应该是单调下降的。由于批梯度下降法需要在整个训练数据上进行计算，当遇到较大的数据集时，由于内存（显存）空间的限制，无法一次读入整个训练集，那么就无法使用该方法了，因此小批量梯度下降法出现了。

## 小批量梯度下降法

&emsp;&emsp;小批量梯度下降法把训练数据集分成若干份，每份数据集即为一个小批量，其所含样本数量为batch size。每次迭代只在一个小批量上进行。伪代码如下：

```python
epoch_nums = N
batch_size = 64
初始化参数parameteres
for epoch in range(epoch_nums):
    for iter in range(m // batch_size):
        mb_X = X[:, iter*batch_size:(iter+1)*batch_size]
		mb_Y = Y[:, iter*batch_size:(iter+1)*batch_size]
        使用mb_X, mb_Y和paramters进行前向传播, 计算损失函数J
        反向传播计算梯度grads
        for key in parameters.keys():
            parameters[key] = parameters[key] - learning_rate * grads[key]
    if(m % batch_size != 0):
        mb_X = X[:, (m // batch_size) * batch_size:]
		mb_Y = Y[:, (m // batch_size) * batch_size:]
        使用mb_X, mb_Y和paramters进行前向传播, 计算损失函数J
        反向传播计算梯度grads
        for key in parameters.keys():
            parameters[key] = parameters[key] - learning_rate * grads[key]
```

小批梯度下降法得到的损失函数随迭代次数变化的曲线会波动下降。当batch size为m时，小批量梯度下降法即为批梯度下降法。batch size为1时的小批量梯度下降法叫做随机梯度下降法（Stochastic Gradient Descent, SGD），此方法收敛的比较快，损失函数随迭代次数变化的曲线波动会较大，但是此时向量化（vectorization，指对数据的计算，尽可能的采用矩阵或向量的形式表示，一些库以及GPU对向量化的计算快很多）的优势就没有了。

## 带Momentum的梯度下降法

### 指数移动平均

&emsp;&emsp;股票交易平台通常会提供移动平均这个指标，以股价的变化为例说明指数移动平均（Exponential Moving Average, EMA）。假设第$t$天的股票价格为$w_t$，EMA为$v_t$，$v_1=0$；$\beta$为EMA参数，$0\le \beta \le 1$，其通常取值为$[0.8, 0.999]$，等下会说明$\beta$的意义。那么$v_{k+1}$的计算如下：

$$
\begin{equation}v_{t+1} = \beta v_t + (1 - \beta)w_{t+1}\end{equation}
$$

把这个式子展开可以得到

$$
\begin{equation}v_{t+1}= \sum_{i=0}^t(1-\beta)\beta^t w_{t+1 -i} \end{equation}
$$

可以发现，时间越近的股价权重越高，时间越远的股价权重越小。实际上，$v_{t+1}$近似于计算前$\frac{1}{1 -\beta}$天的股价平均值，如果$\beta = 0.9$，那么即计算了前10天的股价。$\beta$越大，得到的$v_t$随时间变化曲线越平滑。这里存在一个问题，即最开始的几天计算的EMA偏差很大，假设$\beta = 0.9$，$w_1 = 100$， $w_2 = 50$，那么$v_1 = 10$，$v_2=14$，而实际前两天股价的平均值为$75$，这个估计显然不合理，因此需要进行修正。

### 偏差修正

&emsp;&emsp;偏差修正的计算公式如下：
$$
\begin{equation} v_t = \frac{v_t}{1 - \beta^t}\end{equation}
$$
即计算完当天的EMA以后再除以一个修正向，读者可以带入上面的例子算一下。当$t$越大时，由于$0 \le \beta \le 1$，分母会越来越趋近于1，这符合实际情况，因为当$t$较大时已经无需修正。

### Momentum

&emsp;&emsp;梯度下降法中的Momentum即为计算梯度的EMA，初始化v为一个字典，其keys和grads的keys相同，v[keys]为和grads[keys]形状相同的零矩阵，这个初始化在进行训练前执行。这里只给出一次迭代的伪代码：

```python
mb_X = X[:, iter*batch_size:(iter+1)*batch_size]
mb_Y = Y[:, iter*batch_size:(iter+1)*batch_size]
使用mb_X, mb_Y和paramters进行前向传播, 计算损失函数J
反向传播计算梯度grads
for key in v.keys():
    v[key] = beta * v[key] + (1 - beta)*grads[key]
for key in parameters.keys():
    parameters[key] = parameters[key] - learning_rate * v[key]
```

所以Momentum方法使用的梯度是前$\frac{1}{1 - \beta}$梯度的均值，它可以使得优化的方向更加平滑。



## RMSprop

&emsp;&emsp;RMSprop（Root Mean Squared propagation）也计算了一个EMA，不过它计算的是梯度平方的EMA。初始化s为一个字典，其keys和grads的keys相同，s[keys]为和grads[keys]形状相同的零矩阵，这个初始化在进行训练前执行。一次迭代的伪代码如下：

```python
import numpy as np
mb_X = X[:, iter*batch_size:(iter+1)*batch_size]
mb_Y = Y[:, iter*batch_size:(iter+1)*batch_size]
使用mb_X, mb_Y和paramters进行前向传播, 计算损失函数J
反向传播计算梯度grads
for key in s.keys():
    s[key] = beta * s[key] + (1 - beta)*(grads[key]**2)
for key in parameters.keys():
    parameters[key] = parameters[key] - learning_rate * grads[key]/np.sqrt(s[key] + epsilon)
```

从参数更新公式可以看到，如果之前到梯度太大，除以$\text{np.sqrt(s[key])}$会使得本次更新的步子更小一点；反之，本次更新的步子会更大一点。注意，为了防止除以0的情况，代码中加入了一个epislon常量，通常设置为$10^{-8}$。

## Adam

&emsp;&emsp;Adam方法结合了RMSprop和Momentum，其效果通常是所提到的优化方法中最好的。一次迭代的伪代码如下：

```python
import numpy as np
mb_X = X[:, iter*batch_size:(iter+1)*batch_size]
mb_Y = Y[:, iter*batch_size:(iter+1)*batch_size]
使用mb_X, mb_Y和paramters进行前向传播, 计算损失函数J
反向传播计算梯度grads
for key in s.keys():
    v[key] = beta1 * v[key] + (1 - beta1)*(v[key]**2)
    v[key] /= (1 - beta1 ** t)  # t 为当前迭代次数
    s[key] = beta2 * s[key] + (1 - beta2)*(grads[key]**2)
    s[key] /= (1 - beta2 ** t)  # t 为当前迭代次数
for key in parameters.keys():
    parameters[key] = parameters[key] - learning_rate * v[key]/np.sqrt(s[key] + epsilon)
```

注意Adam中使用了偏差修正，通常beta1的值设为0.9， beta2的值设置为0.999。



**Reference:**

&emsp;&emsp;Cousera课程：Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization