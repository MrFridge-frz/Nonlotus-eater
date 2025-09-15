---
title: "Graph Condensation: A Survey"
description: "图压缩的学习笔记"
publishDate: "2025-09-13"
tags: ["Astro", "Blog"]
draft: false
comment: true
---

## An overview of Graph condensation

![An overview of Graph Condensation](/images/graph_condensation/Overview.png)

> GC focuses on synthesizing a compact yet highly representative graph, enabling GNNs trained on it to achieve performance comparable to those trained on the original large graph.

图压缩旨在应对large scale的图训练任务，通过压缩图的节点个数、labels和节点特征达到减少训练量的作用。问题建模如下：

### Problem Modeling

1. 数据描述
   
   图数据集：A large-scale dataset $\mathcal{T}=(\mathcal{V}, \mathcal{E})$,$|\mathcal{V}|=N$,$|\mathcal{E}|=M$，包括节点特征矩阵$X\in \mathcal{R}^{N \times d}$,邻接矩阵$A\in \mathcal{R}^{N \times N}$和labels $Y$.
2. 任务要求
   
   图压缩需要找到一个小型压缩图$\mathcal{S}=(\mathcal{V'}, \mathcal{E'})$, 其中$|\mathcal{V'}|=N', N'<<N$，当然也包括压缩后的包括节点特征矩阵$X'\in \mathcal{R}^{N' \times d}$,邻接矩阵$A\in \mathcal{R}^{N' \times N'}$和labels $Y$.
3. 超参数
   1. 压缩率 $\tau = \frac{N'}{N}$

4. 目标函数
    
    图压缩的目的是希望通过一个参数化的中继图模型(relay graph model)$f_{\theta}(\cdot)$，找到一个小但保留信息的图 S，使其在中继模型中的表示接近原图，从而将图压缩的任务转化成一个优化问题：

    $$
    S = \arg\min_S \mathcal{L}_{cond}(f_\theta(T), f_\theta(S))
    $$
    其中$\mathcal{L}_{cond}$是图压缩的优化函数。

Graph Condensation主要分为两部分：Optimization strategies 和 Condensed graph generation。本文的后续部分将从这两部分进行介绍。

## Optimization Strategies

$\mathcal{T}$ 和 $\mathcal{S}$ 的Loss:

$$
\mathcal{L}^T(\theta) = \ell(f_\theta(T), \mathbf{Y}), \\
\mathcal{L}^S(\theta) = \ell(f_\theta(S), \mathbf{Y}'),
$$
其中$\ell$是任务特定的目标，如交叉熵，故图压缩的目标函数可以被塑造成接下来的bi-level问题：
$$
\min_S \mathcal{L}^T(\theta^S) \quad \text{s.t.} \quad \theta^S = \arg\min_\theta \mathcal{L}^S(\theta)
$$
结合公式，Bi-level Optimization的思路应该划分为内外两层：

1. 内循环：只使用压缩图 S 和其标签 Y' 来计算损失 $L^S$，并据此更新中继模型 $f_θ$ 的参数。
2. 外循环：使用原始图 T 和其标签 Y 来计算损失 L^T，并据此更新压缩图 S 的结构（节点特征 X' 和邻接矩阵 A'）。

![Bi-level Optimization](/images/graph_condensation/Algorithm1.png)

在具体的实现上，目前已有的方法包括Gradient Matching, Trajectory Matching, Kernel Ridge Regression(KRR), Distribution Matching四种。

| 优化策略 (Optimization Strategy) | 损失函数 (Loss Function) 与 数学符号解释 (Mathematical Symbols Explained) | 优缺点简述 (Brief Pros & Cons)                                       |
| :------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| **梯度匹配 (Gradient Matching)** | **公式:** $L_{\text{cond}} = \mathbb{E}_{\theta_0 \sim \Theta} \left[ \sum_{t=1}^T D(\nabla_\theta L_T(\theta_t), \nabla_\theta L_S(\theta_t)) \right]$<br>**解释:**<br>- $\mathbb{E}_{\theta_0 \sim \Theta}$: 对中继模型 $f_\theta$ 的初始参数 $\theta_0$ 从分布 $\Theta$ 中采样并求期望，以提高鲁棒性。<br>- $D(\cdot, \cdot)$: 距离度量函数（如余弦相似度或L2距离），用于衡量两个梯度向量的差异。<br>- $\nabla_\theta L_T(\theta_t)$: 在第 $t$ 步、模型参数为 $\theta_t$ 时，在**原始图 $T$** 上计算的任务损失 $L_T$ 对参数 $\theta$ 的梯度。<br>- $\nabla_\theta L_S(\theta_t)$: 在第 $t$ 步、模型参数为 $\theta_t$ 时，在**浓缩图 $S$** 上计算的任务损失 $L_S$ 对参数 $\theta$ 的梯度。<br>- $\theta_{t+1} = \text{opt}(L_S(\theta_t))$: 约束条件，表示模型参数 $\theta$ 仅在浓缩图 $S$ 上通过优化器 $\text{opt}(\cdot)$ 进行更新。 | **优:** 主流方法，效果好。<br>**缺:** 计算开销大，是双层优化。             |
| **轨迹匹配 (Trajectory Matching)** | **公式:** $L_{\text{cond}} = \mathbb{E}_{\theta'_t \sim \Theta'} \left[ D(\theta_{t+T}^T, \theta_{t+L}^S) \right]$<br>**解释:**<br>- $\mathbb{E}_{\theta'_t \sim \Theta'}$: 对从原始图训练轨迹中采样的中间参数（检查点）$\theta'_t$ 从集合 $\Theta'$ 中采样并求期望。<br>- $\theta_{t+T}^T$: 从起点 $\theta'_t$ 开始，在**原始图 $T$** 上再训练 $T$ 步后得到的模型参数。<br>- $\theta_{t+L}^S$: 从**同一个起点 $\theta'_t$** 开始，在**浓缩图 $S$** 上再训练 $L$ 步后得到的模型参数。$T$ 和 $L$ 是控制更新步数的超参数。<br>-$\theta_{t+1}^T = \text{opt}(L_T(\theta_t^T))$ <br>-$\theta_{t+1}^S = \text{opt}(L_S(\theta_t^S))$ | **优:** 匹配更全局信息，性能通常更好。<br>**缺:** 计算开销极大，是三层优化。 |
| **核岭回归 (Kernel Ridge Regression)** | **公式:** $L_{\text{cond}} = \frac{1}{2} \| Y - K_{TS} (K_{SS} + \lambda I)^{-1} Y' \|^2$<br>**解释:**<br>- $K_{TS}$: 核矩阵，其元素 $K_{TS}[i,j]$ 表示原始图中第 $i$ 个样本与浓缩图中第 $j$ 个样本在核空间中的相似度。<br>- $K_{SS}$: 核矩阵，其元素 $K_{SS}[i,j]$ 表示浓缩图中第 $i$ 个样本与第 $j$ 个样本在核空间中的相似度。<br>- $\lambda$: 正则化系数| **优:** 计算高效，有闭式解。<br>**缺:** 核矩阵内存消耗大。     |
| **分布匹配 (Distribution Matching)** | **公式:** $L_{\text{cond}} = \mathbb{E}_{\theta_0 \sim \Theta} \left[ D(f_\theta(T), f_\theta(S)) \right]$<br>**解释:**<br>- $f_\theta(\cdot)$: 中继模型 $f_\theta$ 对图进行编码后得到的特征表示集合。 | **优:** 计算最高效，无梯度计算。<br>**缺:** 通常需类别标签，任务适应性受限。 |

## Condensed Graph Generation

1

## 总结

最后一段总结或个人感想。