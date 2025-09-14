---
title: "Graph Condensation: A Survey"
description: "图压缩的学习笔记"
publishDate: "2025-09-13"
tags: ["Astro", "Blog"]
draft: false
comment: true
---

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

详细说明某个知识点或内容。  
可以换行，也可以**加粗**、*斜体*，或者插入[链接](https://example.com)。

- 列表项一
- 列表项二

### 更小的标题

1. 有序列表项
2. 第二项

> 引用内容，可以用来强调观点。

```js
// 代码块示例
console.log('Hello Markdown!');
```

## Condensed Graph Generation

## 总结

最后一段总结或个人感想。