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

图压缩旨在应对large scale的图训练任务，通过压缩图的节点个数、labels和节点特征达到减少训练量的作用。

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