<h1 align="center"><a href="https://github.com/AlgoLink/banditrl">banditrl</a></h1>
<p align="center">
    <em>A lightweight contextual bandit &amp; reinforcement learning library</em>
</p>
<p align="center">
    <a href="https://img.shields.io/github/checks-status/AlgoLink/banditrl/main" target="_blank">
        <img src="https://img.shields.io/github/checks-status/AlgoLink/banditrl/main" alt="Test">
    </a>
    <a href="https://img.shields.io/github/downloads/AlgoLink/banditrl/total" target="_blank">
        <img src="https://img.shields.io/github/downloads/AlgoLink/banditrl/total" alt="Downloads">
    </a>
    <a href="https://img.shields.io/github/commit-activity/w/AlgoLink/banditrl" target="_blank">
        <img src="https://img.shields.io/github/commit-activity/w/AlgoLink/banditrl" alt="Commit activity">
    </a>
    <a href="https://img.shields.io/github/stars/AlgoLink/banditrl?style=social" target="_blank">
        <img src="https://img.shields.io/github/stars/AlgoLink/banditrl?style=social" alt="Stars">
    </a>
</p>

# banditrl
A lightweight contextual bandit &amp; reinforcement learning library designed to be used in production Python services.

## 项目简介
本项目的目标是建立一个灵活简单的在线学习库，并且有足够的性能在生产中使用。在许多现实世界的应用中（例如，推荐系统），action的数量和每秒请求的数量可能非常大，所以我们应该非常小心地管理模型存储、action存储和历史请求数据的存储。因为不同系统的存储管理是非常不同的，我们让用户可以定义如何做。
- 模型存储：更新后如何存储模型，如何加载模型
- 历史请求数据存储：如何存储请求，并在我们获得（延迟的）奖励时找到它
- 行动存储：如何添加/删除行动并定义每个行动的一些特殊属性
banditrl只提供了核心的上下文bandit算法，以及一些常见的存储操作（如内存存储/基于Rlite/redis的存储）。

## 技术架构
<img src="resources/art.png">