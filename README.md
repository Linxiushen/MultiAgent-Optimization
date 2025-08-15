# 多Agent智能分流 + 长程记忆优化系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green)](https://openai.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange)](https://github.com/facebookresearch/faiss)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-red)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.0-lightgrey)](https://flask.palletsprojects.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-2.2.0-blueviolet)](https://www.sbert.net/)

## 项目概述

这是一个基于多Agent架构的智能分流系统，结合长程记忆优化技术，旨在提升AI系统的响应质量和效率。

系统通过智能路由将用户查询分配给最适合的Agent处理，并利用长程记忆系统存储和检索相关信息，以提供更准确和个性化的回答。同时，系统还具备分布式协调和恢复管理能力，确保高可用性和容错性。

### 系统架构

![系统架构图](docs/architecture.svg)

*系统架构图展示了各模块之间的交互关系*

## 核心功能

1. **多Agent智能分流**：
   - 系统包含多种类型的Agent（技术Agent、创意Agent、记忆Agent等），每种Agent专门处理特定类型的任务
   - 通过Router模块智能判断用户查询的类型，并将其路由到最适合的Agent
   - 支持多种路由策略，包括直接路由和记忆增强路由

2. **长程记忆优化**：
   - 使用FAISS向量数据库存储和检索记忆
   - 通过MemoryOptimizer模块优化记忆管理，包括：
     - 计算记忆重要性（考虑时间衰减、访问频率等因素）
     - 压缩相似记忆（基于TF-IDF向量化和余弦相似度）
     - 遗忘低优先级记忆
   - 支持定期优化记忆存储，提高检索效率

3. **分布式协调**：
   - 通过DistributedCoordinator管理多个Agent的注册、注销和心跳
   - 支持多种负载均衡策略（轮询、随机、最少连接、基于性能、基于任务类型）
   - 提供任务分配、消息传递和资源锁机制

4. **恢复管理**：
   - 通过RecoveryManager实现系统故障恢复
   - 支持检查点机制和状态恢复

5. **响应聚合**：
   - Aggregator模块负责融合多个Agent的响应
   - 支持多种融合策略（加权合并、置信度选择、简单合并）
   - 生成包含相关历史信息的最终响应

## 项目结构

```
多Agent智能分流 + 长程记忆优化方案/
├── configs/                 # 配置文件
│   ├── router_config.json    # 路由系统配置
│   ├── memory_config.json    # 记忆系统配置
│   ├── recovery_config.json  # 故障恢复配置
│   ├── distributed_config.json  # 分布式协调配置
│   └── memory_optimizer_config.json  # 记忆优化器配置
├── core/                    # 核心模块
│   ├── __init__.py
│   ├── router.py             # 路由核心实现
│   └── aggregator.py         # 跨Agent信息融合核心
├── memory/                  # 记忆系统
│   ├── __init__.py
│   ├── memory_manager.py     # 记忆管理核心
│   ├── memory_item.py        # 记忆项定义
│   ├── memory_optimizer.py   # 记忆优化器
│   └── memory_optimizer_enhanced.py   # 增强版记忆优化器
├── recovery/                # 故障恢复系统
│   ├── __init__.py
│   └── recovery_manager.py   # 故障恢复核心
├── distributed/             # 分布式协调系统
│   ├── __init__.py
│   └── coordinator.py        # 分布式协调核心
├── adapters/                # 适配器
│   ├── __init__.py
│   └── llm.py               # LLM适配层
├── demos/                   # 演示脚本
│   ├── cli_assistant.py      # 命令行助手演示
│   └── aggregator_demo.py    # 聚合器演示
├── tests/                   # 测试文件
│   ├── test_router.py        # 路由系统测试
│   ├── test_memory_manager.py  # 记忆系统测试
│   ├── test_memory_optimizer.py  # 记忆优化器测试
│   ├── test_distributed_coordinator.py  # 分布式协调测试
│   ├── test_recovery_manager.py  # 故障恢复测试
│   └── performance_test.py   # 性能测试
├── run_tests.py             # 测试运行脚本
├── run_tests_direct.py      # 直接测试运行脚本
├── simple_test.py           # 简单测试脚本
├── run_test_fix.bat         # 测试修复批处理
├── run_test_final.py        # 最终测试运行脚本
└── test_runner_gui.py       # 测试运行GUI程序
```

## 使用指南

### 环境要求
- Python 3.8+
- 相关依赖库（详见 requirements.txt）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
```bash
python main.py
```

## 技术亮点

### 1. 智能冲突解决机制
- 实现了多种冲突解决策略（多数投票、置信度加权、语义分析、源可靠性）
- 能够区分事实性冲突和观点性冲突，并采用不同的解决方案
- 支持用户反馈机制，不断优化冲突解决效果

### 2. 高级路由策略
- 基于历史交互的智能路由，能够学习用户偏好
- 支持多维度路由决策（查询类型、历史表现、Agent专长）
- 动态负载均衡，确保系统资源最优利用

### 3. 记忆优化技术
- 语义相似度计算，实现智能记忆压缩
- 基于重要性评分的记忆管理，优先保留关键信息
- 支持记忆检索增强生成（RAG），提升响应质量

## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

- 提交bug报告和功能请求
- 提交代码改进和新功能
- 改进文档和示例
- 分享使用经验和最佳实践

详细贡献流程请参阅[CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

### 运行测试
```bash
python run_tests.py
```

### 运行聚合器演示
```bash
python demos/aggregator_demo.py
```

### 运行综合演示
```bash
python demos/comprehensive_demo.py
```

## 文档资源

为了更好地理解和使用本系统，我们提供了以下文档资源：

- [技术文档](technical_documentation.md) - 详细的技术实现说明
- [项目归档总结](project_archive_summary.md) - 完整的项目归档信息
- [详细优化计划](detailed_optimization_plan.md) - 包含具体实施步骤的详细优化方案

## 系统架构

![系统架构图](docs/architecture.svg)<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/0583af57-464b-4266-8bab-a9eafb86acef" />


## 演示

- [系统功能演示](demos/system_demo.py) - 展示系统核心功能的交互式演示脚本

## 社区参与

我们欢迎社区的贡献和参与：

- [贡献指南](CONTRIBUTING.md) - 了解如何为项目做贡献
- [报告问题](.github/ISSUE_TEMPLATE/bug_report.md) - 报告您发现的bug
- [功能建议](.github/ISSUE_TEMPLATE/feature_request.md) - 提出新功能建议

## 后续工作
1. 实现更复杂的路由策略
2. 优化记忆系统的性能
3. 增强故障恢复机制的智能化水平
4. 完善分布式协调功能
5. 优化跨Agent信息融合策略
6. 进一步完善增强版记忆优化器的功能

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

## 联系作者

「林修」微信：LinXiu230624
