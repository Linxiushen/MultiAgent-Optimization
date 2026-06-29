# 更新日志

所有对本项目的重要更改都将记录在此文件中。

格式基于[Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- 增强型聚合器功能，添加冲突解决机制
- 智能路由策略，支持基于历史交互的路由
- 记忆压缩的高级算法，提高存储效率
- 分布式协调器的故障转移机制
- `pytest.ini` 配置（`pythonpath`、`testpaths`），统一从仓库根目录运行测试
- `DistributedCoordinator` 新增 `get_available_agents` / `is_agent_available` 接口
- `RecoveryManager` 新增 `notify_admin`、`check_service_health` 等接口并实现单例

### 优化
- 改进了记忆优化算法，添加语义相似度计算
- 优化了路由决策逻辑，提高准确率
- 增强了系统的容错能力
- 重构为标准包结构（`core/`、`memory/`、`distributed/`、`recovery/`、`adapters/`、
  `configs/`、`demos/`、`tests/`），与文档及导入路径保持一致
- 将 CI 工作流移动到 `.github/workflows/`
- 记忆与中文文本处理改用字符级 TF-IDF，并在缺少语义模型/情感词典时优雅降级

### 修复
- 修复了记忆管理中的内存泄漏问题
- 解决了多Agent并发访问时的竞态条件
- 修复了长时间运行导致的性能下降问题
- 修复 `main.py` 无法启动的问题（错误的导入与 `while True: pass` 死循环）
- 修复多处源码损坏（`coordinator.py`、`knowledge_graph.py`、`memory_optimizer.py`
  的缩进/孤儿代码块导致的语法错误）
- 修复 LLM 不可用（缺少 `OPENAI_API_KEY`）时路由与记忆模块的崩溃
- 补全缺失依赖 `networkx`；修正测试套件无法收集/运行的问题（现 125 项全部通过）

## [1.0.0] - 2023-12-15

### 新增
- 初始版本发布
- 基础路由功能
- 简单记忆管理
- 基本的分布式协调
- 响应聚合功能