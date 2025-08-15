# 分布式协调器故障转移机制

## 概述

故障转移机制是分布式协调器的核心功能之一，用于处理Agent故障时的任务重分配和资源释放，确保系统的高可用性和容错性。本文档详细介绍了故障转移机制的设计、配置和使用方法。

## 功能特性

1. **Agent故障检测**：通过心跳机制检测Agent是否存活，超时未收到心跳的Agent将被标记为不活跃。
2. **任务重分配**：当Agent故障时，自动将其未完成的任务重新分配给其他可用的Agent。
3. **资源锁释放**：释放故障Agent持有的所有资源锁，防止资源死锁。
4. **故障事件记录**：记录所有故障事件，包括时间、Agent信息、故障类型和详细信息。
5. **故障通知**：支持多种通知方式，包括日志、邮件、Webhook和系统消息。
6. **高级故障转移策略**：提供多种故障转移策略，包括基于可靠性的策略和备份Agent策略。

## 故障转移策略

### 1. 基础故障转移策略

基础故障转移策略是系统默认的故障转移方式，主要包括：

- **备份Agent机制**：为关键Agent指定备份Agent，当主Agent故障时，任务自动转移到备份Agent。
- **基于能力的选择**：根据任务类型和Agent能力选择合适的替代Agent。
- **基于负载的选择**：优先选择负载较低的Agent进行任务重分配。

### 2. 高级故障转移策略

#### FailoverStrategy

`FailoverStrategy`是一种基于多种因素综合考量的高级故障转移策略，考虑因素包括：

- **任务优先级**：高优先级任务优先获得资源和重分配。
- **Agent可靠性**：基于历史故障记录评估Agent的可靠性。
- **能力匹配度**：评估Agent处理特定任务类型的能力。
- **当前负载**：考虑Agent当前的任务负载情况。
- **历史性能**：考虑Agent历史任务执行的性能表现。

#### BackupAgentStrategy

`BackupAgentStrategy`是一种预先指定备份的故障转移策略，主要特点：

- **主备Agent指定**：为每个任务指定主Agent和多个备份Agent。
- **自动故障切换**：当主Agent故障时，自动切换到备份Agent。
- **备份优先级**：支持设置备份Agent的优先级顺序。
- **动态备份更新**：根据系统状态动态调整备份Agent列表。

## 配置说明

故障转移机制的配置位于`configs/distributed_config.json`文件中的`failover_config`部分：

```json
{
    "failover_config": {
        "enabled": true,                      // 是否启用故障转移
        "priority_based_reassignment": true,  // 是否基于优先级重分配任务
        "preserve_task_priority": true,       // 是否保留任务原优先级
        "max_reassignment_attempts": 3,       // 最大重分配尝试次数
        "critical_agents": [],                // 关键Agent列表
        "backup_agents": {},                  // 备份Agent配置
        "use_failover_strategy": true,        // 是否使用高级故障转移策略
        "failover_strategy_type": "failover", // 故障转移策略类型：failover或backup_agent
        "backup_strategy_count": 2,           // 备份策略中每个任务的备份Agent数量
        "reliability_threshold": 0.8,         // 可靠性阈值
        "task_priority_levels": {             // 任务优先级级别配置
            "high": 8,
            "medium": 5,
            "low": 2
        },
        "failure_cooldown_period": 300        // 故障冷却期（秒）
    }
}
```

## 故障事件记录

系统会记录所有故障事件，包括以下信息：

- **时间戳**：故障发生的时间
- **Agent ID**：发生故障的Agent标识
- **故障类型**：如心跳超时、任务执行失败等
- **详细信息**：故障的详细描述和相关数据

故障历史记录可通过`get_failure_history()`方法获取，也可以在集群状态中查看。

## 故障通知

系统支持多种故障通知方式，可在配置文件中设置：

```json
{
    "failure_notification": {
        "enabled": true,
        "notification_types": ["log", "email", "webhook", "system"],
        "min_severity": "warning"
    },
    "email_notification": {
        "recipients": ["admin@example.com"],
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "username": "notification@example.com",
        "password": "password"
    },
    "webhook_notification": {
        "url": "https://example.com/webhook",
        "headers": {"Content-Type": "application/json"},
        "method": "POST"
    }
}
```

## 使用示例

### 基本使用

```python
from distributed.coordinator import DistributedCoordinator

# 创建协调器实例
coordinator = DistributedCoordinator()

# 注册Agent
coordinator.register_agent('agent1', 'worker', ['task_type1', 'task_type2'], 'http://agent1')
coordinator.register_agent('agent2', 'worker', ['task_type1', 'task_type3'], 'http://agent2')

# 创建任务
task = {
    'id': 'task1',
    'type': 'task_type1',
    'priority': 5,
    'data': {'input': 'data1'}
}

# 分配任务
assigned_agent = coordinator.assign_task(task)
print(f"任务分配给 Agent {assigned_agent}")

# 获取故障历史
failure_history = coordinator.get_failure_history()
for event in failure_history:
    print(f"故障事件: {event}")

# 获取集群状态
status = coordinator.get_cluster_status()
print(f"活跃Agent: {status['active_agents']}")
print(f"不活跃Agent: {status['inactive_agents']}")
print(f"故障事件数: {status['failure_count']}")
```

### 使用高级故障转移策略

```python
from distributed.coordinator import DistributedCoordinator
from distributed.failover_strategy import FailoverStrategy, BackupAgentStrategy

# 创建协调器实例
coordinator = DistributedCoordinator()

# 配置使用FailoverStrategy
coordinator.config['failover_config']['use_failover_strategy'] = True
coordinator.config['failover_config']['failover_strategy_type'] = 'failover'
coordinator.init_failover_strategy()

# 或者使用BackupAgentStrategy
coordinator.config['failover_config']['failover_strategy_type'] = 'backup_agent'
coordinator.init_failover_strategy()
```

## 测试

系统提供了多个测试用例，用于验证故障转移机制的正确性：

- `tests/test_failover.py`：测试基本故障转移功能
- `tests/test_failover_strategy.py`：测试高级故障转移策略
- `tests/test_failover_integration.py`：集成测试

运行测试：

```bash
python -m unittest tests/test_failover.py
python -m unittest tests/test_failover_strategy.py
python -m unittest tests/test_failover_integration.py
```

## 演示

系统提供了一个演示脚本，用于展示故障转移机制的工作流程：

```bash
python examples/failover_demo.py
```

该演示会创建多个Agent和任务，并模拟Agent故障，展示故障转移过程。

## 最佳实践

1. **合理配置心跳间隔**：根据网络状况和系统负载调整心跳间隔，避免误判。
2. **设置适当的重分配次数**：避免无限重分配导致系统资源浪费。
3. **为关键任务指定备份Agent**：确保重要任务在主Agent故障时能快速转移。
4. **定期检查故障历史**：分析故障模式，优化系统配置。
5. **配置多种通知方式**：确保故障能及时被发现和处理。

## 故障排查

1. **Agent频繁被标记为不活跃**：检查网络连接和心跳间隔设置。
2. **任务重分配失败**：检查是否有足够的可用Agent和匹配的能力。
3. **资源锁未释放**：检查资源锁超时设置和释放逻辑。
4. **通知未发送**：检查通知配置和网络连接。

## 总结

故障转移机制是保证分布式系统高可用性的关键组件。通过合理配置和使用，可以有效应对各种故障情况，确保系统的稳定运行。