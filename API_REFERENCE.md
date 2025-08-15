# API参考文档

本文件提供了系统核心组件的API参考，包括类、方法、参数说明和使用示例。

## 1. 分布式协调系统

### 1.1 DistributedCoordinator 类

分布式Agent协调器，负责管理Agent注册、任务分配和通信。采用单例模式实现。

#### 核心方法

```python
# 获取协调器实例
get_instance() -> DistributedCoordinator
```

```python
# 注册新Agent到协调器
register_agent(agent_id: str, agent_type: str, capabilities: List[str], endpoint: str) -> bool
```

```python
# 从协调器注销Agent
unregister_agent(agent_id: str) -> bool
```

```python
# 更新Agent心跳
heartbeat(agent_id: str) -> bool
```

```python
# 分配任务给合适的Agent
assign_task(task_type: str, task_data: Dict, priority: int = 1, callback: Optional[Callable] = None) -> Optional[str]
```

```python
# 在Agent之间发送消息
send_message(from_agent: str, to_agent: str, message_type: str, content: Dict) -> bool
```

```python
# 获取Agent的消息队列
get_messages(agent_id: str) -> List[Dict]
```

```python
# 获取资源锁
acquire_lock(resource_id: str, agent_id: str, timeout: float = 10.0) -> bool
```

```python
# 释放资源锁
release_lock(resource_id: str, agent_id: str) -> bool
```

```python
# 设置负载均衡策略
set_strategy(strategy_name: str) -> bool
```

### 1.2 分布式策略

系统支持多种负载均衡策略，用于优化任务分配。

#### 策略基类

```python
class LoadBalancingStrategy:
    """负载均衡策略基类"""
    def select_agent(self, agent_info: Dict[str, Dict]) -> str:
        """选择一个Agent来执行任务

        Args:
            agent_info: Agent信息字典，键为Agent ID，值为Agent属性字典

        Returns:
            str: 选中的Agent ID
        """
        pass
```

#### 具体策略类

```python
class RoundRobinStrategy(LoadBalancingStrategy):
    """轮询策略：依次选择每个Agent"""

class RandomStrategy(LoadBalancingStrategy):
    """随机策略：随机选择一个Agent"""

class LeastConnectionsStrategy(LoadBalancingStrategy):
    """最少连接策略：选择任务计数最少的Agent"""

class PerformanceBasedStrategy(LoadBalancingStrategy):
    """基于性能的策略：选择性能分数最高的Agent"""

class TaskTypeBasedStrategy(LoadBalancingStrategy):
    """基于任务类型的策略：根据Agent类型和任务类型的匹配度选择Agent"""
```

#### 策略工厂

```python
class StrategyFactory:
    """策略工厂，用于创建不同类型的负载均衡策略"""
    def create_strategy(self, strategy_name: str) -> LoadBalancingStrategy:
        """创建负载均衡策略实例

        Args:
            strategy_name: 策略名称

        Returns:
            LoadBalancingStrategy: 负载均衡策略实例

        Raises:
            ValueError: 如果策略名称无效
        """
        pass
```

### 1.3 配置示例

分布式配置文件 `configs/distributed_config.json` 示例：

```json
{
    "agent_heartbeat_interval": 30,
    "task_timeout": 300,
    "load_balancing_strategy": "round_robin",
    "max_retries": 3
}
```

### 1.4 使用示例

参见 `examples/distributed_strategy_example.py` 文件，该示例展示了如何注册Agent、切换负载均衡策略和分配任务。

## 2. 路由系统 (router)

### Router类
```python
class Router:
    """智能路由系统核心类"""
    def __init__(self, config_path=None):
        """初始化路由器
        Args:
            config_path: 配置文件路径
        """
        
    def route(self, request, agents):
        """根据请求和Agent状态分配最优Agent
        Args:
            request: 请求对象
            agents: Agent列表
        Returns:
            最优Agent
        """
        
    def calculate_score(self, request, agent):
        """计算Agent处理请求的得分
        Args:
            request: 请求对象
            agent: Agent对象
        Returns:
            得分 (float)
        """
        
    def load_config(self, config_path):
        """加载配置
        Args:
            config_path: 配置文件路径
        """
```

### RouteStrategy类
```python
class RouteStrategy:
    """路由策略基类"""
    def calculate(self, request, agent):
        """计算路由得分
        Args:
            request: 请求对象
            agent: Agent对象
        Returns:
            得分 (float)
        """

class LoadBalancingStrategy(RouteStrategy):
    """负载均衡路由策略"""
    
class PriorityBasedStrategy(RouteStrategy):
    """基于优先级的路由策略"""
    
class SkillMatchingStrategy(RouteStrategy):
    """技能匹配路由策略"""
```

## 记忆系统 (memory)

### MemoryManager类
```python
class MemoryManager:
    """记忆管理核心类"""
    def __init__(self, config_path=None):
        """初始化记忆管理器
        Args:
            config_path: 配置文件路径
        """
        
    def store_memory(self, memory_item):
        """存储记忆
        Args:
            memory_item: 记忆项对象
        """
        
    def retrieve_memory(self, query, top_k=5):
        """检索相关记忆
        Args:
            query: 查询文本
            top_k: 返回的最大结果数
        Returns:
            记忆项列表
        """
        
    def forget_memory(self, memory_id):
        """删除记忆
        Args:
            memory_id: 记忆ID
        """
        
    def optimize_memories(self):
        """优化记忆（压缩相似记忆，删除低优先级记忆）"""
        
    def get_memory_stats(self):
        """获取记忆统计信息
        Returns:
            统计信息字典
        """
```

### MemoryOptimizer类
```python
class MemoryOptimizer:
    """记忆优化器"""
    def __init__(self, config_path=None):
        """初始化记忆优化器
        Args:
            config_path: 配置文件路径
        """
        
    def calculate_importance(self, memory_item):
        """计算记忆重要性
        Args:
            memory_item: 记忆项对象
        Returns:
            重要性得分 (float)
        """
        
    def compress_similar_memories(self, memories):
        """压缩相似记忆
        Args:
            memories: 记忆项列表
        Returns:
            压缩后的记忆项列表
        """
        
    def retrieve_relevant_memories(self, query, memories, top_k=5):
        """检索相关记忆
        Args:
            query: 查询文本
            memories: 记忆项列表
            top_k: 返回的最大结果数
        Returns:
            相关记忆项列表
        """
        
    def forget_low_priority_memories(self, memories, max_count=None):
        """遗忘低优先级记忆
        Args:
            memories: 记忆项列表
            max_count: 保留的最大记忆数
        Returns:
            保留的记忆项列表
        """

### MemoryOptimizerEnhanced类
```python
class MemoryOptimizerEnhanced(MemoryOptimizer):
    """增强版记忆优化器，扩展了基础版的功能，增加了聚类、摘要和情感分析"""
    def __init__(self, config_path=None):
        """初始化增强版记忆优化器
        Args:
            config_path: 配置文件路径
        """
        
    def cluster_memories(self, memories):
        """对记忆进行聚类
        Args:
            memories: 记忆项字典
        Returns:
            聚类结果字典
        """
        
    def generate_summary(self, memory_content):
        """生成记忆内容的摘要
        Args:
            memory_content: 记忆内容文本
        Returns:
            生成的摘要
        """
        
    def analyze_memory_sentiment(self, memory):
        """分析记忆的情感
        Args:
            memory: 记忆对象
        Returns:
            情感分析结果
        """
        
    def auto_optimize(self, memories, interval=3600):
        """定期自动优化记忆
        Args:
            memories: 记忆项字典
            interval: 优化间隔（秒）
        Returns:
            优化后的记忆字典和被删除的记忆ID列表
        """
```

## 故障恢复系统 (recovery)

### RecoveryManager类
```python
class RecoveryManager:
    """故障恢复管理器"""
    def __new__(cls, config_path=None):
        """单例模式实现
        Args:
            config_path: 配置文件路径
        Returns:
            RecoveryManager实例
        """
        
    def register_service(self, service_id, health_check_func, recovery_callback=None):
        """注册服务
        Args:
            service_id: 服务ID
            health_check_func: 健康检查函数
            recovery_callback: 恢复回调函数
        """
        
    def unregister_service(self, service_id):
        """注销服务
        Args:
            service_id: 服务ID
        """
        
    def check_service_health(self, service_id=None):
        """检查服务健康状态
        Args:
            service_id: 服务ID，为None时检查所有服务
        Returns:
            健康状态字典
        """
        
    def detect_failures(self):
        """检测故障
        Returns:
            故障列表
        """
        
    def recover_from_failure(self, failure):
        """从故障中恢复
        Args:
            failure: 故障对象
        Returns:
            恢复是否成功 (bool)
        """
        
    def restart_service(self, service_id):
        """重启服务
        Args:
            service_id: 服务ID
        Returns:
            重启是否成功 (bool)
        """
        
    def switch_to_backup(self, service_id):
        """切换到备份服务
        Args:
            service_id: 服务ID
        Returns:
            切换是否成功 (bool)
        """
        
    def notify_administrator(self, service_id, message):
        """通知管理员
        Args:
            service_id: 服务ID
            message: 通知消息
        """
        
    def get_system_status(self):
        """获取系统状态
        Returns:
            系统状态字典
        """
```

## 聚合器模块 (core)

### Aggregator类

聚合器类，用于融合多个Agent的响应。

#### 初始化方法

```python
__init__(self, memory_manager: MemoryManager)
```

**参数**:
- `memory_manager` (MemoryManager): 记忆管理器实例

#### 配置加载方法

```python
load_config(self)
```

加载聚合器配置。

#### 响应融合方法

```python
merge_responses(self, responses: List[Dict]) -> Dict
```

**参数**:
- `responses` (List[Dict]): Agent响应列表

**返回值**:
- `Dict`: 融合后的响应

#### 加权合并策略

```python
_weighted_merge(self, responses: List[Dict]) -> Dict
```

基于权重的响应合并策略。

#### 基于置信度的选择策略

```python
_confidence_based_selection(self, responses: List[Dict]) -> Dict
```

根据置信度选择最佳响应的策略。

#### 简单合并策略

```python
_simple_merge(self, responses: List[Dict]) -> Dict
```

简单的响应合并策略。

#### 生成最终响应

```python
generate_final_response(self, responses: List[Dict], query: str) -> Dict
```

**参数**:
- `responses` (List[Dict]): Agent响应列表
- `query` (str): 用户查询

**返回值**:
- `Dict`: 最终响应

## 分布式协调系统 (distributed)

### DistributedCoordinator类
```python
class DistributedCoordinator:
    """分布式协调器"""
    def __new__(cls, config_path=None):
        """单例模式实现
        Args:
            config_path: 配置文件路径
        Returns:
            DistributedCoordinator实例
        """
        
    def register_agent(self, agent_id, agent_info):
        """注册Agent
        Args:
            agent_id: Agent ID
            agent_info: Agent信息字典
        """
        
    def unregister_agent(self, agent_id):
        """注销Agent
        Args:
            agent_id: Agent ID
        """
        
    def update_agent_heartbeat(self, agent_id):
        """更新Agent心跳
        Args:
            agent_id: Agent ID
        """
        
    def detect_failed_agents(self):
        """检测失败的Agent
        Returns:
            失败的Agent ID列表
        """
        
    def assign_task(self, task, agent_ids=None):
        """分配任务
        Args:
            task: 任务对象
            agent_ids: 可选的Agent ID列表
        Returns:
            分配的Agent ID
        """
        
    def publish_message(self, topic, message):
        """发布消息
        Args:
            topic: 主题
            message: 消息内容
        """
        
    def subscribe_to_topic(self, topic, callback):
        """订阅主题
        Args:
            topic: 主题
            callback: 回调函数
        """
        
    def acquire_resource_lock(self, resource_id, agent_id, timeout=60):
        """获取资源锁
        Args:
            resource_id: 资源ID
            agent_id: Agent ID
            timeout: 超时时间（秒）
        Returns:
            是否获取成功 (bool)
        """
        
    def release_resource_lock(self, resource_id, agent_id):
        """释放资源锁
        Args:
            resource_id: 资源ID
            agent_id: Agent ID
        """
        
    def get_cluster_status(self):
        """获取集群状态
        Returns:
            集群状态字典
        """
```