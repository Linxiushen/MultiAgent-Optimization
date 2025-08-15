import time
import uuid
import threading
from typing import Dict, List, Tuple, Optional, Callable
from queue import PriorityQueue
import json
import os
from distributed.distributed_strategy import StrategyFactory, LoadBalancingStrategy

class DistributedCoordinator:
    """分布式Agent协调器，负责管理Agent注册、任务分配和通信"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DistributedCoordinator, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """初始化协调器状态"""
        self.agents: Dict[str, Dict] = {}
        self.tasks: PriorityQueue = PriorityQueue()
        self.resource_locks: Dict[str, Tuple[str, float]] = {}
        self.message_queue: Dict[str, List[Dict]] = {}
        self.load_config()
        self.strategy_factory = StrategyFactory()
        self.load_balancing_strategy = self._create_strategy()
        self.failure_history = []
        # 初始化故障转移策略
        if self.config.get('failover_config', {}).get('use_failover_strategy', True):
            from .failover_strategy import FailoverStrategy
            self.failover_strategy = FailoverStrategy(coordinator=self)
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_agents, daemon=True)
        self.monitor_thread.start()

    def load_config(self):
        """加载分布式配置"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'distributed_config.json')
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # 使用默认配置
            self.config = {
                "agent_heartbeat_interval": 30,
                "task_timeout": 300,
                "load_balancing_strategy": "round_robin",
                "max_retries": 3
            }

    def _create_strategy(self) -> LoadBalancingStrategy:
        """创建负载均衡策略实例

        Returns:
            LoadBalancingStrategy: 负载均衡策略实例
        """
        strategy_name = self.config.get('load_balancing_strategy', 'round_robin')
        return self.strategy_factory.create_strategy(strategy_name)
        
    def init_failover_strategy(self) -> None:
        """初始化故障转移策略"""
        from .failover_strategy import FailoverStrategy, BackupAgentStrategy
        
        strategy_type = self.config.get('failover_config', {}).get('failover_strategy_type', 'failover')
        if strategy_type == 'failover':
            self.failover_strategy = FailoverStrategy(coordinator=self)
        elif strategy_type == 'backup_agent':
            self.failover_strategy = BackupAgentStrategy(coordinator=self)
        else:
            # 默认使用FailoverStrategy
            self.failover_strategy = FailoverStrategy(coordinator=self)

    def set_strategy(self, strategy_name: str) -> bool:
        """设置负载均衡策略

        Args:
            strategy_name: 策略名称

        Returns:
            bool: 是否成功设置
        """
        try:
            self.load_balancing_strategy = self.strategy_factory.create_strategy(strategy_name)
            self.config['load_balancing_strategy'] = strategy_name
            return True
        except ValueError:
            print(f"无效的策略名称: {strategy_name}")
            return False

    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str], endpoint: str) -> bool:
        """注册新Agent到协调器

        Args:
            agent_id: Agent唯一标识符
            agent_type: Agent类型
            capabilities: Agent具备的能力列表
            endpoint: Agent通信端点

        Returns:
            bool: 注册是否成功
        """
        if agent_id in self.agents:
            print(f"Agent {agent_id} 已存在，更新信息...")

        self.agents[agent_id] = {
            'id': agent_id,
            'type': agent_type,
            'capabilities': capabilities,
            'endpoint': endpoint,
            'last_heartbeat': time.time(),
            'status': 'active',
            'task_count': 0
        }

        # 初始化该Agent的消息队列
        if agent_id not in self.message_queue:
            self.message_queue[agent_id] = []

        print(f"Agent {agent_id} 注册成功")
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """从协调器注销Agent

        Args:
            agent_id: 要注销的Agent ID

        Returns:
            bool: 注销是否成功
        """
        if agent_id not in self.agents:
            print(f"Agent {agent_id} 不存在")
            return False

        del self.agents[agent_id]
        if agent_id in self.message_queue:
            del self.message_queue[agent_id]

        print(f"Agent {agent_id} 注销成功")
        return True

    def heartbeat(self, agent_id: str) -> bool:
        """更新Agent心跳

        Args:
            agent_id: Agent ID

        Returns:
            bool: 更新是否成功
        """
        if agent_id not in self.agents:
            return False

        self.agents[agent_id]['last_heartbeat'] = time.time()
        return True

    def assign_task(self, task_type: str, task_data: Dict, priority: int = 1, callback: Optional[Callable] = None) -> Optional[str]:
        """分配任务给合适的Agent

        Args:
            task_type: 任务类型
            task_data: 任务数据
            priority: 任务优先级，数字越小优先级越高
            callback: 任务完成后的回调函数

        Returns:
            Optional[str]: 任务ID，如果无法分配则返回None
        """
        # 查找有能力执行该任务的活跃Agent
        suitable_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if task_type in agent['capabilities'] and agent['status'] == 'active'
        ]

        if not suitable_agents:
            print(f"没有找到能执行 {task_type} 类型任务的Agent")
            return None

        # 根据负载均衡策略选择Agent
        selected_agent = self._select_agent(suitable_agents)

        # 创建任务
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'type': task_type,
            'data': task_data,
            'priority': priority,
            'assigned_to': selected_agent,
            'assigned_at': time.time(),
            'callback': callback
        }

        # 将任务加入队列
        self.tasks.put((priority, time.time(), task))

        # 更新Agent任务计数
        self.agents[selected_agent]['task_count'] += 1

        print(f"任务 {task_id} 已分配给Agent {selected_agent}")
        return task_id

    def _select_agent(self, agent_ids: List[str]) -> str:
        """根据负载均衡策略选择Agent

        Args:
            agent_ids: 候选Agent ID列表

        Returns:
            str: 选中的Agent ID
        """
        # 准备Agent信息字典，包含选择策略所需的信息
        agent_info = {agent_id: self.agents[agent_id] for agent_id in agent_ids}
        return self.load_balancing_strategy.select_agent(agent_info)

    def send_message(self, from_agent: str, to_agent: str, message_type: str, content: Dict) -> bool:
        """在Agent之间发送消息

        Args:
            from_agent: 发送消息的Agent ID
            to_agent: 接收消息的Agent ID
            message_type: 消息类型
            content: 消息内容

        Returns:
            bool: 发送是否成功
        """
        if to_agent not in self.agents or from_agent not in self.agents:
            return False

        message = {
            'from': from_agent,
            'type': message_type,
            'content': content,
            'timestamp': time.time()
        }

        if to_agent not in self.message_queue:
            self.message_queue[to_agent] = []

        self.message_queue[to_agent].append(message)
        print(f"消息从 {from_agent} 发送到 {to_agent}")
        return True

    def get_messages(self, agent_id: str) -> List[Dict]:
        """获取Agent的消息队列

        Args:
            agent_id: Agent ID

        Returns:
            List[Dict]: 消息列表
        """
        if agent_id not in self.message_queue:
            return []

        messages = self.message_queue[agent_id].copy()
        # 清空队列
        self.message_queue[agent_id] = []
        return messages

    def acquire_lock(self, resource_id: str, agent_id: str, timeout: float = 10.0) -> bool:
        """获取资源锁

        Args:
            resource_id: 资源ID
            agent_id: 请求锁的Agent ID
            timeout: 锁超时时间(秒)

        Returns:
            bool: 是否成功获取锁
        """
        current_time = time.time()

        # 检查锁是否已被持有且未超时
        if resource_id in self.resource_locks:
            holder_id, acquired_time = self.resource_locks[resource_id]
            if current_time - acquired_time < timeout:
                return False
            else:
                print(f"资源 {resource_id} 的锁已超时，释放并重新分配")

        # 获取锁
        self.resource_locks[resource_id] = (agent_id, current_time)
        print(f"Agent {agent_id} 成功获取资源 {resource_id} 的锁")
        return True

    def release_lock(self, resource_id: str, agent_id: str) -> bool:
        """释放资源锁

        Args:
            resource_id: 资源ID
            agent_id: 持有锁的Agent ID

        Returns:
            bool: 是否成功释放锁
        """
        if resource_id not in self.resource_locks:
            return False

        holder_id, _ = self.resource_locks[resource_id]
        if holder_id != agent_id:
            return False

        del self.resource_locks[resource_id]
        print(f"Agent {agent_id} 成功释放资源 {resource_id} 的锁")
        return True

    def _monitor_agents(self):
        """监控Agent状态的后台线程"""
        while True:
            current_time = time.time()
            heartbeat_interval = self.config.get('agent_heartbeat_interval', 30)

            for agent_id, agent in list(self.agents.items()):
                # 检查心跳是否超时
                if current_time - agent['last_heartbeat'] > heartbeat_interval * 2:
                    print(f"Agent {agent_id} 心跳超时，标记为不活跃")
                    agent['status'] = 'inactive'

                    # 处理超时Agent的任务
                    self._handle_timeout_agent(agent_id)

            time.sleep(heartbeat_interval)

    def _handle_timeout_agent(self, agent_id: str):
        """处理超时的Agent，实现故障转移

        Args:
            agent_id: 超时的Agent ID
        """
        print(f"处理超时Agent {agent_id} 的任务，启动故障转移...")
        
        # 获取该Agent的类型和能力，用于寻找替代Agent
        agent_type = self.agents[agent_id]['type']
        agent_capabilities = self.agents[agent_id]['capabilities']
        
        # 获取故障转移配置
        failover_config = self.config.get('failover_config', {})
        failover_enabled = failover_config.get('enabled', True)
        priority_based = failover_config.get('priority_based_reassignment', True)
        preserve_priority = failover_config.get('preserve_task_priority', True)
        max_reassignment = failover_config.get('max_reassignment_attempts', 3)
        backup_agents = failover_config.get('backup_agents', {})
        use_failover_strategy = failover_config.get('use_failover_strategy', True)
        
        if not failover_enabled:
            print(f"故障转移功能已禁用，Agent {agent_id} 的任务将保持原状")
            # 仍然记录故障事件
            self._log_failure_event(agent_id, "heartbeat_timeout", {"failover_disabled": True})
            return
        
        # 查找所有未完成的任务
        pending_tasks = []
        temp_queue = PriorityQueue()
        
        # 遍历任务队列，找出分配给超时Agent的任务
        while not self.tasks.empty():
            priority, timestamp, task = self.tasks.get()
            if task['assigned_to'] == agent_id and 'completed' not in task:
                pending_tasks.append((priority, timestamp, task))
            else:
                temp_queue.put((priority, timestamp, task))
        
        # 恢复其他任务到队列
        while not temp_queue.empty():
            self.tasks.put(temp_queue.get())
        
        # 如果启用了基于优先级的重分配，按优先级排序任务
        if priority_based:
            pending_tasks.sort(key=lambda x: x[0])  # 按优先级排序（数字越小优先级越高）
        
        # 处理需要重新分配的任务
        for priority, timestamp, task in pending_tasks:
            print(f"重新分配任务 {task['id']} (原分配给 {agent_id})")
            
            # 标记原始分配的Agent
            task['original_agent_id'] = agent_id
            
            # 初始化重分配计数器（如果不存在）
            if 'reassignment_count' not in task:
                task['reassignment_count'] = 0
            task['reassignment_count'] += 1
            
            # 初始化之前分配的Agent列表（如果不存在）
            if 'previous_agents' not in task:
                task['previous_agents'] = []
            task['previous_agents'].append(agent_id)
            
            # 检查是否超过最大重分配次数
            if task['reassignment_count'] > max_reassignment:
                print(f"任务 {task['id']} 超过最大重分配次数 {max_reassignment}，标记为失败")
                self._log_failure_event(agent_id, "max_reassignment_exceeded", {
                    "task_id": task['id'],
                    "reassignment_count": task['reassignment_count'],
                    "previous_agents": task['previous_agents']
                })
                continue
            
            # 使用故障转移策略选择新Agent
            new_agent = None
            if use_failover_strategy and self.failover_strategy:
                # 准备可用Agent字典
                available_agents = {}
                for aid, agent in self.agents.items():
                    if aid != agent_id and agent['status'] == 'active':
                        available_agents[aid] = agent
                
                # 使用故障转移策略选择Agent
                new_agent = self.failover_strategy.select_agent(available_agents, task)
            else:
                # 使用传统方法选择Agent
                # 首先检查是否有指定的备份Agent
                if agent_id in backup_agents and backup_agents[agent_id]:
                    backup_agent_id = backup_agents[agent_id]
                    if backup_agent_id in self.agents and self.agents[backup_agent_id]['status'] == 'active':
                        new_agent = backup_agent_id
                        print(f"使用备份Agent {new_agent} 处理任务 {task['id']}")
                
                # 如果没有可用的备份Agent，查找其他替代Agent
                if not new_agent:
                    suitable_agents = [
                        aid for aid, agent in self.agents.items()
                        if aid != agent_id and 
                           agent['status'] == 'active' and 
                           task['type'] in agent['capabilities']
                    ]
                    
                    if suitable_agents:
                        # 选择替代Agent
                        new_agent = self._select_agent_for_failover(suitable_agents, task)
            
            if new_agent:
                # 更新任务信息
                task['assigned_to'] = new_agent
                task['reassigned_at'] = time.time()
                task['reassignment_count'] = task.get('reassignment_count', 0) + 1
                task['previous_agents'] = task.get('previous_agents', []) + [agent_id]
                
                # 检查重试次数是否超过限制
                max_retries = min(self.config.get('max_retries', 3), max_reassignment)
                if task['reassignment_count'] <= max_retries:
                    # 根据配置决定是否保留原任务优先级
                    if not preserve_priority:
                        # 降低优先级（数字越大优先级越低）
                        priority += 1
                        
                    # 将任务重新加入队列
                    self.tasks.put((priority, timestamp, task))
                    self.agents[new_agent]['task_count'] += 1
                    
                    # 记录故障转移成功
                    self._log_failure_event(agent_id, "task_reassigned", {
                        "task_id": task['id'],
                        "new_agent": new_agent,
                        "reassignment_count": task['reassignment_count']
                    })
                    
                    # 如果使用了故障转移策略，更新故障转移历史
                    if use_failover_strategy and self.failover_strategy:
                        self.failover_strategy.update_failover_history(agent_id, success=True)
                else:
                    # 重试次数超过限制，标记任务为失败
                    task['status'] = 'failed'
                    task['failure_reason'] = f"超过最大重试次数 {max_retries}"
                    
                    # 记录故障事件
                    self._log_failure_event(agent_id, "max_retries_exceeded", {
                        "task_id": task['id'],
                        "max_retries": max_retries,
                        "reassignment_count": task['reassignment_count']
                    })
            else:
                # 没有找到合适的Agent，将任务标记为挂起
                task['status'] = 'pending'
                task['pending_reason'] = "无可用Agent处理该任务"
                
                # 将任务放回队列，但降低优先级
                self.tasks.put((priority + 5, timestamp, task))
                
                # 记录故障事件
                self._log_failure_event(agent_id, "no_suitable_agent", {
                    "task_id": task['id'],
                    "task_type": task['type']
                })
        
        # 释放该Agent持有的所有资源锁
        for resource, (lock_holder, _) in list(self.resource_locks.items()):
            if lock_holder == agent_id:
                print(f"释放Agent {agent_id} 持有的资源锁: {resource}")
                del self.resource_locks[resource]
                    
                    # 重新加入任务队列
                    self.tasks.put((priority, timestamp, task))
                    self.agents[new_agent]['task_count'] += 1
                    print(f"任务 {task['id']} 已重新分配给Agent {new_agent}")
                    
                    # 发送通知消息给新Agent
                    self.send_message(
                        from_agent="coordinator",
                        to_agent=new_agent,
                        message_type="task_reassignment",
                        content={
                            "task_id": task['id'],
                            "original_agent": agent_id,
                            "reason": "agent_timeout",
                            "reassignment_count": task['reassignment_count'],
                            "previous_agents": task['previous_agents']
                        }
                    )
                else:
                    print(f"任务 {task['id']} 重试次数已达上限 ({max_retries})，标记为失败")
                    # 如果有回调函数，通知任务失败
                    if 'callback' in task and task['callback']:
                        try:
                            task['callback'](task['id'], None, "max_retries_exceeded")
                        except Exception as e:
                            print(f"执行回调时出错: {e}")
                    
                    # 记录任务失败事件
                    self._log_failure_event(
                        agent_id, 
                        "task_max_retries", 
                        {
                            "task_id": task['id'],
                            "reassignment_count": task['reassignment_count'],
                            "previous_agents": task['previous_agents']
                        }
                    )
            else:
                print(f"没有找到合适的替代Agent处理任务 {task['id']}，任务将被挂起")
                # 将任务标记为挂起，等待有合适的Agent注册
                task['status'] = 'pending'
                task['pending_since'] = time.time()
                # 放回队列，但降低优先级
                self.tasks.put((priority + 1, timestamp, task))
        
        # 释放该Agent持有的所有资源锁
        for resource_id, (holder_id, _) in list(self.resource_locks.items()):
            if holder_id == agent_id:
                del self.resource_locks[resource_id]
                print(f"释放超时Agent {agent_id} 持有的资源锁: {resource_id}")
        
        # 记录故障事件
        self._log_failure_event(agent_id, "heartbeat_timeout")
        
    def _select_agent_for_failover(self, agent_ids: List[str], task: Dict) -> str:
        """为故障转移选择最合适的Agent
        
        Args:
            agent_ids: 可选的Agent ID列表
            task: 需要重新分配的任务
            
        Returns:
            str: 选择的Agent ID
        """
        # 获取故障转移配置
        failover_config = self.config.get('failover_config', {})
        critical_agents = failover_config.get('critical_agents', [])
        
        # 如果任务之前已经被重新分配过，避免分配给之前失败的Agent
        previous_agents = task.get('previous_agents', [])
        available_agents = [aid for aid in agent_ids if aid not in previous_agents]
        
        # 如果没有可用的新Agent，则从所有可选Agent中选择
        if not available_agents:
            available_agents = agent_ids
        
        # 优先考虑关键Agent
        critical_available = [aid for aid in available_agents if aid in critical_agents]
        if critical_available:
            # 在关键Agent中选择负载最低的
            return min(critical_available, key=lambda aid: self.agents[aid]['task_count'])
        
        # 根据任务类型和Agent能力的匹配度排序
        # 计算每个Agent的能力匹配分数
        task_type = task['type']
        agent_scores = {}
        
        for aid in available_agents:
            agent = self.agents[aid]
            # 基础分数：任务类型匹配加10分
            score = 10 if task_type in agent['capabilities'] else 0
            # 减去当前任务数作为负载因子
            score -= agent['task_count']
            # 如果Agent最近有故障，降低分数
            recent_failures = self._count_recent_failures(aid)
            score -= recent_failures * 2
            
            agent_scores[aid] = score
        
        # 选择得分最高的Agent
        if agent_scores:
            return max(agent_scores.items(), key=lambda x: x[1])[0]
        
        # 如果没有特殊条件，使用默认的负载均衡策略
        return self._select_agent(available_agents)
        
    def _log_failure_event(self, agent_id: str, failure_type: str, details: Dict = None):
        """记录故障事件
        
        Args:
            agent_id: 发生故障的Agent ID
            failure_type: 故障类型，如 'heartbeat_timeout', 'task_failure', 'connection_error'
            details: 故障详情，可选
        """
        if details is None:
            details = {}
            
        # 创建故障事件记录
        failure_event = {
            'agent_id': agent_id,
            'failure_type': failure_type,
            'timestamp': time.time(),
            'details': details
        }
        
        # 添加Agent信息
        if agent_id in self.agents:
            failure_event['agent_info'] = {
                'type': self.agents[agent_id]['type'],
                'capabilities': self.agents[agent_id]['capabilities'],
                'task_count': self.agents[agent_id]['task_count'],
                'last_heartbeat': self.agents[agent_id]['last_heartbeat']
            }
        
        # 将故障事件添加到历史记录
        if not hasattr(self, 'failure_history'):
            self.failure_history = []
        self.failure_history.append(failure_event)
        
        # 限制历史记录大小
        max_history = self.config.get('max_failure_history', 100)
        if len(self.failure_history) > max_history:
            self.failure_history = self.failure_history[-max_history:]
        
        # 可以在这里添加持久化存储逻辑，如写入日志文件或数据库
        print(f"故障事件已记录: {failure_type} - Agent {agent_id}")
        
        # 触发故障通知
        self._notify_failure(failure_event)
        
    def _notify_failure(self, failure_event: Dict):
        """发送故障通知
        
        Args:
            failure_event: 故障事件信息
        """
        # 获取通知配置
        notification_config = self.config.get('failure_notification', {})
        enabled = notification_config.get('enabled', False)
        
        if not enabled:
            return
            
        notification_types = notification_config.get('types', [])
        severity = self._determine_failure_severity(failure_event)
        
        # 根据故障严重程度和配置决定是否发送通知
        if severity < notification_config.get('min_severity', 'low'):
            return
            
        # 准备通知内容
        notification = {
            'type': 'failure_alert',
            'severity': severity,
            'timestamp': failure_event['timestamp'],
            'agent_id': failure_event['agent_id'],
            'failure_type': failure_event['failure_type'],
            'details': failure_event['details'],
            'message': f"Agent {failure_event['agent_id']} 发生 {failure_event['failure_type']} 故障"
        }
        
        # 发送各类通知
        for notification_type in notification_types:
            try:
                if notification_type == 'log':
                    self._send_log_notification(notification)
                elif notification_type == 'email':
                    self._send_email_notification(notification)
                elif notification_type == 'webhook':
                    self._send_webhook_notification(notification)
                elif notification_type == 'system':
                    self._send_system_notification(notification)
            except Exception as e:
                print(f"发送 {notification_type} 通知失败: {e}")
                
    def _determine_failure_severity(self, failure_event: Dict) -> str:
        """确定故障严重程度
        
        Args:
            failure_event: 故障事件信息
            
        Returns:
            str: 严重程度，'critical', 'high', 'medium', 'low' 之一
        """
        failure_type = failure_event['failure_type']
        
        # 根据故障类型确定基础严重程度
        if failure_type in ['system_crash', 'data_corruption']:
            base_severity = 'critical'
        elif failure_type in ['heartbeat_timeout', 'connection_error']:
            base_severity = 'high'
        elif failure_type in ['task_failure', 'resource_exhaustion']:
            base_severity = 'medium'
        else:
            base_severity = 'low'
            
        # 考虑其他因素调整严重程度
        # 例如，如果是关键Agent或影响多个任务，提高严重程度
        if failure_event.get('details', {}).get('is_critical_agent', False):
            if base_severity == 'medium':
                base_severity = 'high'
            elif base_severity == 'low':
                base_severity = 'medium'
                
        # 如果短时间内同一Agent多次失败，提高严重程度
        recent_failures = self._count_recent_failures(failure_event['agent_id'])
        if recent_failures > 3 and base_severity != 'critical':
            severity_levels = ['low', 'medium', 'high', 'critical']
            current_index = severity_levels.index(base_severity)
            if current_index < len(severity_levels) - 1:
                base_severity = severity_levels[current_index + 1]
                
        return base_severity
        
    def _count_recent_failures(self, agent_id: str, time_window: int = 300) -> int:
        """计算最近一段时间内某Agent的故障次数
        
        Args:
            agent_id: Agent ID
            time_window: 时间窗口，单位为秒，默认5分钟
            
        Returns:
            int: 故障次数
        """
        if not hasattr(self, 'failure_history'):
            return 0
            
        current_time = time.time()
        count = 0
        
        for event in self.failure_history:
            if event['agent_id'] == agent_id and current_time - event['timestamp'] <= time_window:
                count += 1
                
        return count
        
    def _send_log_notification(self, notification: Dict):
        """发送日志通知
        
        Args:
            notification: 通知信息
        """
        severity = notification['severity']
        message = notification['message']
        details = json.dumps(notification['details'], indent=2)
        
        log_message = f"[{severity.upper()}] {message}\n{details}"
        print(log_message)
        
        # 这里可以集成日志系统，如logging模块或ELK
        
    def _send_email_notification(self, notification: Dict):
        """发送邮件通知
        
        Args:
            notification: 通知信息
        """
        # 邮件通知配置
        email_config = self.config.get('email_notification', {})
        recipients = email_config.get('recipients', [])
        
        if not recipients:
            print("未配置邮件接收者，跳过邮件通知")
            return
            
        # 这里可以集成邮件发送功能
        print(f"将发送邮件通知到: {', '.join(recipients)}")
        
    def _send_webhook_notification(self, notification: Dict):
        """发送Webhook通知
        
        Args:
            notification: 通知信息
        """
        # Webhook通知配置
        webhook_config = self.config.get('webhook_notification', {})
        webhook_url = webhook_config.get('url')
        
        if not webhook_url:
            print("未配置Webhook URL，跳过Webhook通知")
            return
            
        # 这里可以集成Webhook发送功能
        print(f"将发送Webhook通知到: {webhook_url}")
        
    def _send_system_notification(self, notification: Dict):
        """发送系统通知
        
        Args:
            notification: 通知信息
        """
        # 向所有活跃Agent广播系统通知
        for agent_id, agent in self.agents.items():
            if agent['status'] == 'active':
                self.send_message(
                    from_agent="coordinator",
                    to_agent=agent_id,
                    message_type="system_notification",
                    content={
                        "notification_type": "failure_alert",
                        "severity": notification['severity'],
                        "message": notification['message'],
                        "affected_agent": notification['agent_id']
                    }
                )
                
    def get_failure_history(self, filters: Dict = None) -> List[Dict]:
        """获取故障历史记录
        
        Args:
            filters: 过滤条件，可选，例如 {'agent_id': 'agent1', 'failure_type': 'heartbeat_timeout'}
            
        Returns:
            List[Dict]: 符合条件的故障历史记录
        """
        if not hasattr(self, 'failure_history'):
            return []
            
        if filters is None:
            return self.failure_history
            
        filtered_history = []
        for event in self.failure_history:
            match = True
            for key, value in filters.items():
                if key not in event or event[key] != value:
                    match = False
                    break
            if match:
                filtered_history.append(event)
                
        return filtered_history

    def get_cluster_status(self) -> Dict:
        """获取集群状态

        Returns:
            Dict: 集群状态信息
        """
        active_agents = [agent for agent in self.agents.values() if agent['status'] == 'active']
        inactive_agents = [agent for agent in self.agents.values() if agent['status'] == 'inactive']

        # 获取最近的故障历史
        recent_failures = []
        if hasattr(self, 'failure_history'):
            # 只获取最近24小时的故障
            current_time = time.time()
            recent_failures = [f for f in self.failure_history 
                              if current_time - f['timestamp'] <= 86400]

        return {
            'total_agents': len(self.agents),
            'active_agents': len(active_agents),
            'inactive_agents': len(inactive_agents),
            'pending_tasks': self.tasks.qsize(),
            'active_agents_details': active_agents,
            'locked_resources': list(self.resource_locks.keys()),
            'recent_failures': recent_failures,
            'failure_count': len(recent_failures) if hasattr(self, 'failure_history') else 0
        }

# 创建单例实例
coordinator = DistributedCoordinator()