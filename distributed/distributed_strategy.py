from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import random
import time

class LoadBalancingStrategy(ABC):
    """负载均衡策略基类"""
    @abstractmethod
    def select_agent(self, agents: Dict[str, Dict[str, Any]], task: Dict[str, Any]) -> str:
        """选择一个Agent来处理任务
        Args:
            agents: 可用Agent字典，键为Agent ID，值为Agent信息
            task: 任务信息
        Returns:
            选中的Agent ID
        """
        pass

class RoundRobinStrategy(LoadBalancingStrategy):
    """轮询负载均衡策略"""
    def __init__(self):
        self.current_index = 0
        self.agent_ids = []
        self.last_update_time = 0

    def select_agent(self, agents: Dict[str, Dict[str, Any]], task: Dict[str, Any]) -> str:
        # 检查Agent列表是否有变化
        current_agent_ids = list(agents.keys())
        if current_agent_ids != self.agent_ids or time.time() - self.last_update_time > 60:
            self.agent_ids = current_agent_ids
            self.current_index = 0
            self.last_update_time = time.time()

        if not self.agent_ids:
            raise ValueError("No available agents")

        # 轮询选择Agent
        selected_agent_id = self.agent_ids[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.agent_ids)
        return selected_agent_id

class RandomStrategy(LoadBalancingStrategy):
    """随机负载均衡策略"""
    def select_agent(self, agents: Dict[str, Dict[str, Any]], task: Dict[str, Any]) -> str:
        agent_ids = list(agents.keys())
        if not agent_ids:
            raise ValueError("No available agents")
        return random.choice(agent_ids)

class LeastConnectionsStrategy(LoadBalancingStrategy):
    """最少连接负载均衡策略"""
    def select_agent(self, agents: Dict[str, Dict[str, Any]], task: Dict[str, Any]) -> str:
        agent_ids = list(agents.keys())
        if not agent_ids:
            raise ValueError("No available agents")

        # 找出连接数最少的Agent
        min_connections = float('inf')
        selected_agent_id = None

        for agent_id in agent_ids:
            connections = agents[agent_id].get('active_connections', 0)
            if connections < min_connections:
                min_connections = connections
                selected_agent_id = agent_id
            elif connections == min_connections:
                # 连接数相同时随机选择
                if random.random() < 0.5:
                    selected_agent_id = agent_id

        return selected_agent_id

class PerformanceBasedStrategy(LoadBalancingStrategy):
    """基于性能的负载均衡策略"""
    def select_agent(self, agents: Dict[str, Dict[str, Any]], task: Dict[str, Any]) -> str:
        agent_ids = list(agents.keys())
        if not agent_ids:
            raise ValueError("No available agents")

        # 计算每个Agent的性能得分
        agent_scores = {}
        for agent_id in agent_ids:
            # 获取Agent性能指标
            response_time = agents[agent_id].get('avg_response_time', 1.0)
            success_rate = agents[agent_id].get('success_rate', 0.5)
            load = agents[agent_id].get('load', 0.5)
            task_compatibility = agents[agent_id].get('task_compatibility', 0.5)

            # 计算综合得分 (响应时间越低、成功率越高、负载越低、兼容性越高，得分越高)
            score = (1.0 / response_time) * success_rate * (1.0 - load) * task_compatibility
            agent_scores[agent_id] = score

        # 选择得分最高的Agent
        max_score = max(agent_scores.values())
        candidates = [agent_id for agent_id, score in agent_scores.items() if score == max_score]

        return random.choice(candidates)

class TaskTypeBasedStrategy(LoadBalancingStrategy):
    """基于任务类型的负载均衡策略"""
    def select_agent(self, agents: Dict[str, Dict[str, Any]], task: Dict[str, Any]) -> str:
        agent_ids = list(agents.keys())
        if not agent_ids:
            raise ValueError("No available agents")

        task_type = task.get('type', 'default')
        best_agent_id = None
        best_match_score = 0

        # 找到最适合处理该类型任务的Agent
        for agent_id in agent_ids:
            agent_task_types = agents[agent_id].get('supported_task_types', [])
            if task_type in agent_task_types:
                # 计算匹配度 (简单实现)
                match_score = 1.0
                # 检查是否有专门针对该任务类型的技能评分
                if 'task_type_scores' in agents[agent_id]:
                    match_score = agents[agent_id]['task_type_scores'].get(task_type, 1.0)

                if match_score > best_match_score:
                    best_match_score = match_score
                    best_agent_id = agent_id

        # 如果没有找到专门处理该任务类型的Agent，则随机选择
        if best_agent_id is None:
            best_agent_id = random.choice(agent_ids)

        return best_agent_id

class StrategyFactory:
    """策略工厂，用于创建不同的负载均衡策略"""
    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> LoadBalancingStrategy:
        """创建负载均衡策略
        Args:
            strategy_type: 策略类型
            **kwargs: 策略参数
        Returns:
            负载均衡策略实例
        """
        # 导入故障转移策略
        from .failover_strategy import FailoverStrategy, BackupAgentStrategy
        
        strategies = {
            'round_robin': RoundRobinStrategy,
            'random': RandomStrategy,
            'least_connections': LeastConnectionsStrategy,
            'performance_based': PerformanceBasedStrategy,
            'task_type_based': TaskTypeBasedStrategy,
            'failover': FailoverStrategy,
            'backup_agent': BackupAgentStrategy
        }

        if strategy_type not in strategies:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

        return strategies[strategy_type](**kwargs)