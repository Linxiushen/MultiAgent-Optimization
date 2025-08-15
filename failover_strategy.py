from typing import List, Dict, Any, Optional
from .distributed_strategy import LoadBalancingStrategy
import random
import time

class FailoverStrategy(LoadBalancingStrategy):
    """故障转移策略
    
    专门用于处理Agent故障时的任务重分配
    """
    
    def __init__(self, coordinator=None):
        self.coordinator = coordinator
        self.failover_history = {}
        # 记录最近一次故障转移时间
        self.last_failover_time = {}
    
    def select_agent(self, agents: Dict[str, Dict[str, Any]], task: Dict[str, Any]) -> str:
        """选择一个Agent来处理故障转移的任务
        
        Args:
            agents: 可用Agent字典，键为Agent ID，值为Agent信息
            task: 任务信息，包含原始分配的Agent ID和任务详情
            
        Returns:
            选中的Agent ID
        """
        if not agents:
            return None
            
        agent_ids = list(agents.keys())
        if not agent_ids:
            return None
            
        # 获取任务信息
        task_type = task.get('type', 'default')
        task_priority = task.get('priority', 1)
        failed_agent_id = task.get('original_agent_id')
        retry_count = task.get('retry_count', 0)
        
        # 排除已失败的Agent
        if failed_agent_id in agent_ids:
            agent_ids.remove(failed_agent_id)
            
        if not agent_ids:
            return None
            
        # 根据任务优先级和重试次数调整选择策略
        if task_priority >= 8 or retry_count >= 2:
            # 高优先级任务或多次重试的任务，选择最可靠的Agent
            return self._select_most_reliable_agent(agent_ids, agents, task_type)
        elif task_priority >= 5:
            # 中优先级任务，选择能力匹配且负载较低的Agent
            return self._select_capability_matched_agent(agent_ids, agents, task_type)
        else:
            # 低优先级任务，选择负载最低的Agent
            return self._select_least_loaded_agent(agent_ids, agents)
    
    def _select_most_reliable_agent(self, agent_ids: List[str], agents: Dict[str, Dict[str, Any]], task_type: str) -> str:
        """选择最可靠的Agent
        
        根据历史成功率、响应时间和故障历史选择最可靠的Agent
        """
        reliability_scores = {}
        
        for agent_id in agent_ids:
            # 基础可靠性分数
            score = 100
            
            # 考虑Agent的历史成功率
            success_rate = agents[agent_id].get('success_rate', 0.9)
            score *= success_rate
            
            # 考虑Agent的平均响应时间
            avg_response_time = agents[agent_id].get('avg_response_time', 1.0)
            response_time_factor = 1.0 / max(avg_response_time, 0.1)
            score *= min(response_time_factor, 2.0)  # 限制响应时间因子的影响
            
            # 考虑Agent的故障历史
            failure_count = self.failover_history.get(agent_id, 0)
            score -= min(failure_count * 5, 50)  # 每次故障减5分，最多减50分
            
            # 考虑最近一次故障时间
            last_failure_time = self.last_failover_time.get(agent_id, 0)
            time_since_last_failure = time.time() - last_failure_time
            if time_since_last_failure < 300:  # 5分钟内
                score -= 20
            elif time_since_last_failure < 1800:  # 30分钟内
                score -= 10
            
            # 考虑任务类型匹配度
            if 'supported_task_types' in agents[agent_id] and task_type in agents[agent_id]['supported_task_types']:
                score += 20
            
            # 确保分数为正
            reliability_scores[agent_id] = max(score, 1)
        
        # 选择可靠性分数最高的Agent
        best_agent_id = max(reliability_scores.items(), key=lambda x: x[1])[0]
        return best_agent_id
    
    def _select_capability_matched_agent(self, agent_ids: List[str], agents: Dict[str, Dict[str, Any]], task_type: str) -> str:
        """选择能力匹配的Agent
        
        根据任务类型和Agent能力进行匹配，在匹配的Agent中选择负载较低的
        """
        # 找出支持该任务类型的Agent
        capable_agents = []
        for agent_id in agent_ids:
            if 'supported_task_types' in agents[agent_id] and task_type in agents[agent_id]['supported_task_types']:
                capable_agents.append(agent_id)
        
        # 如果没有找到匹配的Agent，使用所有可用Agent
        if not capable_agents:
            capable_agents = agent_ids
        
        # 在匹配的Agent中选择负载最低的
        return self._select_least_loaded_agent(capable_agents, agents)
    
    def _select_least_loaded_agent(self, agent_ids: List[str], agents: Dict[str, Dict[str, Any]]) -> str:
        """选择负载最低的Agent
        
        根据当前任务数和资源使用率选择负载最低的Agent
        """
        if not agent_ids:
            return None
            
        # 计算每个Agent的负载分数
        load_scores = {}
        for agent_id in agent_ids:
            # 获取Agent的任务数
            task_count = agents[agent_id].get('task_count', 0)
            # 获取Agent的资源使用率
            cpu_usage = agents[agent_id].get('cpu_usage', 0.5)
            memory_usage = agents[agent_id].get('memory_usage', 0.5)
            
            # 计算综合负载分数 (任务数越少、资源使用率越低，分数越高)
            load_score = 100 - (task_count * 5) - (cpu_usage * 50) - (memory_usage * 30)
            load_scores[agent_id] = max(load_score, 1)  # 确保分数为正
        
        # 选择负载分数最高的Agent
        best_agent_id = max(load_scores.items(), key=lambda x: x[1])[0]
        return best_agent_id
    
    def update_failover_history(self, agent_id: str, success: bool = False):
        """更新故障转移历史
        
        记录Agent的故障转移历史，用于后续选择Agent时参考
        
        Args:
            agent_id: Agent ID
            success: 故障转移是否成功
        """
        if not success:
            # 记录故障次数
            self.failover_history[agent_id] = self.failover_history.get(agent_id, 0) + 1
            # 记录最近一次故障时间
            self.last_failover_time[agent_id] = time.time()
        else:
            # 故障转移成功，减少故障计数
            if agent_id in self.failover_history and self.failover_history[agent_id] > 0:
                self.failover_history[agent_id] -= 1
    
    def get_failover_stats(self) -> Dict[str, Any]:
        """获取故障转移统计信息
        
        Returns:
            故障转移统计信息，包括总故障次数、各Agent故障次数等
        """
        total_failures = sum(self.failover_history.values())
        return {
            'total_failures': total_failures,
            'agent_failures': self.failover_history.copy(),
            'last_failover_times': self.last_failover_time.copy()
        }

class BackupAgentStrategy(LoadBalancingStrategy):
    """备份Agent策略
    
    为每个任务指定主Agent和备份Agent，当主Agent失败时自动切换到备份Agent
    """
    
    def __init__(self, backup_count: int = 1):
        self.backup_count = backup_count
        self.task_assignments = {}  # 记录任务分配情况
    
    def select_agent(self, agents: Dict[str, Dict[str, Any]], task: Dict[str, Any]) -> str:
        """选择一个Agent来处理任务，并指定备份Agent
        
        Args:
            agents: 可用Agent字典，键为Agent ID，值为Agent信息
            task: 任务信息
            
        Returns:
            选中的Agent ID
        """
        if not agents:
            return None
            
        agent_ids = list(agents.keys())
        if not agent_ids:
            return None
            
        task_id = task.get('id')
        if not task_id:
            # 如果任务没有ID，无法跟踪，直接随机选择一个Agent
            return random.choice(agent_ids)
        
        # 检查是否是故障转移情况
        failed_agent_id = task.get('failed_agent_id')
        if failed_agent_id and task_id in self.task_assignments:
            # 获取该任务的备份Agent列表
            backup_agents = self.task_assignments[task_id]['backup_agents']
            # 从备份列表中移除已失败的Agent
            if failed_agent_id in backup_agents:
                backup_agents.remove(failed_agent_id)
            
            # 如果还有备份Agent，选择第一个
            if backup_agents and backup_agents[0] in agent_ids:
                selected_agent = backup_agents[0]
                # 更新任务分配记录
                self.task_assignments[task_id]['primary_agent'] = selected_agent
                self.task_assignments[task_id]['backup_agents'] = backup_agents[1:]
                return selected_agent
        
        # 新任务或没有可用的备份Agent，重新选择
        # 按负载排序Agent
        sorted_agents = sorted(agent_ids, key=lambda aid: agents[aid].get('task_count', 0))
        
        # 选择负载最低的作为主Agent
        primary_agent = sorted_agents[0]
        
        # 选择其他Agent作为备份
        backup_agents = []
        for i in range(1, min(self.backup_count + 1, len(sorted_agents))):
            backup_agents.append(sorted_agents[i])
        
        # 记录任务分配情况
        self.task_assignments[task_id] = {
            'primary_agent': primary_agent,
            'backup_agents': backup_agents,
            'assignment_time': time.time()
        }
        
        return primary_agent
    
    def get_backup_agents(self, task_id: str) -> List[str]:
        """获取任务的备份Agent列表
        
        Args:
            task_id: 任务ID
            
        Returns:
            备份Agent ID列表
        """
        if task_id in self.task_assignments:
            return self.task_assignments[task_id]['backup_agents']
        return []
    
    def cleanup_old_assignments(self, max_age: int = 3600):
        """清理过期的任务分配记录
        
        Args:
            max_age: 最大保留时间（秒）
        """
        current_time = time.time()
        expired_tasks = []
        
        for task_id, assignment in self.task_assignments.items():
            if current_time - assignment['assignment_time'] > max_age:
                expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            del self.task_assignments[task_id]