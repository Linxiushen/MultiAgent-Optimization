import unittest
import time
from unittest.mock import MagicMock, patch
from distributed.failover_strategy import FailoverStrategy, BackupAgentStrategy

class TestFailoverStrategy(unittest.TestCase):
    """测试故障转移策略"""
    
    def setUp(self):
        # 创建模拟协调器
        self.mock_coordinator = MagicMock()
        # 创建故障转移策略实例
        self.failover_strategy = FailoverStrategy(coordinator=self.mock_coordinator)
        
        # 模拟Agent数据
        self.agents = {
            'agent1': {
                'id': 'agent1',
                'type': 'worker',
                'capabilities': ['task_type1', 'task_type2'],
                'task_count': 2,
                'success_rate': 0.95,
                'avg_response_time': 0.5,
                'status': 'active'
            },
            'agent2': {
                'id': 'agent2',
                'type': 'worker',
                'capabilities': ['task_type1', 'task_type3'],
                'task_count': 5,
                'success_rate': 0.9,
                'avg_response_time': 0.8,
                'status': 'active'
            },
            'agent3': {
                'id': 'agent3',
                'type': 'worker',
                'capabilities': ['task_type2', 'task_type3'],
                'task_count': 1,
                'success_rate': 0.8,
                'avg_response_time': 0.3,
                'status': 'active'
            }
        }
        
        # 模拟任务数据
        self.task = {
            'id': 'task1',
            'type': 'task_type1',
            'priority': 5,
            'original_agent_id': 'agent4',  # 已失败的Agent
            'retry_count': 0
        }
    
    def test_select_agent_for_high_priority_task(self):
        """测试高优先级任务的Agent选择"""
        # 设置高优先级任务
        high_priority_task = self.task.copy()
        high_priority_task['priority'] = 9
        
        # 调用故障转移策略选择Agent
        selected_agent = self.failover_strategy.select_agent(self.agents, high_priority_task)
        
        # 验证选择了最可靠的Agent（agent1，因为它有最高的成功率）
        self.assertEqual(selected_agent, 'agent1')
    
    def test_select_agent_for_medium_priority_task(self):
        """测试中优先级任务的Agent选择"""
        # 设置中优先级任务
        medium_priority_task = self.task.copy()
        medium_priority_task['priority'] = 5
        
        # 调用故障转移策略选择Agent
        selected_agent = self.failover_strategy.select_agent(self.agents, medium_priority_task)
        
        # 验证选择了能力匹配且负载较低的Agent
        self.assertIn(selected_agent, ['agent1', 'agent2'])  # 这两个Agent都支持task_type1
    
    def test_select_agent_for_low_priority_task(self):
        """测试低优先级任务的Agent选择"""
        # 设置低优先级任务
        low_priority_task = self.task.copy()
        low_priority_task['priority'] = 2
        
        # 调用故障转移策略选择Agent
        selected_agent = self.failover_strategy.select_agent(self.agents, low_priority_task)
        
        # 验证选择了负载最低的Agent（agent3，因为它的任务数最少）
        self.assertEqual(selected_agent, 'agent3')
    
    def test_update_failover_history(self):
        """测试故障转移历史更新"""
        # 初始状态下故障历史应为空
        self.assertEqual(len(self.failover_strategy.failover_history), 0)
        
        # 记录一次故障
        self.failover_strategy.update_failover_history('agent4', success=False)
        
        # 验证故障历史已更新
        self.assertEqual(self.failover_strategy.failover_history['agent4'], 1)
        self.assertIn('agent4', self.failover_strategy.last_failover_time)
        
        # 再记录一次故障
        self.failover_strategy.update_failover_history('agent4', success=False)
        
        # 验证故障计数增加
        self.assertEqual(self.failover_strategy.failover_history['agent4'], 2)
        
        # 记录一次成功
        self.failover_strategy.update_failover_history('agent4', success=True)
        
        # 验证故障计数减少
        self.assertEqual(self.failover_strategy.failover_history['agent4'], 1)
    
    def test_get_failover_stats(self):
        """测试获取故障转移统计信息"""
        # 记录几次故障
        self.failover_strategy.update_failover_history('agent4', success=False)
        self.failover_strategy.update_failover_history('agent5', success=False)
        self.failover_strategy.update_failover_history('agent4', success=False)
        
        # 获取故障统计信息
        stats = self.failover_strategy.get_failover_stats()
        
        # 验证统计信息正确
        self.assertEqual(stats['total_failures'], 3)
        self.assertEqual(stats['agent_failures']['agent4'], 2)
        self.assertEqual(stats['agent_failures']['agent5'], 1)
        self.assertIn('agent4', stats['last_failover_times'])
        self.assertIn('agent5', stats['last_failover_times'])

class TestBackupAgentStrategy(unittest.TestCase):
    """测试备份Agent策略"""
    
    def setUp(self):
        # 创建备份Agent策略实例
        self.backup_strategy = BackupAgentStrategy(backup_count=2)
        
        # 模拟Agent数据
        self.agents = {
            'agent1': {'id': 'agent1', 'task_count': 2},
            'agent2': {'id': 'agent2', 'task_count': 5},
            'agent3': {'id': 'agent3', 'task_count': 1},
            'agent4': {'id': 'agent4', 'task_count': 3},
            'agent5': {'id': 'agent5', 'task_count': 0}
        }
        
        # 模拟任务数据
        self.task = {'id': 'task1', 'type': 'task_type1'}
    
    def test_select_agent_new_task(self):
        """测试为新任务选择Agent"""
        # 调用备份Agent策略选择Agent
        selected_agent = self.backup_strategy.select_agent(self.agents, self.task)
        
        # 验证选择了负载最低的Agent作为主Agent
        self.assertEqual(selected_agent, 'agent5')  # agent5的任务数为0
        
        # 验证任务分配记录已创建
        self.assertIn('task1', self.backup_strategy.task_assignments)
        
        # 验证备份Agent列表包含两个Agent
        backup_agents = self.backup_strategy.task_assignments['task1']['backup_agents']
        self.assertEqual(len(backup_agents), 2)
        
        # 验证备份Agent是负载第二低和第三低的Agent
        self.assertIn('agent3', backup_agents)  # agent3的任务数为1
        self.assertIn('agent1', backup_agents)  # agent1的任务数为2
    
    def test_select_agent_failover(self):
        """测试故障转移时选择备份Agent"""
        # 先为任务分配主Agent和备份Agent
        self.backup_strategy.task_assignments['task1'] = {
            'primary_agent': 'agent1',
            'backup_agents': ['agent3', 'agent5'],
            'assignment_time': time.time()
        }
        
        # 创建故障转移任务
        failover_task = self.task.copy()
        failover_task['failed_agent_id'] = 'agent1'  # 主Agent失败
        
        # 调用备份Agent策略选择Agent
        selected_agent = self.backup_strategy.select_agent(self.agents, failover_task)
        
        # 验证选择了第一个备份Agent
        self.assertEqual(selected_agent, 'agent3')
        
        # 验证任务分配记录已更新
        self.assertEqual(self.backup_strategy.task_assignments['task1']['primary_agent'], 'agent3')
        self.assertEqual(self.backup_strategy.task_assignments['task1']['backup_agents'], ['agent5'])
    
    def test_get_backup_agents(self):
        """测试获取备份Agent列表"""
        # 设置任务分配记录
        self.backup_strategy.task_assignments['task1'] = {
            'primary_agent': 'agent1',
            'backup_agents': ['agent3', 'agent5'],
            'assignment_time': time.time()
        }
        
        # 获取备份Agent列表
        backup_agents = self.backup_strategy.get_backup_agents('task1')
        
        # 验证备份Agent列表正确
        self.assertEqual(backup_agents, ['agent3', 'agent5'])
        
        # 测试获取不存在的任务的备份Agent
        backup_agents = self.backup_strategy.get_backup_agents('task2')
        self.assertEqual(backup_agents, [])
    
    def test_cleanup_old_assignments(self):
        """测试清理过期的任务分配记录"""
        # 设置一个旧的任务分配记录
        self.backup_strategy.task_assignments['old_task'] = {
            'primary_agent': 'agent1',
            'backup_agents': ['agent3', 'agent5'],
            'assignment_time': time.time() - 7200  # 2小时前
        }
        
        # 设置一个新的任务分配记录
        self.backup_strategy.task_assignments['new_task'] = {
            'primary_agent': 'agent2',
            'backup_agents': ['agent4'],
            'assignment_time': time.time() - 1800  # 30分钟前
        }
        
        # 清理超过1小时的任务分配记录
        self.backup_strategy.cleanup_old_assignments(max_age=3600)
        
        # 验证旧的任务分配记录已被清理
        self.assertNotIn('old_task', self.backup_strategy.task_assignments)
        
        # 验证新的任务分配记录仍然存在
        self.assertIn('new_task', self.backup_strategy.task_assignments)

if __name__ == '__main__':
    unittest.main()