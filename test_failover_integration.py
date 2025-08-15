import unittest
import time
import threading
from unittest.mock import MagicMock, patch
from distributed.coordinator import DistributedCoordinator
from distributed.failover_strategy import FailoverStrategy, BackupAgentStrategy

class TestFailoverIntegration(unittest.TestCase):
    """测试故障转移机制的集成测试"""
    
    def setUp(self):
        # 创建协调器实例
        self.coordinator = DistributedCoordinator()
        
        # 模拟配置
        self.coordinator.config = {
            "agent_heartbeat_interval": 5,
            "task_timeout": 30,
            "load_balancing_strategy": "failover",
            "max_retries": 3,
            "failover_config": {
                "enabled": True,
                "priority_based_reassignment": True,
                "preserve_task_priority": True,
                "max_reassignment_attempts": 3,
                "use_failover_strategy": True,
                "failover_strategy_type": "failover"
            }
        }
        
        # 初始化故障事件记录
        self.coordinator.failure_history = []
        self.coordinator._log_failure_event = MagicMock()
        
        # 注册测试Agent
        self.coordinator.register_agent('agent1', 'worker', ['task_type1', 'task_type2'], 'http://agent1')
        self.coordinator.register_agent('agent2', 'worker', ['task_type1', 'task_type3'], 'http://agent2')
        self.coordinator.register_agent('agent3', 'worker', ['task_type2', 'task_type3'], 'http://agent3')
        
        # 更新Agent状态
        self.coordinator.agents['agent1']['task_count'] = 2
        self.coordinator.agents['agent2']['task_count'] = 1
        self.coordinator.agents['agent3']['task_count'] = 0
    
    def test_agent_timeout_detection(self):
        """测试Agent超时检测"""
        # 模拟Agent1心跳超时
        self.coordinator.agents['agent1']['last_heartbeat'] = time.time() - 100  # 设置为很久以前
        
        # 手动调用监控函数
        with patch('time.sleep'):  # 防止线程睡眠
            self.coordinator._monitor_agents()
        
        # 验证Agent1被标记为不活跃
        self.assertEqual(self.coordinator.agents['agent1']['status'], 'inactive')
        
        # 验证调用了故障处理函数
        self.coordinator._log_failure_event.assert_called()
    
    def test_task_reassignment(self):
        """测试任务重新分配"""
        # 创建一个分配给agent1的任务
        task = {
            'id': 'task1',
            'type': 'task_type1',
            'assigned_to': 'agent1',
            'priority': 5,
            'created_at': time.time()
        }
        
        # 将任务加入队列
        self.coordinator.tasks.put((5, time.time(), task))
        
        # 模拟agent1心跳超时
        self.coordinator.agents['agent1']['last_heartbeat'] = time.time() - 100
        self.coordinator.agents['agent1']['status'] = 'inactive'
        
        # 调用故障处理函数
        self.coordinator._handle_timeout_agent('agent1')
        
        # 验证任务被重新分配
        self.assertFalse(self.coordinator.tasks.empty())
        priority, _, reassigned_task = self.coordinator.tasks.get()
        
        # 验证任务被分配给了另一个Agent
        self.assertNotEqual(reassigned_task['assigned_to'], 'agent1')
        self.assertIn(reassigned_task['assigned_to'], ['agent2', 'agent3'])
        
        # 验证任务包含重新分配信息
        self.assertEqual(reassigned_task['reassignment_count'], 1)
        self.assertIn('agent1', reassigned_task['previous_agents'])
    
    def test_resource_lock_release(self):
        """测试资源锁释放"""
        # 模拟agent1持有资源锁
        self.coordinator.resource_locks['resource1'] = ('agent1', time.time())
        self.coordinator.resource_locks['resource2'] = ('agent1', time.time())
        self.coordinator.resource_locks['resource3'] = ('agent2', time.time())
        
        # 模拟agent1心跳超时
        self.coordinator.agents['agent1']['last_heartbeat'] = time.time() - 100
        self.coordinator.agents['agent1']['status'] = 'inactive'
        
        # 调用故障处理函数
        self.coordinator._handle_timeout_agent('agent1')
        
        # 验证agent1持有的资源锁被释放
        self.assertNotIn('resource1', self.coordinator.resource_locks)
        self.assertNotIn('resource2', self.coordinator.resource_locks)
        self.assertIn('resource3', self.coordinator.resource_locks)  # agent2的锁不应被释放
    
    def test_failover_strategy_integration(self):
        """测试故障转移策略集成"""
        # 设置使用故障转移策略
        self.coordinator.config['failover_config']['use_failover_strategy'] = True
        
        # 创建三个不同优先级的任务
        high_task = {
            'id': 'high_task',
            'type': 'task_type1',
            'assigned_to': 'agent1',
            'priority': 9,
            'created_at': time.time()
        }
        
        medium_task = {
            'id': 'medium_task',
            'type': 'task_type1',
            'assigned_to': 'agent1',
            'priority': 5,
            'created_at': time.time()
        }
        
        low_task = {
            'id': 'low_task',
            'type': 'task_type2',
            'assigned_to': 'agent1',
            'priority': 2,
            'created_at': time.time()
        }
        
        # 将任务加入队列
        self.coordinator.tasks.put((9, time.time(), high_task))
        self.coordinator.tasks.put((5, time.time(), medium_task))
        self.coordinator.tasks.put((2, time.time(), low_task))
        
        # 模拟agent1心跳超时
        self.coordinator.agents['agent1']['last_heartbeat'] = time.time() - 100
        self.coordinator.agents['agent1']['status'] = 'inactive'
        
        # 调用故障处理函数
        with patch('distributed.failover_strategy.FailoverStrategy.select_agent') as mock_select:
            # 模拟故障转移策略的选择结果
            mock_select.side_effect = ['agent2', 'agent3', 'agent3']
            
            self.coordinator._handle_timeout_agent('agent1')
        
        # 验证故障转移策略被调用了3次（对应3个任务）
        self.assertEqual(mock_select.call_count, 3)
        
        # 验证任务被重新分配
        self.assertEqual(self.coordinator.tasks.qsize(), 3)
    
    def test_multiple_agent_failures(self):
        """测试多个Agent同时故障"""
        # 创建分配给不同Agent的任务
        task1 = {
            'id': 'task1',
            'type': 'task_type1',
            'assigned_to': 'agent1',
            'priority': 5,
            'created_at': time.time()
        }
        
        task2 = {
            'id': 'task2',
            'type': 'task_type3',
            'assigned_to': 'agent2',
            'priority': 5,
            'created_at': time.time()
        }
        
        # 将任务加入队列
        self.coordinator.tasks.put((5, time.time(), task1))
        self.coordinator.tasks.put((5, time.time(), task2))
        
        # 模拟两个Agent同时心跳超时
        self.coordinator.agents['agent1']['last_heartbeat'] = time.time() - 100
        self.coordinator.agents['agent1']['status'] = 'inactive'
        self.coordinator.agents['agent2']['last_heartbeat'] = time.time() - 100
        self.coordinator.agents['agent2']['status'] = 'inactive'
        
        # 处理第一个Agent故障
        self.coordinator._handle_timeout_agent('agent1')
        
        # 处理第二个Agent故障
        self.coordinator._handle_timeout_agent('agent2')
        
        # 验证任务被重新分配给剩余的Agent
        self.assertFalse(self.coordinator.tasks.empty())
        
        # 获取重新分配的任务
        priority1, _, reassigned_task1 = self.coordinator.tasks.get()
        priority2, _, reassigned_task2 = self.coordinator.tasks.get()
        
        # 验证两个任务都被分配给了agent3（唯一剩余的活跃Agent）
        self.assertEqual(reassigned_task1['assigned_to'], 'agent3')
        self.assertEqual(reassigned_task2['assigned_to'], 'agent3')
    
    def test_backup_agent_strategy(self):
        """测试备份Agent策略"""
        # 设置使用备份Agent策略
        self.coordinator.config['failover_config']['use_failover_strategy'] = True
        self.coordinator.config['failover_config']['failover_strategy_type'] = 'backup_agent'
        
        # 创建备份Agent策略实例
        with patch('distributed.distributed_strategy.StrategyFactory.create_strategy') as mock_create:
            # 创建备份Agent策略的模拟实例
            mock_backup_strategy = MagicMock(spec=BackupAgentStrategy)
            mock_backup_strategy.select_agent.return_value = 'agent3'
            mock_create.return_value = mock_backup_strategy
            
            # 重新设置策略
            self.coordinator.set_strategy('backup_agent')
        
        # 创建一个任务
        task = {
            'id': 'task1',
            'type': 'task_type1',
            'assigned_to': 'agent1',
            'priority': 5,
            'created_at': time.time()
        }
        
        # 将任务加入队列
        self.coordinator.tasks.put((5, time.time(), task))
        
        # 模拟agent1心跳超时
        self.coordinator.agents['agent1']['last_heartbeat'] = time.time() - 100
        self.coordinator.agents['agent1']['status'] = 'inactive'
        
        # 调用故障处理函数
        self.coordinator._handle_timeout_agent('agent1')
        
        # 验证任务被重新分配
        self.assertFalse(self.coordinator.tasks.empty())
        priority, _, reassigned_task = self.coordinator.tasks.get()
        
        # 验证任务被分配给了备份Agent
        self.assertEqual(reassigned_task['assigned_to'], 'agent3')

if __name__ == '__main__':
    unittest.main()