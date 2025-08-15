import sys
import os
import time
import threading
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distributed.coordinator import coordinator

class TestFailoverMechanism(unittest.TestCase):
    def setUp(self):
        # 重置协调器状态
        coordinator.agents = {}
        coordinator.tasks = coordinator.tasks.__class__()
        coordinator.resource_locks = {}
        if hasattr(coordinator, 'failure_history'):
            coordinator.failure_history = []
        
        # 模拟配置
        coordinator.config = {
            'agent_heartbeat_interval': 5,
            'max_retries': 2,
            'failure_notification': {
                'enabled': True,
                'types': ['log'],
                'min_severity': 'low'
            },
            'max_failure_history': 10
        }
        
        # 注册测试Agent
        self.agent1 = 'test_agent_1'
        self.agent2 = 'test_agent_2'
        self.agent3 = 'test_agent_3'
        
        coordinator.register_agent(self.agent1, 'worker', ['task_type_1', 'task_type_2'])
        coordinator.register_agent(self.agent2, 'worker', ['task_type_1', 'task_type_3'])
        coordinator.register_agent(self.agent3, 'worker', ['task_type_2', 'task_type_3'])
        
        # 更新心跳，确保所有Agent都是活跃的
        for agent_id in [self.agent1, self.agent2, self.agent3]:
            coordinator.heartbeat(agent_id)
    
    def test_agent_timeout_detection(self):
        """测试Agent超时检测"""
        # 模拟Agent1心跳超时
        # 将Agent1的最后心跳时间设置为很久以前
        coordinator.agents[self.agent1]['last_heartbeat'] = time.time() - 100
        
        # 手动调用监控函数一次
        with patch('time.sleep'):
            coordinator._monitor_agents()
        
        # 验证Agent1被标记为不活跃
        self.assertEqual(coordinator.agents[self.agent1]['status'], 'inactive')
        
        # 验证其他Agent仍然活跃
        self.assertEqual(coordinator.agents[self.agent2]['status'], 'active')
        self.assertEqual(coordinator.agents[self.agent3]['status'], 'active')
    
    def test_task_reassignment(self):
        """测试任务重新分配"""
        # 分配任务给Agent1
        task_id = 'test_task_1'
        task_type = 'task_type_1'
        task_data = {'key': 'value'}
        
        # 模拟任务分配
        coordinator.assign_task(task_id, task_type, task_data)
        
        # 获取任务并验证分配
        assigned_agent = None
        while not coordinator.tasks.empty():
            _, _, task = coordinator.tasks.get()
            if task['id'] == task_id:
                assigned_agent = task['assigned_to']
                # 将任务重新放回队列
                coordinator.tasks.put((1, time.time(), task))
                break
        
        self.assertIsNotNone(assigned_agent, "任务未被分配")
        
        # 模拟分配的Agent超时
        coordinator.agents[assigned_agent]['last_heartbeat'] = time.time() - 100
        
        # 手动调用超时处理
        with patch('builtins.print'):
            coordinator._handle_timeout_agent(assigned_agent)
        
        # 验证任务被重新分配
        new_assigned_agent = None
        reassigned = False
        
        while not coordinator.tasks.empty():
            _, _, task = coordinator.tasks.get()
            if task['id'] == task_id:
                new_assigned_agent = task['assigned_to']
                reassigned = 'reassigned_at' in task
                break
        
        self.assertIsNotNone(new_assigned_agent, "任务未被重新分配")
        self.assertNotEqual(new_assigned_agent, assigned_agent, "任务被分配给了同一个Agent")
        self.assertTrue(reassigned, "任务未标记为已重新分配")
    
    def test_resource_lock_release(self):
        """测试资源锁释放"""
        # Agent1获取资源锁
        resource_id = 'test_resource'
        coordinator.acquire_lock(self.agent1, resource_id)
        
        # 验证资源锁已被获取
        self.assertIn(resource_id, coordinator.resource_locks)
        self.assertEqual(coordinator.resource_locks[resource_id][0], self.agent1)
        
        # 模拟Agent1超时
        coordinator.agents[self.agent1]['last_heartbeat'] = time.time() - 100
        
        # 手动调用超时处理
        with patch('builtins.print'):
            coordinator._handle_timeout_agent(self.agent1)
        
        # 验证资源锁已被释放
        self.assertNotIn(resource_id, coordinator.resource_locks)
    
    def test_failure_logging(self):
        """测试故障日志记录"""
        # 模拟Agent1超时
        coordinator.agents[self.agent1]['last_heartbeat'] = time.time() - 100
        
        # 手动调用超时处理
        with patch('builtins.print'):
            coordinator._handle_timeout_agent(self.agent1)
        
        # 验证故障事件已记录
        self.assertTrue(hasattr(coordinator, 'failure_history'))
        self.assertGreater(len(coordinator.failure_history), 0)
        
        # 验证最新的故障事件
        latest_failure = coordinator.failure_history[-1]
        self.assertEqual(latest_failure['agent_id'], self.agent1)
        self.assertEqual(latest_failure['failure_type'], 'heartbeat_timeout')
    
    def test_failure_notification(self):
        """测试故障通知"""
        # 模拟通知方法
        with patch.object(coordinator, '_send_log_notification') as mock_log_notify:
            # 手动调用故障事件记录和通知
            failure_event = {
                'agent_id': self.agent1,
                'failure_type': 'heartbeat_timeout',
                'timestamp': time.time(),
                'details': {}
            }
            
            coordinator._notify_failure(failure_event)
            
            # 验证通知方法被调用
            mock_log_notify.assert_called_once()
    
    def test_multiple_failures(self):
        """测试多次故障的严重程度升级"""
        agent_id = self.agent1
        
        # 创建多个故障事件
        for i in range(4):
            failure_event = {
                'agent_id': agent_id,
                'failure_type': 'task_failure',
                'timestamp': time.time() - (300 - i * 60),  # 在5分钟内的不同时间点
                'details': {}
            }
            coordinator.failure_history.append(failure_event)
        
        # 创建新的故障事件
        new_failure = {
            'agent_id': agent_id,
            'failure_type': 'task_failure',
            'timestamp': time.time(),
            'details': {}
        }
        
        # 检查严重程度是否升级
        severity = coordinator._determine_failure_severity(new_failure)
        self.assertEqual(severity, 'high', "多次故障后严重程度未升级")

if __name__ == '__main__':
    unittest.main()