import unittest
import time
import uuid
from distributed.coordinator import DistributedCoordinator

class TestDistributedCoordinator(unittest.TestCase):
    def setUp(self):
        # 获取协调器单例实例
        self.coordinator = DistributedCoordinator()
        # 重置协调器状态
        self.coordinator.agents = {}
        self.coordinator.tasks = []
        self.coordinator.resource_locks = {}
        self.coordinator.message_queue = {}

    def test_register_agent(self):
        # 测试Agent注册
        agent_id = 'agent_1'
        result = self.coordinator.register_agent(
            agent_id=agent_id,
            agent_type='llm',
            capabilities=['text_generation', 'translation'],
            endpoint='localhost:8001'
        )
        self.assertTrue(result, "Agent注册失败")
        self.assertIn(agent_id, self.coordinator.agents, "注册的Agent不在Agent列表中")
        self.assertEqual(self.coordinator.agents[agent_id]['type'], 'llm', "Agent类型不正确")
        self.assertEqual(self.coordinator.agents[agent_id]['status'], 'active', "Agent状态应为active")

    def test_unregister_agent(self):
        # 测试Agent注销
        agent_id = 'agent_2'
        self.coordinator.register_agent(
            agent_id=agent_id,
            agent_type='memory',
            capabilities=['memory_retrieval'],
            endpoint='localhost:8002'
        )
        result = self.coordinator.unregister_agent(agent_id)
        self.assertTrue(result, "Agent注销失败")
        self.assertNotIn(agent_id, self.coordinator.agents, "注销的Agent仍在Agent列表中")

    def test_heartbeat(self):
        # 测试心跳更新
        agent_id = 'agent_3'
        self.coordinator.register_agent(
            agent_id=agent_id,
            agent_type='router',
            capabilities=['task_routing'],
            endpoint='localhost:8003'
        )
        old_heartbeat = self.coordinator.agents[agent_id]['last_heartbeat']
        time.sleep(1)
        result = self.coordinator.heartbeat(agent_id)
        self.assertTrue(result, "心跳更新失败")
        self.assertGreater(self.coordinator.agents[agent_id]['last_heartbeat'], old_heartbeat, "心跳时间戳未更新")

    def test_assign_task(self):
        # 测试任务分配
        # 注册两个具有相同能力的Agent
        self.coordinator.register_agent(
            agent_id='agent_4',
            agent_type='llm',
            capabilities=['text_generation'],
            endpoint='localhost:8004'
        )
        self.coordinator.register_agent(
            agent_id='agent_5',
            agent_type='llm',
            capabilities=['text_generation'],
            endpoint='localhost:8005'
        )

        # 分配任务
        task_id = self.coordinator.assign_task(
            task_type='text_generation',
            task_data={'prompt': '生成一段文本'},
            priority=1
        )
        self.assertIsNotNone(task_id, "任务分配失败")
        self.assertEqual(len(self.coordinator.tasks), 1, "任务未添加到任务队列")

        # 验证任务被分配给了其中一个Agent
        assigned_agent = None
        for _, _, task in self.coordinator.tasks:
            if task['id'] == task_id:
                assigned_agent = task['assigned_to']
                break
        self.assertIsNotNone(assigned_agent, "任务未分配给任何Agent")
        self.assertIn(assigned_agent, ['agent_4', 'agent_5'], "任务分配给了错误的Agent")

    def test_send_message(self):
        # 测试消息传递
        self.coordinator.register_agent(
            agent_id='agent_6',
            agent_type='sender',
            capabilities=['message_sending'],
            endpoint='localhost:8006'
        )
        self.coordinator.register_agent(
            agent_id='agent_7',
            agent_type='receiver',
            capabilities=['message_receiving'],
            endpoint='localhost:8007'
        )

        # 发送消息
        result = self.coordinator.send_message(
            from_agent='agent_6',
            to_agent='agent_7',
            message_type='info',
            content={'data': '测试消息内容'}
        )
        self.assertTrue(result, "消息发送失败")

        # 检查消息是否被正确接收
        messages = self.coordinator.get_messages('agent_7')
        self.assertEqual(len(messages), 1, "接收的消息数量不正确")
        self.assertEqual(messages[0]['from'], 'agent_6', "消息发送者不正确")
        self.assertEqual(messages[0]['type'], 'info', "消息类型不正确")
        self.assertEqual(messages[0]['content'], {'data': '测试消息内容'}, "消息内容不正确")

    def test_acquire_release_lock(self):
        # 测试资源锁获取和释放
        self.coordinator.register_agent(
            agent_id='agent_8',
            agent_type='lock_user',
            capabilities=['lock_operations'],
            endpoint='localhost:8008'
        )

        # 获取锁
        result = self.coordinator.acquire_lock('resource_1', 'agent_8')
        self.assertTrue(result, "获取锁失败")
        self.assertIn('resource_1', self.coordinator.resource_locks, "资源锁未记录")
        self.assertEqual(self.coordinator.resource_locks['resource_1'][0], 'agent_8', "锁持有者不正确")

        # 尝试再次获取同一锁
        result2 = self.coordinator.acquire_lock('resource_1', 'agent_8')
        self.assertFalse(result2, "不应该能重复获取同一锁")

        # 释放锁
        result3 = self.coordinator.release_lock('resource_1', 'agent_8')
        self.assertTrue(result3, "释放锁失败")
        self.assertNotIn('resource_1', self.coordinator.resource_locks, "资源锁未释放")

    def test_get_cluster_status(self):
        # 测试获取集群状态
        self.coordinator.register_agent(
            agent_id='agent_9',
            agent_type='status_agent',
            capabilities=['status_reporting'],
            endpoint='localhost:8009'
        )

        status = self.coordinator.get_cluster_status()
        self.assertEqual(status['total_agents'], 1, "集群Agent总数不正确")
        self.assertEqual(status['active_agents'], 1, "活跃Agent数量不正确")
        self.assertEqual(status['inactive_agents'], 0, "不活跃Agent数量不正确")
        self.assertEqual(status['pending_tasks'], 0, "待处理任务数量不正确")

if __name__ == '__main__':
    unittest.main()