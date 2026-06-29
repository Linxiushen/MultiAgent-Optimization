import unittest
import time
import uuid
from memory.memory_optimizer import MemoryOptimizer

class TestMemoryOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = MemoryOptimizer()
        self.memories = {
            'mem1': {
                'id': 'mem1',
                'content': '这是第一条记忆内容',
                'timestamp': time.time() - 3600,
                'importance': 0.8,
                'access_count': 5
            },
            'mem2': {
                'id': 'mem2',
                'content': '这是第二条记忆内容，与第一条记忆内容有些相似',
                'timestamp': time.time() - 7200,
                'importance': 0.6,
                'access_count': 3
            },
            'mem3': {
                'id': 'mem3',
                'content': '这是一条完全不同的记忆内容，涉及到不同的主题',
                'timestamp': time.time() - 1800,
                'importance': 0.7,
                'access_count': 2
            },
            'mem4': {
                'id': 'mem4',
                'content': '这是第四条记忆，和第一条记忆内容非常相似，几乎重复',
                'timestamp': time.time() - 1000,
                'importance': 0.5,
                'access_count': 1
            },
            'mem5': {
                'id': 'mem5',
                'content': '这是最后一条记忆，时间戳最新',
                'timestamp': time.time() - 600,
                'importance': 0.9,
                'access_count': 4
            }
        }

    def test_calculate_importance(self):
        # 测试重要性计算
        for mem_id, memory in self.memories.items():
            importance = self.optimizer.calculate_importance(memory)
            self.assertTrue(0 <= importance <= 2.25, f"重要性分数 {importance} 超出合理范围")

        # 验证最新的记忆(高重要性和高访问量)应该有最高的分数
        importance_scores = {mem_id: self.optimizer.calculate_importance(memory) for mem_id, memory in self.memories.items()}
        max_importance_id = max(importance_scores, key=importance_scores.get)
        self.assertEqual(max_importance_id, 'mem5', f"最高重要性记忆应为 mem5，但实际是 {max_importance_id}")

    def test_retrieve_relevant_memories(self):
        # 测试相关记忆检索
        query = '记忆内容'
        results = self.optimizer.retrieve_relevant_memories(query, self.memories, top_k=3)

        self.assertEqual(len(results), 3, f"应该返回3条相关记忆，但实际返回了 {len(results)} 条")
        self.assertIn('mem1', [mid for mid, _ in results], "mem1 应该在相关记忆结果中")
        self.assertIn('mem2', [mid for mid, _ in results], "mem2 应该在相关记忆结果中")
        self.assertIn('mem4', [mid for mid, _ in results], "mem4 应该在相关记忆结果中")

        # 测试不相关的查询
        irrelevant_query = '不相关的查询内容'
        irrelevant_results = self.optimizer.retrieve_relevant_memories(irrelevant_query, self.memories)
        self.assertEqual(len(irrelevant_results), 0, f"不相关查询应该返回0条结果，但实际返回了 {len(irrelevant_results)} 条")

    def test_compress_memories(self):
        # 测试记忆压缩
        compressed_memories, deleted = self.optimizer.compress_memories(self.memories)

        # mem1, mem2和mem4应该被压缩合并
        self.assertLess(len(compressed_memories), len(self.memories), "压缩后的记忆数量应该减少")
        self.assertTrue(len(deleted) >= 2, f"至少应该删除2条记忆，但实际删除了 {len(deleted)} 条")

        # 验证mem3和mem5应该保留
        self.assertIn('mem3', compressed_memories, "mem3 应该保留在压缩后的记忆中")
        self.assertIn('mem5', compressed_memories, "mem5 应该保留在压缩后的记忆中")

    def test_prioritize_memories(self):
        # 测试记忆优先级排序
        prioritized = self.optimizer.prioritize_memories(self.memories)

        self.assertEqual(len(prioritized), len(self.memories), "优先级排序后记忆数量应该不变")
        self.assertEqual(prioritized[0][0], 'mem5', f"优先级最高的记忆应为 mem5，但实际是 {prioritized[0][0]}")

    def test_forget_low_priority_memories(self):
        # 测试遗忘低优先级记忆
        to_forget = self.optimizer.forget_low_priority_memories(self.memories, keep_ratio=0.6)

        self.assertEqual(len(to_forget), 2, f"应该遗忘2条记忆，但实际遗忘了 {len(to_forget)} 条")
        self.assertIn('mem2', to_forget, "mem2 应该被遗忘")
        self.assertIn('mem4', to_forget, "mem4 应该被遗忘")

    def test_optimize_memory_storage(self):
        # 测试全面优化记忆存储
        optimized_memories, deleted = self.optimizer.optimize_memory_storage(self.memories)

        self.assertLess(len(optimized_memories), len(self.memories), "优化后的记忆数量应该减少")
        self.assertTrue(len(deleted) >= 2, f"至少应该删除2条记忆，但实际删除了 {len(deleted)} 条")

        # 验证重要记忆保留
        self.assertIn('mem5', optimized_memories, "mem5 应该保留")
        self.assertIn('mem1', optimized_memories, "mem1 应该保留")
        self.assertIn('mem3', optimized_memories, "mem3 应该保留")

if __name__ == '__main__':
    unittest.main()