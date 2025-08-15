import unittest
import time
import os
import tempfile
from memory.memory_optimizer_enhanced import MemoryOptimizerEnhanced

class TestMemoryOptimizerEnhanced(unittest.TestCase):
    def setUp(self):
        self.optimizer = MemoryOptimizerEnhanced()
        self.test_memories = {
            'mem1': {
                'content': '这是第一条记忆内容，关于人工智能的发展历史。人工智能的概念最早可以追溯到20世纪50年代。',
                'timestamp': time.time() - 3600,
                'importance': 0.8,
                'access_count': 5,
                'tags': ['ai', 'history']
            },
            'mem2': {
                'content': '这是第二条记忆内容，关于机器学习的算法。常见的机器学习算法包括决策树、支持向量机和神经网络。',
                'timestamp': time.time() - 1800,
                'importance': 0.9,
                'access_count': 3,
                'tags': ['machine_learning', 'algorithms']
            },
            'mem3': {
                'content': '这是第三条记忆内容，关于深度学习的应用。深度学习在计算机视觉和自然语言处理方面取得了重大突破。',
                'timestamp': time.time() - 1200,
                'importance': 0.7,
                'access_count': 4,
                'tags': ['deep_learning', 'applications']
            },
            'mem4': {
                'content': '这是第四条记忆内容，关于人工智能的伦理问题。随着AI技术的发展，伦理问题日益受到关注。',
                'timestamp': time.time() - 600,
                'importance': 0.6,
                'access_count': 2,
                'tags': ['ai', 'ethics']
            },
            'mem5': {
                'content': '这是第五条记忆内容，关于神经网络的结构。神经网络由输入层、隐藏层和输出层组成。',
                'timestamp': time.time() - 300,
                'importance': 0.85,
                'access_count': 6,
                'tags': ['neural_networks', 'structure']
            },
            'mem6': {
                'content': '我很高兴今天的项目进展顺利！所有测试都通过了，新功能也实现了。',
                'timestamp': time.time() - 100,
                'importance': 0.75,
                'access_count': 1,
                'tags': ['project', 'positive']
            },
            'mem7': {
                'content': '我很失望，今天的会议被取消了，项目进度可能会延迟。',
                'timestamp': time.time() - 50,
                'importance': 0.7,
                'access_count': 1,
                'tags': ['project', 'negative']
            }
        }

    def test_calculate_importance(self):
        # 测试重要性计算
        for mid, memory in self.test_memories.items():
            importance = self.optimizer.calculate_importance(memory)
            self.assertIsInstance(importance, float)
            self.assertGreaterEqual(importance, 0)
            self.assertLessEqual(importance, 2.0)  # 合理的上限

        # 特别测试情感因素的影响
        positive_memory = self.test_memories['mem6']
        negative_memory = self.test_memories['mem7']

        positive_importance = self.optimizer.calculate_importance(positive_memory)
        negative_importance = self.optimizer.calculate_importance(negative_memory)

        # 积极情感的记忆应该比消极情感的记忆重要性更高
        # 即使它们的基础重要性和时间戳相似
        self.assertGreater(positive_importance, negative_importance)

    def test_cluster_memories(self):
        # 测试记忆聚类
        clusters = self.optimizer.cluster_memories(self.test_memories)

        # 检查聚类结果是否合理
        self.assertIsInstance(clusters, dict)
        self.assertGreater(len(clusters), 0)
        self.assertLessEqual(len(clusters), self.optimizer.config['cluster_count'])

        # 检查所有记忆都被分配到了聚类
        all_clustered_mems = []
        for cluster_id, mem_ids in clusters.items():
            all_clustered_mems.extend(mem_ids)
        self.assertEqual(set(all_clustered_mems), set(self.test_memories.keys()))

    def test_generate_summary(self):
        # 测试摘要生成
        for mid, memory in self.test_memories.items():
            summary = self.optimizer.generate_summary(memory['content'])
            self.assertIsInstance(summary, str)
            self.assertLessEqual(len(summary), len(memory['content']))

        # 特别测试长文本的摘要
        long_text = "这是一个非常长的文本，包含多个句子。" * 10
        summary = self.optimizer.generate_summary(long_text)
        self.assertIsInstance(summary, str)
        sentences = summary.split('. ')
        self.assertLessEqual(len(sentences), self.optimizer.config['summary_length'] + 1)

    def test_analyze_memory_sentiment(self):
        # 测试情感分析
        positive_memory = self.test_memories['mem6']
        negative_memory = self.test_memories['mem7']

        positive_sentiment = self.optimizer.analyze_memory_sentiment(positive_memory)
        negative_sentiment = self.optimizer.analyze_memory_sentiment(negative_memory)

        # 检查情感分析结果格式
        self.assertIsInstance(positive_sentiment, dict)
        self.assertIn('neg', positive_sentiment)
        self.assertIn('neu', positive_sentiment)
        self.assertIn('pos', positive_sentiment)
        self.assertIn('compound', positive_sentiment)

        # 积极文本应该有更高的正面分数和复合分数
        self.assertGreater(positive_sentiment['pos'], negative_sentiment['pos'])
        self.assertGreater(positive_sentiment['compound'], negative_sentiment['compound'])

        # 消极文本应该有更高的负面分数
        self.assertGreater(negative_sentiment['neg'], positive_sentiment['neg'])

    def test_get_memory_connections(self):
        # 首先需要创建向量
        self.optimizer.cluster_memories(self.test_memories)

        # 测试记忆关联
        mem1_connections = self.optimizer.get_memory_connections('mem1', self.test_memories)
        mem2_connections = self.optimizer.get_memory_connections('mem2', self.test_memories)

        self.assertIsInstance(mem1_connections, list)
        self.assertIsInstance(mem2_connections, list)

        # mem1是关于AI历史的，应该与mem4(AI伦理)关联更紧密
        mem1_related_ids = [mid for mid, _ in mem1_connections]
        self.assertIn('mem4', mem1_related_ids)

        # mem2是关于机器学习算法的，应该与mem5(神经网络结构)关联更紧密
        mem2_related_ids = [mid for mid, _ in mem2_connections]
        self.assertIn('mem5', mem2_related_ids)

    def test_export_import_memory_snapshot(self):
        # 测试导出和导入记忆快照
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8') as temp_file:
            temp_file_path = temp_file.name

        try:
            # 导出快照
            export_success = self.optimizer.export_memory_snapshot(self.test_memories, temp_file_path)
            self.assertTrue(export_success)

            # 导入快照
            imported_memories = self.optimizer.import_memory_snapshot(temp_file_path)
            self.assertIsInstance(imported_memories, dict)
            self.assertEqual(len(imported_memories), len(self.test_memories))

            # 检查导入的记忆是否正确
            for mid, memory in self.test_memories.items():
                self.assertIn(mid, imported_memories)
                imported_memory = imported_memories[mid]
                for key, value in memory.items():
                    self.assertEqual(value, imported_memory.get(key))
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_auto_optimize(self):
        # 测试自动优化
        # 首先设置最后优化时间为很久以前
        self.optimizer.last_optimization_time = 0

        # 执行自动优化
        optimized_memories, deleted = self.optimizer.auto_optimize(self.test_memories, interval=0)

        self.assertIsInstance(optimized_memories, dict)
        self.assertIsInstance(deleted, list)

        # 优化后的记忆数量应该少于或等于原始数量
        self.assertLessEqual(len(optimized_memories), len(self.test_memories))

        # 被删除的记忆不应该在优化后的记忆中
        for mid in deleted:
            self.assertNotIn(mid, optimized_memories)

if __name__ == '__main__':
    unittest.main()