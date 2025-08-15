import unittest
import os
import json
import shutil
from memory.memory_manager import MemoryManager
from unittest.mock import patch, MagicMock
import numpy as np

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        """测试前设置"""
        # 临时目录和文件
        self.test_memory_dir = "tests/temp_memory_store"
        self.test_index_path = "tests/temp_index.faiss"

        # 确保测试目录不存在
        if os.path.exists(self.test_memory_dir):
            shutil.rmtree(self.test_memory_dir)
        if os.path.exists(self.test_index_path):
            os.remove(self.test_index_path)

        # 创建MemoryManager实例
        self.memory_manager = MemoryManager(
            memory_dir=self.test_memory_dir,
            index_path=self.test_index_path
        )

    def tearDown(self):
        """测试后清理"""
        # 删除临时目录和文件
        if os.path.exists(self.test_memory_dir):
            shutil.rmtree(self.test_memory_dir)
        if os.path.exists(self.test_index_path):
            os.remove(self.test_index_path)

    def test_init_index(self):
        """测试初始化索引"""
        # 测试新索引创建
        self.assertEqual(self.memory_manager.index.ntotal, 0)

        # 测试保存和加载索引
        self.memory_manager._save_index()
        new_memory_manager = MemoryManager(
            memory_dir=self.test_memory_dir,
            index_path=self.test_index_path
        )
        # 由于索引是空的，加载后也应该为空
        self.assertEqual(new_memory_manager.index.ntotal, 0)

    @patch('memory.memory_manager.lite_llm_infer')
    def test_store_memory(self, mock_lite_llm_infer):
        """测试存储记忆"""
        # 模拟LLM响应
        mock_embedding = [0.1] * 768
        mock_lite_llm_infer.return_value = json.dumps(mock_embedding)

        # 测试存储记忆
        content = "这是一段测试记忆内容"
        metadata = {"source": "test", "importance": "high"}
        memory_id = self.memory_manager.store_memory(content, metadata)

        # 验证索引已更新
        self.assertEqual(self.memory_manager.index.ntotal, 1)

        # 验证文件已创建
        memory_file = os.path.join(self.test_memory_dir, f"{memory_id}.json")
        self.assertTrue(os.path.exists(memory_file))

        # 验证内容正确
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
            self.assertEqual(memory_data["id"], memory_id)
            self.assertEqual(memory_data["content"], content)
            self.assertEqual(memory_data["metadata"], metadata)
            self.assertIn("timestamp", memory_data)

    @patch('memory.memory_manager.lite_llm_infer')
    def test_retrieve_memory(self, mock_lite_llm_infer):
        """测试检索记忆"""
        # 模拟LLM响应
        mock_embedding1 = [0.1] * 768
        mock_embedding2 = [0.2] * 768
        mock_embedding_query = [0.15] * 768

        # 存储两个记忆
        mock_lite_llm_infer.return_value = json.dumps(mock_embedding1)
        memory_id1 = self.memory_manager.store_memory("这是第一段测试记忆")

        mock_lite_llm_infer.return_value = json.dumps(mock_embedding2)
        memory_id2 = self.memory_manager.store_memory("这是第二段测试记忆")

        # 检索记忆
        mock_lite_llm_infer.return_value = json.dumps(mock_embedding_query)
        retrieved_memories = self.memory_manager.retrieve_memory("测试查询")

        # 验证结果
        self.assertEqual(len(retrieved_memories), 2)
        self.assertEqual(retrieved_memories[0]["id"], memory_id1)  # 假设第一个记忆更相似
        self.assertEqual(retrieved_memories[1]["id"], memory_id2)
        self.assertIn("similarity_score", retrieved_memories[0])
        self.assertIn("similarity_score", retrieved_memories[1])

    def test_get_memory_by_id(self):
        """测试通过ID获取记忆"""
        # 先存储一个记忆
        with patch('memory.memory_manager.lite_llm_infer') as mock_lite_llm_infer:
            mock_lite_llm_infer.return_value = json.dumps([0.1] * 768)
            memory_id = self.memory_manager.store_memory("测试记忆内容")

        # 测试获取存在的记忆
        memory = self.memory_manager.get_memory_by_id(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory["id"], memory_id)
        self.assertEqual(memory["content"], "测试记忆内容")

        # 测试获取不存在的记忆
        non_existent_memory = self.memory_manager.get_memory_by_id("non_existent_id")
        self.assertIsNone(non_existent_memory)

    def test_delete_memory(self):
        """测试删除记忆"""
        # 先存储一个记忆
        with patch('memory.memory_manager.lite_llm_infer') as mock_lite_llm_infer:
            mock_lite_llm_infer.return_value = json.dumps([0.1] * 768)
            memory_id = self.memory_manager.store_memory("测试记忆内容")

        # 测试删除存在的记忆
        memory_file = os.path.join(self.test_memory_dir, f"{memory_id}.json")
        self.assertTrue(os.path.exists(memory_file))

        delete_result = self.memory_manager.delete_memory(memory_id)
        self.assertTrue(delete_result)
        self.assertFalse(os.path.exists(memory_file))

        # 测试删除不存在的记忆
        delete_result = self.memory_manager.delete_memory("non_existent_id")
        self.assertFalse(delete_result)

    def test_clear_all_memories(self):
        """测试清除所有记忆"""
        # 先存储两个记忆
        with patch('memory.memory_manager.lite_llm_infer') as mock_lite_llm_infer:
            mock_lite_llm_infer.return_value = json.dumps([0.1] * 768)
            self.memory_manager.store_memory("测试记忆1")
            self.memory_manager.store_memory("测试记忆2")

        # 验证存储成功
        self.assertEqual(len(os.listdir(self.test_memory_dir)), 2)
        self.assertEqual(self.memory_manager.index.ntotal, 2)

        # 测试清除所有记忆
        clear_result = self.memory_manager.clear_all_memories()
        self.assertTrue(clear_result)

        # 验证清除成功
        self.assertEqual(len(os.listdir(self.test_memory_dir)), 0)
        self.assertEqual(self.memory_manager.index.ntotal, 0)

if __name__ == '__main__':
    unittest.main()