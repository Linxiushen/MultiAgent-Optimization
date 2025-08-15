"""
聚合器模块测试文件
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.aggregator import Aggregator
from memory.memory_manager import MemoryManager


class TestAggregator(unittest.TestCase):
    """聚合器测试类"""
    
    def setUp(self):
        """测试初始化"""
        # 创建模拟的记忆管理器
        self.memory_manager = Mock(spec=MemoryManager)
        # 创建聚合器实例
        self.aggregator = Aggregator()
        self.aggregator.memory_manager = self.memory_manager
    
    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.aggregator, Aggregator)
        self.assertEqual(self.aggregator.memory_manager, self.memory_manager)
    
    def test_weighted_merge(self):
        """测试加权合并策略"""
        responses = [
            {"content": "Response 1", "confidence": 0.8, "source": "agent1"},
            {"content": "Response 2", "confidence": 0.6, "source": "agent2"},
            {"content": "Response 3", "confidence": 0.9, "source": "agent3"}
        ]
        
        result = self.aggregator._weighted_merge(responses)
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)
    
    def test_confidence_based_selection(self):
        """测试基于置信度的选择策略"""
        responses = [
            {"content": "Response 1", "confidence": 0.8, "source": "agent1"},
            {"content": "Response 2", "confidence": 0.6, "source": "agent2"},
            {"content": "Response 3", "confidence": 0.9, "source": "agent3"}
        ]
        
        result = self.aggregator._confidence_based_selection(responses)
        self.assertEqual(result["content"], "Response 3")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["source"], "agent3")
    
    def test_simple_merge(self):
        """测试简单合并策略"""
        responses = [
            {"content": "Response 1", "confidence": 0.8, "source": "agent1"},
            {"content": "Response 2", "confidence": 0.6, "source": "agent2"},
            {"content": "Response 3", "confidence": 0.9, "source": "agent3"}
        ]
        
        result = self.aggregator._simple_merge(responses)
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)
    
    def test_merge_responses(self):
        """测试响应融合方法"""
        responses = [
            {"content": "Response 1", "confidence": 0.8, "source": "agent1"},
            {"content": "Response 2", "confidence": 0.6, "source": "agent2"},
            {"content": "Response 3", "confidence": 0.9, "source": "agent3"}
        ]
        
        result = self.aggregator.fuse_responses(responses)
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)
    
    def test_generate_final_response(self):
        """测试生成最终响应方法"""
        responses = [
            {"content": "Response 1", "confidence": 0.8, "source": "agent1"},
            {"content": "Response 2", "confidence": 0.6, "source": "agent2"},
            {"content": "Response 3", "confidence": 0.9, "source": "agent3"}
        ]
        query = "Test query"
        
        # 模拟记忆管理器的retrieve_memory方法
        self.memory_manager.retrieve_memory.return_value = []
        
        fused_response = self.aggregator.fuse_responses(responses)
        result = self.aggregator.generate_final_response(fused_response, query)
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)


def main():
    """主函数"""
    unittest.main()


if __name__ == "__main__":
    main()