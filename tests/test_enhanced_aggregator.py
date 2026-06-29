import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.enhanced_aggregator import EnhancedAggregator

class TestEnhancedAggregator(unittest.TestCase):
    """测试增强型聚合器"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 模拟MemoryManager
        self.memory_patcher = patch('memory.memory_manager.MemoryManager')
        self.mock_memory = self.memory_patcher.start()
        self.mock_memory_instance = MagicMock()
        self.mock_memory.return_value = self.mock_memory_instance
        self.mock_memory_instance.retrieve_memory.return_value = []
        
        # 初始化增强型聚合器
        self.aggregator = EnhancedAggregator()
        
        # 测试数据
        self.test_responses = [
            {
                "agent_id": "technical_agent",
                "content": "Python是一种解释型语言，执行速度较慢。它的优点是简单易学，适合初学者。Python的主要应用领域包括数据分析、人工智能和Web开发。",
                "confidence": 0.8
            },
            {
                "agent_id": "creative_agent",
                "content": "Python是一种编译型语言，执行速度很快。它的优点是简单易学，适合初学者。Python主要用于游戏开发和系统编程。",
                "confidence": 0.6
            },
            {
                "agent_id": "memory_agent",
                "content": "根据历史记录，Python是一种解释型语言，执行速度中等。它适合初学者学习，主要用于数据分析和Web开发。我建议你先学习Python基础语法，然后专注于特定领域的应用。",
                "confidence": 0.7
            }
        ]
        
        self.single_response = [
            {
                "agent_id": "technical_agent",
                "content": "Python是一种解释型语言，执行速度较慢。",
                "confidence": 0.8
            }
        ]
    
    def tearDown(self):
        """测试后的清理工作"""
        self.memory_patcher.stop()
    
    def test_load_config(self):
        """测试配置加载"""
        # 测试默认配置
        self.assertEqual(self.aggregator.config["fusion_strategy"], "weighted_merge")
        self.assertTrue(self.aggregator.config["conflict_resolution"]["enabled"])
        
        # 测试从文件加载配置
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data='{"fusion_strategy": "confidence_based"}')):            
            aggregator = EnhancedAggregator("dummy_path")
            self.assertEqual(aggregator.config["fusion_strategy"], "confidence_based")
    
    def test_fuse_responses_single(self):
        """测试单个响应的融合"""
        result = self.aggregator.fuse_responses(self.single_response)
        
        # 验证结果
        self.assertEqual(result, self.single_response[0])
    
    def test_fuse_responses_with_conflicts(self):
        """测试有冲突的响应融合"""
        # 确保冲突解决已启用
        self.aggregator.config["conflict_resolution"]["enabled"] = True
        
        result = self.aggregator.fuse_responses(self.test_responses)
        
        # 验证结果
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)
        self.assertIn("conflict_info", result)
        
        # 验证冲突信息
        self.assertTrue(result["conflict_info"]["detected"])
        self.assertTrue(result["conflict_info"]["count"] > 0)
    
    def test_fuse_responses_without_conflicts(self):
        """测试无冲突的响应融合"""
        # 禁用冲突解决
        self.aggregator.config["conflict_resolution"]["enabled"] = False
        
        result = self.aggregator.fuse_responses(self.test_responses)
        
        # 验证结果
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)
        self.assertIn("conflict_info", result)
        
        # 验证无冲突信息
        self.assertFalse(result["conflict_info"]["detected"])
    
    def test_weighted_merge(self):
        """测试加权合并策略"""
        result = self.aggregator._weighted_merge(self.test_responses)
        
        # 验证结果
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)
        
        # 验证内容包含所有响应
        for resp in self.test_responses:
            self.assertTrue(resp["content"] in result["content"])
        
        # 验证置信度是平均值
        expected_confidence = sum(resp["confidence"] for resp in self.test_responses) / len(self.test_responses)
        self.assertAlmostEqual(result["confidence"], expected_confidence)
    
    def test_confidence_based_selection(self):
        """测试基于置信度的选择策略"""
        result = self.aggregator._confidence_based_selection(self.test_responses)
        
        # 验证结果
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        
        # 验证选择了置信度最高的响应
        max_confidence_resp = max(self.test_responses, key=lambda x: x["confidence"])
        self.assertEqual(result["content"], max_confidence_resp["content"])
        self.assertEqual(result["confidence"], max_confidence_resp["confidence"])
    
    def test_simple_merge(self):
        """测试简单合并策略"""
        result = self.aggregator._simple_merge(self.test_responses)
        
        # 验证结果
        self.assertIn("content", result)
        self.assertIn("confidence", result)
        self.assertIn("sources", result)
        
        # 验证内容包含所有响应和Agent ID
        for resp in self.test_responses:
            self.assertTrue(resp["content"] in result["content"])
            self.assertTrue(resp["agent_id"] in result["content"])
    
    def test_generate_final_response(self):
        """测试最终响应生成"""
        fused_response = {
            "content": "这是融合后的内容",
            "confidence": 0.75,
            "sources": ["technical_agent", "creative_agent"],
            "conflict_info": {"detected": True, "count": 2, "types": ["factual"]}
        }
        
        # 设置响应格式配置
        self.aggregator.config["response_format"]["include_confidence"] = True
        self.aggregator.config["response_format"]["include_sources"] = True
        self.aggregator.config["response_format"]["include_conflict_info"] = True
        
        result = self.aggregator.generate_final_response(fused_response, "什么是Python?")
        
        # 验证结果
        self.assertTrue(fused_response["content"] in result)
        self.assertTrue("冲突" in result)
        self.assertTrue("technical_agent" in result)
        self.assertTrue("creative_agent" in result)
        self.assertTrue("置信度" in result)
    
    def test_aggregate_responses(self):
        """测试响应聚合"""
        result = self.aggregator.aggregate_responses(self.test_responses, "什么是Python?")
        
        # 验证结果不为空
        self.assertTrue(result)
        
        # 验证内容包含Python相关信息
        self.assertTrue("Python" in result)
    
    def test_provide_feedback(self):
        """测试提供反馈"""
        feedback = {
            "rating": 4,
            "comment": "回答很好，但有一些不准确的地方",
            "conflict_detected": True,
            "conflict_resolved": True
        }
        
        # 调用提供反馈方法
        self.aggregator.provide_feedback("什么是Python?", "Python是一种编程语言", feedback)
        
        # 验证记忆管理器的store_memory方法被调用
        self.mock_memory_instance.store_memory.assert_called_once()
        args = self.mock_memory_instance.store_memory.call_args[0]
        self.assertEqual(args[0], "feedback")

if __name__ == "__main__":
    unittest.main()