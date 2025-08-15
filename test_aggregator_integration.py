import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.enhanced_aggregator import EnhancedAggregator
from core.enhanced_router import EnhancedRouter
from core.coordinator import Coordinator

class TestAggregatorIntegration(unittest.TestCase):
    """测试增强型聚合器与其他系统组件的集成"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 模拟MemoryManager
        self.memory_patcher = patch('memory.memory_manager.MemoryManager')
        self.mock_memory = self.memory_patcher.start()
        self.mock_memory_instance = MagicMock()
        self.mock_memory.return_value = self.mock_memory_instance
        self.mock_memory_instance.retrieve_memory.return_value = []
        
        # 模拟LLM调用
        self.llm_patcher = patch('utils.llm_utils.call_llm')
        self.mock_llm = self.llm_patcher.start()
        self.mock_llm.return_value = {
            "content": "这是LLM的响应",
            "confidence": 0.8
        }
        
        # 初始化组件
        self.aggregator = EnhancedAggregator()
        self.router = EnhancedRouter()
        self.coordinator = Coordinator()
        
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
    
    def tearDown(self):
        """测试后的清理工作"""
        self.memory_patcher.stop()
        self.llm_patcher.stop()
    
    def test_aggregator_with_coordinator(self):
        """测试聚合器与协调器的集成"""
        # 模拟协调器获取多个Agent响应
        self.coordinator.get_agent_responses = MagicMock(return_value=self.test_responses)
        
        # 模拟用户查询
        query = "什么是Python?"
        
        # 通过协调器获取响应
        responses = self.coordinator.get_agent_responses(query)
        
        # 使用聚合器聚合响应
        final_response = self.aggregator.aggregate_responses(responses, query)
        
        # 验证结果
        self.assertTrue(final_response)
        self.assertTrue("Python" in final_response)
        
        # 验证冲突解决
        if self.aggregator.config["conflict_resolution"]["enabled"]:
            self.assertTrue("解释型" in final_response)
    
    def test_aggregator_with_router(self):
        """测试聚合器与路由器的集成"""
        # 模拟路由器确定Agent
        self.router.determine_agent = MagicMock(return_value="multi_agent")
        self.router.route_query = MagicMock(return_value=self.test_responses)
        
        # 模拟用户查询
        query = "什么是Python?"
        context = {"history": []}
        
        # 通过路由器确定Agent并路由查询
        agent = self.router.determine_agent(query, context)
        responses = self.router.route_query(query, agent, context)
        
        # 使用聚合器聚合响应
        final_response = self.aggregator.aggregate_responses(responses, query)
        
        # 验证结果
        self.assertTrue(final_response)
        self.assertTrue("Python" in final_response)
    
    def test_end_to_end_flow(self):
        """测试端到端流程"""
        # 模拟路由器和协调器
        self.router.determine_agent = MagicMock(return_value="multi_agent")
        self.coordinator.execute_strategy = MagicMock(return_value=self.test_responses)
        
        # 模拟用户查询
        query = "什么是Python?"
        context = {"history": []}
        
        # 1. 路由器确定Agent
        agent = self.router.determine_agent(query, context)
        self.assertEqual(agent, "multi_agent")
        
        # 2. 协调器执行策略获取响应
        responses = self.coordinator.execute_strategy(query, agent, context)
        self.assertEqual(len(responses), 3)
        
        # 3. 聚合器聚合响应
        final_response = self.aggregator.aggregate_responses(responses, query)
        
        # 验证结果
        self.assertTrue(final_response)
        self.assertTrue("Python" in final_response)
    
    def test_conflict_resolution_in_flow(self):
        """测试流程中的冲突解决"""
        # 确保冲突解决已启用
        self.aggregator.config["conflict_resolution"]["enabled"] = True
        
        # 模拟路由器和协调器
        self.router.determine_agent = MagicMock(return_value="multi_agent")
        self.coordinator.execute_strategy = MagicMock(return_value=self.test_responses)
        
        # 模拟用户查询
        query = "Python是解释型语言还是编译型语言?"
        context = {"history": []}
        
        # 执行流程
        agent = self.router.determine_agent(query, context)
        responses = self.coordinator.execute_strategy(query, agent, context)
        final_response = self.aggregator.aggregate_responses(responses, query)
        
        # 验证结果
        self.assertTrue(final_response)
        self.assertTrue("Python" in final_response)
        self.assertTrue("解释型" in final_response)
        self.assertTrue("冲突" in final_response)
    
    def test_feedback_mechanism(self):
        """测试反馈机制"""
        # 模拟用户查询和响应
        query = "什么是Python?"
        response = "Python是一种编程语言"
        
        # 模拟用户反馈
        feedback = {
            "rating": 4,
            "comment": "回答很好，但有一些不准确的地方",
            "conflict_detected": True,
            "conflict_resolved": True
        }
        
        # 提供反馈
        self.aggregator.provide_feedback(query, response, feedback)
        
        # 验证记忆管理器的store_memory方法被调用
        self.mock_memory_instance.store_memory.assert_called_once()
        
        # 验证反馈被存储
        args = self.mock_memory_instance.store_memory.call_args[0]
        self.assertEqual(args[0], "feedback")
        self.assertEqual(args[1]["query"], query)
        self.assertEqual(args[1]["response"], response)
        self.assertEqual(args[1]["feedback"], feedback)
    
    def test_response_format_customization(self):
        """测试响应格式自定义"""
        # 设置不同的响应格式配置
        format_configs = [
            {"include_confidence": True, "include_sources": True, "include_conflict_info": True},
            {"include_confidence": False, "include_sources": True, "include_conflict_info": True},
            {"include_confidence": True, "include_sources": False, "include_conflict_info": True},
            {"include_confidence": True, "include_sources": True, "include_conflict_info": False},
        ]
        
        for config in format_configs:
            # 更新配置
            self.aggregator.config["response_format"] = config
            
            # 聚合响应
            final_response = self.aggregator.aggregate_responses(self.test_responses, "什么是Python?")
            
            # 验证结果
            self.assertTrue(final_response)
            self.assertTrue("Python" in final_response)
            
            # 验证格式
            if config["include_confidence"]:
                self.assertTrue("置信度" in final_response)
            else:
                self.assertFalse("置信度" in final_response)
                
            if config["include_sources"]:
                self.assertTrue("technical_agent" in final_response or "来源" in final_response)
            
            if config["include_conflict_info"] and self.aggregator.config["conflict_resolution"]["enabled"]:
                self.assertTrue("冲突" in final_response)

if __name__ == "__main__":
    unittest.main()