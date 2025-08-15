import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.routing_strategies import (
    RoutingStrategy, 
    HistoryBasedStrategy, 
    SemanticRoutingStrategy,
    AdaptiveRoutingStrategy,
    MultiAgentRoutingStrategy,
    StrategyFactory
)

class TestRoutingStrategies(unittest.TestCase):
    def setUp(self):
        # 测试用的Agent配置
        self.agent_profiles = {
            "default_agent": {
                "description": "通用Agent，处理一般性查询",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "supported_intents": ["general", "information", "conversation"]
            },
            "technical_agent": {
                "description": "技术Agent，处理技术问题和代码相关任务",
                "model": "gpt-4",
                "temperature": 0.5,
                "supported_intents": ["technical", "coding", "debugging", "explanation"]
            },
            "creative_agent": {
                "description": "创意Agent，处理创意和文案相关任务",
                "model": "gpt-4",
                "temperature": 0.8,
                "supported_intents": ["creative", "writing", "brainstorming", "design"]
            }
        }
        
        # 测试用的上下文
        self.context = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
        ]
    
    def test_strategy_factory(self):
        """测试策略工厂创建不同类型的策略"""
        # 测试创建历史交互策略
        history_strategy = StrategyFactory.create_strategy("history")
        self.assertIsInstance(history_strategy, HistoryBasedStrategy)
        
        # 测试创建语义路由策略
        semantic_strategy = StrategyFactory.create_strategy("semantic")
        self.assertIsInstance(semantic_strategy, SemanticRoutingStrategy)
        
        # 测试创建自适应路由策略
        adaptive_strategy = StrategyFactory.create_strategy("adaptive")
        self.assertIsInstance(adaptive_strategy, AdaptiveRoutingStrategy)
        
        # 测试创建多Agent协作策略
        multi_agent_strategy = StrategyFactory.create_strategy("multi_agent")
        self.assertIsInstance(multi_agent_strategy, MultiAgentRoutingStrategy)
        
        # 测试创建未知策略类型（应返回语义路由策略）
        unknown_strategy = StrategyFactory.create_strategy("unknown")
        self.assertIsInstance(unknown_strategy, SemanticRoutingStrategy)
    
    def test_history_based_strategy(self):
        """测试基于历史交互的路由策略"""
        # 创建临时历史文件
        temp_history_file = "temp_history.json"
        config = {"history_file": temp_history_file}
        
        try:
            # 初始化策略
            strategy = HistoryBasedStrategy(config)
            
            # 测试更新历史记录
            strategy.update_history("如何编写Python代码？", "technical_agent", True, 5)
            strategy.update_history("写一个故事", "creative_agent", True, 4)
            strategy.update_history("什么是人工智能？", "default_agent", True, 3)
            
            # 测试获取用户偏好
            preferences = strategy.get_user_preference("如何编写Java代码？")
            self.assertIn("technical_agent", preferences)
            self.assertIn("creative_agent", preferences)
            self.assertIn("default_agent", preferences)
            
            # 测试选择Agent
            agent_id = strategy.select_agent("如何编写Java代码？", self.agent_profiles)
            self.assertEqual(agent_id, "technical_agent")
            
            # 测试选择Agent（创意查询）
            agent_id = strategy.select_agent("写一个关于未来的故事", self.agent_profiles)
            self.assertEqual(agent_id, "creative_agent")
        finally:
            # 清理临时文件
            if os.path.exists(temp_history_file):
                os.remove(temp_history_file)
    
    @patch('core.routing_strategies.SemanticRoutingStrategy.get_query_embedding')
    def test_semantic_routing_strategy(self, mock_get_embedding):
        """测试基于语义理解的路由策略"""
        # 模拟embedding函数
        mock_get_embedding.side_effect = lambda query: [float(i) for i in range(128)]
        
        # 配置意图模式
        config = {
            "intent_patterns": {
                "technical": ["代码", "编程", "开发", "调试"],
                "creative": ["创意", "设计", "写作", "文案"],
                "general": ["什么是", "如何", "为什么"]
            }
        }
        
        # 初始化策略
        strategy = SemanticRoutingStrategy(config)
        
        # 测试意图检测
        intent_scores = strategy.detect_intent("如何编写Python代码？")
        self.assertGreater(intent_scores.get("technical", 0), 0)
        
        # 测试选择Agent（技术查询）
        agent_id = strategy.select_agent("如何编写Python代码？", self.agent_profiles)
        self.assertEqual(agent_id, "technical_agent")
        
        # 测试选择Agent（创意查询）
        agent_id = strategy.select_agent("帮我设计一个创意广告文案", self.agent_profiles)
        self.assertEqual(agent_id, "creative_agent")
        
        # 测试选择Agent（一般查询）
        agent_id = strategy.select_agent("什么是人工智能？", self.agent_profiles)
        self.assertEqual(agent_id, "default_agent")
    
    @patch('core.routing_strategies.AdaptiveRoutingStrategy.select_strategy')
    def test_adaptive_routing_strategy(self, mock_select_strategy):
        """测试自适应路由策略"""
        # 模拟策略选择函数
        mock_select_strategy.return_value = "semantic"
        
        # 初始化策略
        strategy = AdaptiveRoutingStrategy()
        
        # 模拟语义路由策略
        semantic_mock = MagicMock()
        semantic_mock.select_agent.return_value = "technical_agent"
        strategy.strategies["semantic"] = semantic_mock
        
        # 测试选择Agent
        agent_id = strategy.select_agent("如何编写Python代码？", self.agent_profiles)
        self.assertEqual(agent_id, "technical_agent")
        
        # 测试反馈机制
        strategy.feedback("semantic", True)
        self.assertEqual(strategy.strategy_performance["semantic"]["success"], 1)
        self.assertEqual(strategy.strategy_performance["semantic"]["total"], 1)
    
    def test_multi_agent_routing_strategy(self):
        """测试多Agent协作路由策略"""
        # 初始化策略
        strategy = MultiAgentRoutingStrategy()
        
        # 测试任务分解
        sub_tasks = strategy.decompose_task("这是一个简单的查询")
        self.assertEqual(len(sub_tasks), 1)
        
        # 测试复杂任务分解
        complex_query = "请帮我编写一个Python函数来计算斐波那契数列。然后解释一下这个算法的时间复杂度。最后，给我一些优化建议。"
        sub_tasks = strategy.decompose_task(complex_query)
        self.assertGreater(len(sub_tasks), 1)
        
        # 测试为子任务分配Agent
        task_agent_pairs = strategy.select_agents_for_subtasks(sub_tasks, self.agent_profiles)
        self.assertEqual(len(task_agent_pairs), len(sub_tasks))
        
        # 测试结果聚合
        results = [
            {"task": {"order": 0}, "agent_id": "technical_agent", "response": "函数实现"},
            {"task": {"order": 1}, "agent_id": "technical_agent", "response": "时间复杂度分析"},
            {"task": {"order": 2}, "agent_id": "technical_agent", "response": "优化建议"}
        ]
        aggregated = strategy.aggregate_results(results)
        self.assertIn("函数实现", aggregated["response"])
        self.assertIn("时间复杂度分析", aggregated["response"])
        self.assertIn("优化建议", aggregated["response"])
        self.assertIn("technical_agent", aggregated["agents_used"])

if __name__ == '__main__':
    unittest.main()