import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.enhanced_router import EnhancedRouter
from core.routing_strategies import (
    RoutingStrategy, HistoryBasedStrategy, 
    SemanticRoutingStrategy, AdaptiveRoutingStrategy, 
    MultiAgentRoutingStrategy, StrategyFactory
)
from core.coordinator import Coordinator
from core.aggregator import Aggregator

class TestRoutingIntegration(unittest.TestCase):
    """测试路由策略与其他系统组件的集成"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 模拟LLM调用
        self.llm_patcher = patch('adapters.llm.call_model')
        self.mock_llm = self.llm_patcher.start()
        self.mock_llm.return_value = "这是模拟的LLM响应"
        
        # 模拟lite_llm_infer调用
        self.lite_llm_patcher = patch('adapters.llm.lite_llm_infer')
        self.mock_lite_llm = self.lite_llm_patcher.start()
        self.mock_lite_llm.return_value = "technical_agent"
        
        # 初始化路由器
        self.router = EnhancedRouter("configs/router_config.json")
        
        # 初始化协调器
        self.coordinator = MagicMock(spec=Coordinator)
        self.coordinator.get_available_agents.return_value = ["default_agent", "technical_agent", "creative_agent"]
        
        # 初始化聚合器
        self.aggregator = MagicMock(spec=Aggregator)
        self.aggregator.aggregate_responses.return_value = "这是聚合后的响应"
    
    def tearDown(self):
        """测试后的清理工作"""
        self.llm_patcher.stop()
        self.lite_llm_patcher.stop()
    
    def test_router_with_coordinator(self):
        """测试路由器与协调器的集成"""
        # 设置协调器的模拟行为
        self.coordinator.is_agent_available.return_value = True
        self.coordinator.assign_task.return_value = True
        
        # 将协调器注入到路由器
        self.router.coordinator = self.coordinator
        
        # 执行路由
        query = "如何使用Python处理JSON数据？"
        result = self.router.execute_route(query)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertTrue("agent_id" in result)
        self.assertTrue("strategy" in result)
        self.assertTrue("response" in result)
        
        # 验证协调器方法被调用
        self.coordinator.is_agent_available.assert_called()
        self.coordinator.assign_task.assert_called()
    
    def test_router_with_unavailable_agent(self):
        """测试当首选Agent不可用时的路由行为"""
        # 设置协调器的模拟行为，首选Agent不可用，备选Agent可用
        self.coordinator.is_agent_available.side_effect = lambda agent_id: agent_id != "technical_agent"
        self.coordinator.assign_task.return_value = True
        
        # 将协调器注入到路由器
        self.router.coordinator = self.coordinator
        
        # 执行路由
        query = "如何使用Python处理JSON数据？"
        result = self.router.execute_route(query)
        
        # 验证结果，应该选择备选Agent
        self.assertIsNotNone(result)
        self.assertNotEqual(result["agent_id"], "technical_agent")
    
    def test_router_with_aggregator(self):
        """测试路由器与聚合器的集成"""
        # 设置协调器的模拟行为
        self.coordinator.is_agent_available.return_value = True
        self.coordinator.assign_task.return_value = True
        
        # 将协调器和聚合器注入到路由器
        self.router.coordinator = self.coordinator
        self.router.aggregator = self.aggregator
        
        # 修改路由器的默认策略为多Agent协作
        original_strategy = self.router.default_strategy_name
        self.router.default_strategy_name = "multi_agent"
        
        # 确保多Agent协作策略已初始化
        if "multi_agent" not in self.router.strategy_instances:
            self.router.strategy_instances["multi_agent"] = StrategyFactory.create_strategy("multi_agent")
        
        # 模拟任务分解
        multi_agent_strategy = self.router.strategy_instances["multi_agent"]
        multi_agent_strategy.decompose_task = MagicMock(return_value=[
            {"sub_query": "子任务1", "agent_id": "technical_agent"},
            {"sub_query": "子任务2", "agent_id": "creative_agent"}
        ])
        
        # 执行路由
        query = "这是一个复杂的查询，需要多个Agent协作处理"
        result = self.router.execute_route(query)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "multi_agent")
        
        # 验证聚合器被调用
        self.aggregator.aggregate_responses.assert_called()
        
        # 恢复原始默认策略
        self.router.default_strategy_name = original_strategy
    
    def test_adaptive_strategy_integration(self):
        """测试自适应策略与其他组件的集成"""
        # 设置协调器的模拟行为
        self.coordinator.is_agent_available.return_value = True
        self.coordinator.assign_task.return_value = True
        
        # 将协调器注入到路由器
        self.router.coordinator = self.coordinator
        
        # 确保自适应策略已初始化
        if "adaptive" not in self.router.strategy_instances:
            self.router.strategy_instances["adaptive"] = StrategyFactory.create_strategy("adaptive")
        
        adaptive_strategy = self.router.strategy_instances["adaptive"]
        
        # 添加一些策略性能记录
        adaptive_strategy.update_strategy_performance("semantic", True)
        adaptive_strategy.update_strategy_performance("history", False)
        
        # 修改路由器的默认策略为自适应
        original_strategy = self.router.default_strategy_name
        self.router.default_strategy_name = "adaptive"
        
        # 执行路由
        query = "如何使用Python处理JSON数据？"
        result = self.router.execute_route(query)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "adaptive")
        
        # 恢复原始默认策略
        self.router.default_strategy_name = original_strategy
    
    def test_history_based_strategy_integration(self):
        """测试基于历史交互的策略与其他组件的集成"""
        # 设置协调器的模拟行为
        self.coordinator.is_agent_available.return_value = True
        self.coordinator.assign_task.return_value = True
        
        # 将协调器注入到路由器
        self.router.coordinator = self.coordinator
        
        # 确保历史策略已初始化
        if "history" not in self.router.strategy_instances:
            self.router.strategy_instances["history"] = StrategyFactory.create_strategy("history")
        
        history_strategy = self.router.strategy_instances["history"]
        
        # 添加一些历史交互记录
        history_strategy.update_history("如何编写Python代码？", "technical_agent", True, 5)
        history_strategy.update_history("Python和Java有什么区别？", "technical_agent", True, 4)
        
        # 修改路由器的默认策略为基于历史交互
        original_strategy = self.router.default_strategy_name
        self.router.default_strategy_name = "history"
        
        # 执行路由
        query = "如何学习Java编程？"
        result = self.router.execute_route(query)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "history")
        
        # 恢复原始默认策略
        self.router.default_strategy_name = original_strategy
    
    def test_semantic_strategy_integration(self):
        """测试语义路由策略与其他组件的集成"""
        # 设置协调器的模拟行为
        self.coordinator.is_agent_available.return_value = True
        self.coordinator.assign_task.return_value = True
        
        # 将协调器注入到路由器
        self.router.coordinator = self.coordinator
        
        # 确保语义策略已初始化
        if "semantic" not in self.router.strategy_instances:
            self.router.strategy_instances["semantic"] = StrategyFactory.create_strategy("semantic")
        
        # 修改路由器的默认策略为语义路由
        original_strategy = self.router.default_strategy_name
        self.router.default_strategy_name = "semantic"
        
        # 执行路由
        query = "如何使用Python处理JSON数据？"
        result = self.router.execute_route(query)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "semantic")
        
        # 恢复原始默认策略
        self.router.default_strategy_name = original_strategy
    
    def test_fallback_mechanism(self):
        """测试当所有Agent都不可用时的回退机制"""
        # 设置协调器的模拟行为，所有Agent都不可用
        self.coordinator.is_agent_available.return_value = False
        self.coordinator.get_available_agents.return_value = []
        
        # 将协调器注入到路由器
        self.router.coordinator = self.coordinator
        
        # 执行路由
        query = "如何使用Python处理JSON数据？"
        result = self.router.execute_route(query)
        
        # 验证结果，应该使用回退Agent
        self.assertIsNotNone(result)
        self.assertEqual(result["agent_id"], self.router.fallback_agent)

if __name__ == "__main__":
    unittest.main()