import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.enhanced_router import EnhancedRouter
from core.routing_strategies import (
    HistoryBasedStrategy, 
    SemanticRoutingStrategy,
    AdaptiveRoutingStrategy,
    MultiAgentRoutingStrategy
)

class TestEnhancedRouter(unittest.TestCase):
    def setUp(self):
        # 创建临时配置文件
        self.temp_config_file = "temp_router_config.json"
        self.config = {
            "agent_profiles": {
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
            },
            "routing_strategies": {
                "direct": {
                    "description": "直接路由策略",
                    "enabled": True
                },
                "memory_enhanced": {
                    "description": "记忆增强路由策略",
                    "enabled": True
                },
                "semantic": {
                    "description": "基于语义理解的路由策略",
                    "enabled": True,
                    "config": {
                        "intent_patterns": {
                            "technical": ["代码", "编程", "开发", "调试"],
                            "creative": ["创意", "设计", "写作", "文案"],
                            "general": ["什么是", "如何", "为什么"]
                        }
                    }
                }
            },
            "agent_strategy_mapping": {
                "default_agent": "direct",
                "technical_agent": "direct",
                "creative_agent": "direct"
            },
            "default_strategy": "semantic",
            "routing_threshold": 0.75,
            "fallback_agent": "default_agent"
        }
        
        # 写入临时配置文件
        with open(self.temp_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        # 测试用的上下文
        self.context = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
        ]
    
    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.temp_config_file):
            os.remove(self.temp_config_file)
    
    def test_initialization(self):
        """测试路由器初始化"""
        router = EnhancedRouter(self.temp_config_file)
        
        # 验证配置加载
        self.assertEqual(router.default_strategy_name, "semantic")
        self.assertEqual(router.fallback_agent, "default_agent")
        self.assertEqual(router.routing_threshold, 0.75)
        
        # 验证策略实例初始化
        self.assertIn("direct", router.strategy_instances)
        self.assertIn("memory_enhanced", router.strategy_instances)
        self.assertIn("semantic", router.strategy_instances)
    
    @patch('core.enhanced_router.StrategyFactory.create_strategy')
    def test_strategy_initialization(self, mock_create_strategy):
        """测试策略初始化"""
        # 模拟策略创建
        mock_create_strategy.side_effect = lambda strategy_type, config: MagicMock()
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 验证策略工厂调用
        self.assertEqual(mock_create_strategy.call_count, 3)  # direct, memory_enhanced, semantic
    
    @patch('core.enhanced_router.lite_llm_infer')
    def test_llm_based_routing(self, mock_lite_llm_infer):
        """测试基于LLM的路由决策"""
        # 模拟LLM响应
        mock_lite_llm_infer.return_value = "technical_agent"
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 测试LLM路由
        agent_id = router._llm_based_routing("如何编写Python代码？")
        self.assertEqual(agent_id, "technical_agent")
        
        # 测试无效响应
        mock_lite_llm_infer.return_value = "invalid_agent"
        agent_id = router._llm_based_routing("如何编写Python代码？")
        self.assertEqual(agent_id, "default_agent")
    
    @patch('core.enhanced_router.EnhancedRouter._llm_based_routing')
    def test_determine_agent_fallback(self, mock_llm_routing):
        """测试Agent确定的回退机制"""
        # 模拟LLM路由
        mock_llm_routing.return_value = "technical_agent"
        
        # 创建路由器，但不使用默认策略
        config = self.config.copy()
        config["default_strategy"] = "unknown_strategy"
        with open(self.temp_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 测试确定Agent（应回退到LLM路由）
        agent_id = router.determine_agent("如何编写Python代码？")
        self.assertEqual(agent_id, "technical_agent")
        mock_llm_routing.assert_called_once()
    
    @patch('core.routing_strategies.SemanticRoutingStrategy.select_agent')
    def test_determine_agent_with_strategy(self, mock_select_agent):
        """测试使用策略确定Agent"""
        # 模拟策略选择
        mock_select_agent.return_value = "technical_agent"
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 测试确定Agent
        agent_id = router.determine_agent("如何编写Python代码？")
        self.assertEqual(agent_id, "technical_agent")
        mock_select_agent.assert_called_once()
    
    def test_route_query(self):
        """测试查询路由"""
        router = EnhancedRouter(self.temp_config_file)
        
        # 模拟determine_agent方法
        router.determine_agent = MagicMock(return_value="technical_agent")
        
        # 测试路由查询
        route_info = router.route_query("如何编写Python代码？")
        self.assertEqual(route_info["agent_id"], "technical_agent")
        self.assertEqual(route_info["strategy"], "direct")
        self.assertEqual(route_info["query"], "如何编写Python代码？")
    
    @patch('core.enhanced_router.call_model')
    def test_execute_direct_strategy(self, mock_call_model):
        """测试执行直接路由策略"""
        # 模拟模型调用
        mock_call_model.return_value = "这是模型的响应"
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 测试执行直接策略
        result = router._execute_direct_strategy(
            "如何编写Python代码？", 
            "technical_agent", 
            "gpt-4", 
            0.5
        )
        
        self.assertEqual(result["agent_id"], "technical_agent")
        self.assertEqual(result["response"], "这是模型的响应")
        self.assertEqual(result["strategy"], "direct")
        self.assertTrue(result["success"])
    
    @patch('core.enhanced_router.call_model')
    def test_execute_memory_enhanced_strategy(self, mock_call_model):
        """测试执行记忆增强路由策略"""
        # 模拟模型调用
        mock_call_model.return_value = "这是模型的响应"
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 测试执行记忆增强策略
        result = router._execute_memory_enhanced_strategy(
            "如何编写Python代码？", 
            self.context,
            "technical_agent", 
            "gpt-4", 
            0.5
        )
        
        self.assertEqual(result["agent_id"], "technical_agent")
        self.assertEqual(result["response"], "这是模型的响应")
        self.assertEqual(result["strategy"], "memory_enhanced")
        self.assertTrue(result["success"])
    
    @patch('core.enhanced_router.call_model')
    def test_execute_advanced_strategy(self, mock_call_model):
        """测试执行高级路由策略"""
        # 模拟模型调用
        mock_call_model.return_value = "这是模型的响应"
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 创建模拟策略
        mock_strategy = MagicMock()
        mock_strategy.update_history = MagicMock()
        router.strategy_instances["semantic"] = mock_strategy
        
        # 测试执行高级策略
        result = router._execute_advanced_strategy(
            "如何编写Python代码？", 
            self.context,
            "technical_agent", 
            "semantic",
            "gpt-4", 
            0.5
        )
        
        self.assertEqual(result["agent_id"], "technical_agent")
        self.assertEqual(result["response"], "这是模型的响应")
        self.assertEqual(result["strategy"], "semantic")
        self.assertTrue(result["success"])
        
        # 验证策略更新调用
        mock_strategy.update_history.assert_called_once()
    
    @patch('core.enhanced_router.EnhancedRouter._execute_direct_strategy')
    def test_execute_route_direct(self, mock_execute_direct):
        """测试执行路由（直接策略）"""
        # 模拟直接策略执行
        mock_execute_direct.return_value = {
            "agent_id": "technical_agent",
            "response": "这是模型的响应",
            "strategy": "direct",
            "success": True
        }
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 模拟route_query方法
        router.route_query = MagicMock(return_value={
            "agent_id": "technical_agent",
            "strategy": "direct",
            "query": "如何编写Python代码？"
        })
        
        # 测试执行路由
        result = router.execute_route("如何编写Python代码？")
        
        self.assertEqual(result["agent_id"], "technical_agent")
        self.assertEqual(result["response"], "这是模型的响应")
        self.assertEqual(result["strategy"], "direct")
        self.assertTrue(result["success"])
    
    @patch('core.enhanced_router.EnhancedRouter._execute_memory_enhanced_strategy')
    def test_execute_route_memory_enhanced(self, mock_execute_memory):
        """测试执行路由（记忆增强策略）"""
        # 模拟记忆增强策略执行
        mock_execute_memory.return_value = {
            "agent_id": "memory_agent",
            "response": "这是模型的响应",
            "strategy": "memory_enhanced",
            "success": True
        }
        
        router = EnhancedRouter(self.temp_config_file)
        
        # 模拟route_query方法
        router.route_query = MagicMock(return_value={
            "agent_id": "memory_agent",
            "strategy": "memory_enhanced",
            "query": "你还记得我之前问的问题吗？"
        })
        
        # 测试执行路由
        result = router.execute_route("你还记得我之前问的问题吗？", self.context)
        
        self.assertEqual(result["agent_id"], "memory_agent")
        self.assertEqual(result["response"], "这是模型的响应")
        self.assertEqual(result["strategy"], "memory_enhanced")
        self.assertTrue(result["success"])
    
    def test_provide_feedback(self):
        """测试提供反馈"""
        router = EnhancedRouter(self.temp_config_file)
        
        # 创建模拟策略
        mock_strategy = MagicMock()
        mock_strategy.update_history = MagicMock()
        router.strategy_instances["semantic"] = mock_strategy
        
        # 创建模拟自适应策略
        mock_adaptive = MagicMock()
        mock_adaptive.feedback = MagicMock()
        router.strategy_instances["adaptive"] = mock_adaptive
        
        # 测试提供反馈
        router.provide_feedback(
            "如何编写Python代码？",
            "technical_agent",
            "semantic",
            True,
            5
        )
        
        # 验证策略更新调用
        mock_strategy.update_history.assert_called_once_with(
            "如何编写Python代码？",
            "technical_agent",
            True,
            5
        )
        
        # 验证自适应策略反馈调用
        mock_adaptive.feedback.assert_called_once_with("semantic", True)

if __name__ == '__main__':
    unittest.main()