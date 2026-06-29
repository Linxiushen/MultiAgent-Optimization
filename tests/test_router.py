import unittest
import json
import os
from core.router import Router
from unittest.mock import patch, MagicMock

class TestRouter(unittest.TestCase):
    def setUp(self):
        """测试前设置"""
        # 创建临时配置文件
        self.config_path = "tests/temp_router_config.json"
        self.test_config = {
            "agent_profiles": {
                "test_agent1": {
                    "description": "测试Agent 1，擅长处理测试问题",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                },
                "test_agent2": {
                    "description": "测试Agent 2，擅长处理技术问题",
                    "model": "gpt-4",
                    "temperature": 0.5
                }
            },
            "routing_strategies": {
                "test_agent1": "direct",
                "test_agent2": "memory_enhanced"
            },
            "default_strategy": "direct"
        }

        # 写入临时配置文件
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f, ensure_ascii=False, indent=2)

        # 创建Router实例
        self.router = Router(config_path=self.config_path)

    def tearDown(self):
        """测试后清理"""
        # 删除临时配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    def test_load_config(self):
        """测试加载配置文件"""
        # 测试成功加载
        self.assertEqual(self.router.agent_profiles, self.test_config["agent_profiles"])
        self.assertEqual(self.router.routing_strategies, self.test_config["routing_strategies"])
        self.assertEqual(self.router.default_strategy, self.test_config["default_strategy"])

        # 测试加载不存在的配置文件
        router_without_config = Router(config_path="non_existent_config.json")
        self.assertEqual(router_without_config.agent_profiles, {})
        self.assertEqual(router_without_config.routing_strategies, {})
        self.assertEqual(router_without_config.default_strategy, "direct")

    @patch('core.router.lite_llm_infer')
    def test_determine_agent(self, mock_lite_llm_infer):
        """测试确定Agent功能"""
        # 模拟LLM响应
        mock_lite_llm_infer.return_value = "test_agent1"

        # 测试正常情况
        query = "这是一个测试问题"
        agent_id = self.router.determine_agent(query)
        self.assertEqual(agent_id, "test_agent1")

        # 测试LLM返回无效Agent ID的情况
        mock_lite_llm_infer.return_value = "invalid_agent"
        agent_id = self.router.determine_agent(query)
        self.assertEqual(agent_id, "default_agent")

        # 测试没有Agent配置的情况
        router_without_agents = Router(config_path=self.config_path)
        router_without_agents.agent_profiles = {}
        agent_id = router_without_agents.determine_agent(query)
        self.assertEqual(agent_id, "default_agent")

    @patch('core.router.Router.determine_agent')
    def test_route_query(self, mock_determine_agent):
        """测试路由查询功能"""
        # 模拟determine_agent方法
        mock_determine_agent.return_value = "test_agent1"

        # 测试路由查询
        query = "这是一个测试问题"
        route_result = self.router.route_query(query)

        self.assertEqual(route_result["agent_id"], "test_agent1")
        self.assertEqual(route_result["strategy"], "direct")
        self.assertEqual(route_result["query"], query)
        self.assertIn("timestamp", route_result)

    @patch('core.router.call_model')
    @patch('core.router.Router.determine_agent')
    def test_execute_route_direct(self, mock_determine_agent, mock_call_model):
        """测试直接执行路由功能"""
        # 模拟determine_agent方法
        mock_determine_agent.return_value = "test_agent1"

        # 模拟call_model方法
        mock_response = {"choices": [{"message": {"content": "这是测试响应"}}]}
        mock_call_model.return_value = mock_response

        # 测试执行路由
        query = "这是一个测试问题"
        route_result = self.router.route_query(query)
        execute_result = self.router.execute_route(route_result)

        self.assertEqual(execute_result["agent_id"], "test_agent1")
        self.assertEqual(execute_result["response"], mock_response)
        self.assertEqual(execute_result["model_used"], "gpt-3.5-turbo")
        self.assertTrue(execute_result["success"])

    @patch('core.router.Router.determine_agent')
    @patch('core.router.MemoryManager')
    def test_execute_route_memory_enhanced(self, mock_memory_manager, mock_determine_agent):
        """测试记忆增强执行路由功能"""
        # 模拟determine_agent方法
        mock_determine_agent.return_value = "test_agent2"

        # 模拟MemoryManager
        mock_memory_instance = MagicMock()
        mock_memory_instance.retrieve_memory.return_value = []
        mock_memory_manager.return_value = mock_memory_instance

        # 模拟call_model方法
        with patch('core.router.call_model') as mock_call_model:
            mock_response = {"choices": [{"message": {"content": "这是记忆增强响应"}}]}
            mock_call_model.return_value = mock_response

            # 测试执行路由
            query = "这是一个技术问题"
            route_result = self.router.route_query(query)
            execute_result = self.router.execute_route(route_result)

            self.assertEqual(execute_result["agent_id"], "test_agent2")
            self.assertEqual(execute_result["response"], mock_response)
            self.assertTrue(execute_result["success"])

if __name__ == '__main__':
    unittest.main()