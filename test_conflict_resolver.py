import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.conflict_resolver import ConflictResolver

class TestConflictResolver(unittest.TestCase):
    """测试冲突解决器"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.resolver = ConflictResolver()
        
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
    
    def test_extract_key_information(self):
        """测试关键信息提取"""
        for response in self.test_responses:
            info = self.resolver._extract_key_information(response)
            
            # 验证提取的信息
            self.assertIn("facts", info)
            self.assertIn("opinions", info)
            self.assertIn("recommendations", info)
            self.assertEqual(info["agent_id"], response["agent_id"])
            self.assertEqual(info["confidence"], response["confidence"])
            
            # 验证提取的事实、观点和推荐不为空
            if response["agent_id"] == "technical_agent":
                self.assertTrue(len(info["facts"]) > 0)
            elif response["agent_id"] == "memory_agent":
                self.assertTrue(len(info["recommendations"]) > 0)
    
    def test_detect_conflicts(self):
        """测试冲突检测"""
        conflicts = self.resolver.detect_conflicts(self.test_responses)
        
        # 验证检测到的冲突
        self.assertTrue(len(conflicts) > 0, "应该检测到至少一个冲突")
        
        # 验证冲突类型
        conflict_types = set(conflict["type"] for conflict in conflicts)
        self.assertTrue("factual" in conflict_types, "应该检测到事实性冲突")
        
        # 验证冲突内容
        for conflict in conflicts:
            if conflict["type"] == "factual":
                statements = conflict["statements"]
                # 检查是否包含关于Python是解释型还是编译型的冲突
                python_type_conflict = any("解释型" in stmt and "Python" in stmt for stmt in statements) and \
                                     any("编译型" in stmt and "Python" in stmt for stmt in statements)
                if python_type_conflict:
                    self.assertTrue(True, "检测到关于Python类型的冲突")
                    break
        else:
            self.fail("未检测到预期的Python类型冲突")
    
    def test_resolve_conflicts(self):
        """测试冲突解决"""
        conflicts = self.resolver.detect_conflicts(self.test_responses)
        resolved_response = self.resolver.resolve_conflicts(conflicts, self.test_responses)
        
        # 验证解决后的响应
        self.assertIn("content", resolved_response)
        self.assertIn("confidence", resolved_response)
        self.assertIn("sources", resolved_response)
        self.assertIn("has_conflicts", resolved_response)
        self.assertIn("conflict_count", resolved_response)
        
        # 验证解决后的内容包含冲突信息
        self.assertTrue("Python" in resolved_response["content"])
        self.assertTrue(resolved_response["has_conflicts"])
        self.assertTrue(resolved_response["conflict_count"] > 0)
    
    def test_majority_voting(self):
        """测试多数投票策略"""
        statements = [
            "Python是解释型语言",
            "Python是解释型编程语言",
            "Python是编译型语言"
        ]
        confidences = [0.8, 0.7, 0.6]
        
        result = self.resolver.apply_majority_voting(statements, confidences)
        
        # 验证结果应该选择第一个或第二个陈述（解释型）
        self.assertTrue("解释型" in result)
        self.assertFalse("编译型" in result)
    
    def test_confidence_weighted(self):
        """测试置信度加权策略"""
        statements = [
            "Python是解释型语言",
            "Python是编译型语言"
        ]
        confidences = [0.6, 0.9]
        
        result = self.resolver.apply_confidence_weighted(statements, confidences)
        
        # 验证结果应该选择第二个陈述（置信度更高）
        self.assertEqual(result, "Python是编译型语言")
    
    def test_semantic_analysis(self):
        """测试语义分析策略"""
        statements = [
            "Python是一种高级编程语言，特点是简单易学",
            "Python是高级语言，其特点是容易学习和使用"
        ]
        
        result = self.resolver.apply_semantic_analysis(statements)
        
        # 验证结果应该包含共同部分
        self.assertTrue("Python" in result)
        self.assertTrue("高级" in result)
    
    def test_source_reliability(self):
        """测试源可靠性策略"""
        statements = [
            "Python是解释型语言",
            "Python是编译型语言"
        ]
        agents = ["technical_agent", "creative_agent"]
        
        result = self.resolver.apply_source_reliability(statements, agents, self.test_responses)
        
        # 验证结果应该选择technical_agent的陈述（更可靠）
        self.assertEqual(result, "Python是解释型语言")

if __name__ == "__main__":
    unittest.main()