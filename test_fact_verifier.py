import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fact_verifier import FactVerifier

class TestFactVerifier(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.fact_verifier = FactVerifier()
        
        # 准备测试数据
        self.test_statements = [
            "地球是太阳系中的第三颗行星",
            "水的化学式是H2O",
            "人类的心脏有四个腔室"
        ]
        
        # 添加一些初始知识
        for statement in self.test_statements:
            self.fact_verifier.learn_from_statement(statement, 0.9, "test_agent")
    
    def test_verify_fact(self):
        """测试事实验证功能"""
        # 测试已知事实
        result = self.fact_verifier.verify_fact("地球是太阳系中的第三颗行星")
        self.assertTrue(result["verified"])
        self.assertGreaterEqual(result["confidence"], 0.7)
        
        # 测试相关事实
        result = self.fact_verifier.verify_fact("地球围绕太阳运行")
        self.assertFalse(result["verified"])  # 应该无法直接验证
        
        # 添加相关知识后再验证
        self.fact_verifier.learn_from_statement("地球围绕太阳运行", 0.9, "test_agent")
        self.fact_verifier.learn_from_statement("行星围绕恒星运行", 0.9, "test_agent")
        self.fact_verifier.learn_from_statement("太阳是一颗恒星", 0.9, "test_agent")
        
        result = self.fact_verifier.verify_fact("地球围绕太阳运行")
        self.assertTrue(result["verified"])
    
    def test_verify_conflict(self):
        """测试冲突验证功能"""
        # 准备冲突陈述
        statements = [
            "水的沸点是100摄氏度",
            "水的沸点是212华氏度"
        ]
        
        # 添加相关知识
        self.fact_verifier.learn_from_statement("100摄氏度等于212华氏度", 0.9, "test_agent")
        
        # 验证冲突
        result = self.fact_verifier.verify_conflict(statements)
        self.assertTrue(result["verified"])
        self.assertIn("两个陈述实际上是一致的", result["resolution"])
        
        # 测试真正的冲突
        statements = [
            "地球是太阳系中的第三颗行星",
            "地球是太阳系中的第四颗行星"
        ]
        
        result = self.fact_verifier.verify_conflict(statements)
        self.assertTrue(result["verified"])
        self.assertIn("第三颗", result["resolution"])
    
    def test_learn_from_statement(self):
        """测试从陈述中学习新知识"""
        # 学习新知识
        self.fact_verifier.learn_from_statement("猫是哺乳动物", 0.9, "test_agent")
        
        # 验证学习效果
        result = self.fact_verifier.verify_fact("猫是哺乳动物")
        self.assertTrue(result["verified"])
        
        # 测试知识推理
        self.fact_verifier.learn_from_statement("哺乳动物是脊椎动物", 0.9, "test_agent")
        result = self.fact_verifier.get_related_facts("猫")
        
        # 应该能找到直接和间接相关的事实
        found_direct = False
        found_indirect = False
        
        for fact in result["facts"]:
            if "猫是哺乳动物" in fact["statement"]:
                found_direct = True
            if "哺乳动物是脊椎动物" in fact["statement"]:
                found_indirect = True
        
        self.assertTrue(found_direct)
        self.assertTrue(found_indirect)
    
    @patch('requests.post')
    def test_external_verification(self, mock_post):
        """测试外部API验证功能"""
        # 模拟外部API响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "verified": True,
            "confidence": 0.95,
            "evidence": "来自权威科学数据库的验证"
        }
        mock_post.return_value = mock_response
        
        # 启用外部API
        self.fact_verifier.config["external_api_enabled"] = True
        self.fact_verifier.config["external_api_url"] = "https://api.factcheck.example.com/verify"
        
        # 验证未知事实
        result = self.fact_verifier.verify_fact("量子计算机使用量子比特而不是传统比特")
        
        # 验证结果
        self.assertTrue(result["verified"])
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["evidence"], "来自权威科学数据库的验证")
        
        # 验证API调用
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main()