import json
import os
from typing import Dict, List, Any, Tuple, Optional
from core.knowledge_graph import KnowledgeGraph

class FactVerifier:
    """基于知识图谱的事实验证器"""
    
    def __init__(self, config_path: str = None):
        """初始化事实验证器
        
        Args:
            config_path: 配置文件路径
        """
        self.knowledge_graph = KnowledgeGraph(config_path)
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载事实验证器配置失败: {e}")
        
        # 默认配置
        return {
            "verification_threshold": 0.6,
            "external_api_enabled": False,
            "external_api_url": "",
            "external_api_key": "",
            "learning_enabled": True,
            "max_related_facts": 5
        }
    
    def verify_statement(self, statement: str) -> Dict[str, Any]:
        """验证事实陈述
        
        Args:
            statement: 事实陈述
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        # 首先使用知识图谱验证
        verified, confidence, evidence = self.knowledge_graph.verify_fact(statement)
        
        # 如果知识图谱验证失败且启用了外部API，尝试使用外部API验证
        if not verified and self.config["external_api_enabled"]:
            verified, confidence, evidence = self._verify_with_external_api(statement)
        
        # 构建验证结果
        result = {
            "statement": statement,
            "verified": verified,
            "confidence": confidence,
            "evidence": evidence,
            "related_facts": []
        }
        
        # 如果验证通过，获取相关事实
        if verified:
            # 从陈述中提取主体
            subject, _, _ = self.knowledge_graph._parse_statement(statement)
            if subject:
                related_facts = self.knowledge_graph.get_related_facts(
                    subject, 
                    max_distance=2
                )
                # 限制相关事实数量
                result["related_facts"] = related_facts[:self.config["max_related_facts"]]
        
        return result
    
    def _verify_with_external_api(self, statement: str) -> Tuple[bool, float, Optional[str]]:
        """使用外部API验证事实
        
        Args:
            statement: 事实陈述
            
        Returns:
            Tuple[bool, float, Optional[str]]: (是否验证通过, 置信度, 证据)
        """
        # 这里是外部API调用的实现
        # 在实际应用中，可以集成各种知识库API或搜索引擎API
        
        try:
            import requests
            
            if not self.config["external_api_url"]:
                return False, 0.0, None
                
            headers = {}
            if self.config.get("external_api_key"):
                headers["Authorization"] = f"Bearer {self.config['external_api_key']}"
            
            payload = {
                "statement": statement,
                "detailed": True
            }
            
            response = requests.post(
                self.config["external_api_url"],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return (
                    result.get("verified", False),
                    result.get("confidence", 0.0),
                    result.get("evidence", "外部API验证")
                )
            else:
                print(f"API请求失败: {response.status_code} {response.text}")
                return False, 0.0, None
                
        except Exception as e:
            print(f"外部API验证异常: {e}")
            return False, 0.0, None
    
    def learn_from_statement(self, statement: str, confidence: float = None, source: str = "agent") -> bool:
        """从陈述中学习新知识
        
        Args:
            statement: 事实陈述
            confidence: 置信度
            source: 知识来源
            
        Returns:
            bool: 是否成功学习
        """
        if not self.config["learning_enabled"]:
            return False
        
        # 解析陈述
        subject, relation, object = self.knowledge_graph._parse_statement(statement)
        
        if not subject or not relation or not object:
            return False
        
        # 添加到知识图谱
        self.knowledge_graph.add_fact(subject, relation, object, confidence, source)
        return True
    
    def verify_conflict(self, statements: List[str]) -> Dict[str, Any]:
        """验证冲突的多个陈述
        
        Args:
            statements: 冲突的事实陈述列表
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if not statements or len(statements) < 2:
            return {
                "verified": False,
                "resolution": "需要至少两个陈述进行冲突验证",
                "confidence": 0.0,
                "evidence": None
            }
        
        # 验证每个陈述
        verification_results = [self.verify_statement(stmt) for stmt in statements]
        
        # 找出验证通过的陈述
        verified_statements = [result for result in verification_results if result["verified"]]
        
        if not verified_statements:
            # 没有陈述能被验证，尝试分析陈述之间的关系
            # 提取主体和关系
            parsed_statements = []
            for stmt in statements:
                parsed = self.knowledge_graph._parse_statement(stmt)
                if all(parsed):
                    parsed_statements.append(parsed)
            
            # 检查是否可能是等价陈述（例如，不同单位或表述方式）
            if len(parsed_statements) >= 2:
                subjects = [p[0] for p in parsed_statements]
                relations = [p[1] for p in parsed_statements]
                objects = [p[2] for p in parsed_statements]
                
                # 检查主体是否相同
                if len(set(subjects)) == 1:
                    # 主体相同，检查关系是否相似
                    if len(set(relations)) <= 2:  # 允许有限的关系变化
                        # 尝试查找对象之间的关系
                        for i in range(len(objects)):
                            for j in range(i+1, len(objects)):
                                # 检查两个对象是否有关联
                                obj1_facts = self.knowledge_graph.get_related_facts(objects[i])
                                for fact in obj1_facts:
                                    if fact["object"] == objects[j] or fact["subject"] == objects[j]:
                                        # 找到关联，可能是等价陈述
                                        return {
                                            "verified": True,
                                            "resolution": f"两个陈述可能是等价的: {statements[0]} 和 {statements[1]}",
                                            "confidence": 0.7,
                                            "evidence": f"发现对象关联: {objects[i]} 和 {objects[j]}"
                                        }
            
            # 无法验证任何陈述
            return {
                "verified": False,
                "resolution": "无法验证任何陈述",
                "confidence": 0.0,
                "evidence": "知识图谱中没有相关信息"
            }
        
        if len(verified_statements) == 1:
            # 只有一个陈述被验证
            verified = verified_statements[0]
            return {
                "verified": True,
                "resolution": verified["statement"],
                "confidence": verified["confidence"],
                "evidence": verified["evidence"]
            }
        
        # 多个陈述被验证，检查是否可能是等价陈述
        # 提取主体和关系
        parsed_verified = []
        for result in verified_statements:
            parsed = self.knowledge_graph._parse_statement(result["statement"])
            if all(parsed):
                parsed_verified.append((parsed, result))
        
        # 检查是否是等价陈述
        if len(parsed_verified) >= 2:
            subjects = [p[0][0] for p in parsed_verified]
            relations = [p[0][1] for p in parsed_verified]
            objects = [p[0][2] for p in parsed_verified]
            
            # 检查主体是否相同
            if len(set(subjects)) == 1:
                # 主体相同，检查关系
                if len(set(relations)) <= 2:  # 允许有限的关系变化
                    # 检查对象之间是否有关联
                    obj_related = False
                    for i in range(len(objects)):
                        for j in range(i+1, len(objects)):
                            obj1_facts = self.knowledge_graph.get_related_facts(objects[i])
                            for fact in obj1_facts:
                                if fact["object"] == objects[j] or fact["subject"] == objects[j]:
                                    obj_related = True
                                    break
                    
                    if obj_related:
                        # 对象之间有关联，可能是等价陈述
                        return {
                            "verified": True,
                            "resolution": f"两个陈述实际上是一致的，表达方式不同",
                            "confidence": 0.9,
                            "evidence": f"主体相同，对象之间存在关联"
                        }
        
        # 选择置信度最高的陈述
        best_verified = max(verified_statements, key=lambda x: x["confidence"])
        
        return {
            "verified": True,
            "resolution": best_verified["statement"],
            "confidence": best_verified["confidence"],
            "evidence": best_verified["evidence"]
        }
    
    def save_knowledge(self, file_path: str):
        """保存知识图谱
        
        Args:
            file_path: 文件路径
        """
        self.knowledge_graph.save_to_file(file_path)
    
    def load_knowledge(self, file_path: str):
        """加载知识图谱
        
        Args:
            file_path: 文件路径
        """
        self.knowledge_graph.load_from_file(file_path)