import networkx as nx
import json
import os
from typing import Dict, List, Any, Tuple, Optional
import re

class KnowledgeGraph:
    """知识图谱类，用于存储和查询事实性知识"""
    
    def __init__(self, config_path: str = None):
        """初始化知识图谱
        
        Args:
            config_path: 配置文件路径
        """
        self.graph = nx.DiGraph()
        self.config = self._load_config(config_path)
        self.load_initial_knowledge()
    
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
                print(f"加载知识图谱配置失败: {e}")
        
        # 默认配置
        return {
            "confidence_threshold": 0.7,
            "relation_types": ["is_a", "has_property", "part_of", "related_to"],
            "knowledge_sources": ["system", "user", "agent", "external"],
            "default_confidence": 0.8
        }
    
    def load_initial_knowledge(self, knowledge_file=None):
        """从文件加载初始知识
        
        Args:
            knowledge_file: 知识文件路径，如果为None则不加载
        """
        # 从配置文件加载基础知识
        if 'basic_facts' in self.config:
            for fact in self.config['basic_facts']:
                if len(fact) >= 5:
                    subject, relation, object_value, confidence, source = fact
                    self.add_fact(subject, relation, object_value, confidence, source)
                elif len(fact) >= 3:
                     subject, relation, object_value = fact
                     self.add_fact(subject, relation, object_value)
                 else:
                     self.logger.warning(f"基础知识格式不正确: {fact}")
             self.logger.info(f"从配置加载了 {len(self.config.get('basic_facts', []))} 条基础知识")
                 
         # 从文件加载知识
         if knowledge_file and os.path.exists(knowledge_file):
             self.load_from_file(knowledge_file)
             self.logger.info(f"从 {knowledge_file} 加载了初始知识")
         else:
             # 如果没有配置文件和知识文件，加载默认示例知识
             if 'basic_facts' not in self.config:
                 self.add_fact("人工智能", "is_a", "技术领域", confidence=0.99, source="system")
                 self.add_fact("机器学习", "is_a", "人工智能子领域", confidence=0.99, source="system")
                 self.add_fact("深度学习", "is_a", "机器学习技术", confidence=0.98, source="system")
                 self.add_fact("神经网络", "is_a", "深度学习模型", confidence=0.97, source="system")
                 self.add_fact("Python", "is_a", "编程语言", confidence=0.99, source="system")
                 self.add_fact("TensorFlow", "is_a", "深度学习框架", confidence=0.95, source="system")
                 self.add_fact("PyTorch", "is_a", "深度学习框架", confidence=0.95, source="system")
                 self.logger.info("加载了默认示例知识")
    
    def add_fact(self, subject: str, relation: str, object: str, confidence: float = None, source: str = "agent"):
        """添加事实到知识图谱
        
        Args:
            subject: 主体
            relation: 关系
            object: 客体
            confidence: 置信度
            source: 知识来源
        """
        if confidence is None:
            confidence = self.config["default_confidence"]
        
        # 添加节点
        if not self.graph.has_node(subject):
            self.graph.add_node(subject, type="entity")
        
        if not self.graph.has_node(object):
            self.graph.add_node(object, type="entity")
        
        # 添加边（关系）
        self.graph.add_edge(subject, object, relation=relation, confidence=confidence, source=source)
    
    def verify_fact(self, statement: str) -> Tuple[bool, float, Optional[str]]:
        """验证事实陈述
        
        Args:
            statement: 事实陈述
            
        Returns:
            Tuple[bool, float, Optional[str]]: (是否验证通过, 置信度, 证据)
        """
        # 解析陈述，提取主体、关系和客体
        subject, relation, object = self._parse_statement(statement)
        
        if not subject or not relation or not object:
            return False, 0.0, None
        
        # 直接查找
        if self.graph.has_edge(subject, object):
            edge_data = self.graph.get_edge_data(subject, object)
            if edge_data.get("relation") == relation:
                return True, edge_data.get("confidence", 0.5), f"直接关系: {subject} {relation} {object}"
        
        # 间接推理
        paths = list(nx.all_simple_paths(self.graph, subject, object, cutoff=3))
        if paths:
            # 找到最短路径
            shortest_path = min(paths, key=len)
            path_confidence = self._calculate_path_confidence(shortest_path)
            path_desc = " -> ".join(shortest_path)
            return True, path_confidence, f"推理路径: {path_desc}"
        
        return False, 0.0, None
    
    def _parse_statement(self, statement: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """解析事实陈述
        
        Args:
            statement: 事实陈述
            
        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]: (主体, 关系, 客体)
        """
        # 简单实现，实际应用中可能需要更复杂的NLP技术
        patterns = [
            r"(.*?)是(.*?)",  # X是Y
            r"(.*?)属于(.*?)",  # X属于Y
            r"(.*?)包含(.*?)",  # X包含Y
            r"(.*?)由(.*?)组成",  # X由Y组成
            r"(.*?)的一部分是(.*?)"  # X的一部分是Y
        ]
        
        for pattern in patterns:
            match = re.search(pattern, statement)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    subject = groups[0].strip()
                    object = groups[1].strip()
                    
                    # 根据模式确定关系类型
                    if "是" in pattern:
                        relation = "is_a"
                    elif "属于" in pattern:
                        relation = "is_a"
                    elif "包含" in pattern:
                        relation = "has_property"
                    elif "组成" in pattern:
                        relation = "part_of"
                    elif "一部分" in pattern:
                        relation = "part_of"
                    else:
                        relation = "related_to"
                    
                    return subject, relation, object
        
        return None, None, None
    
    def _calculate_path_confidence(self, path: List[str]) -> float:
        """计算路径的置信度
        
        Args:
            path: 节点路径
            
        Returns:
            float: 路径置信度
        """
        if len(path) <= 1:
            return 0.0
        
        # 计算路径上所有边的置信度乘积
        confidence = 1.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                edge_confidence = edge_data.get("confidence", 0.5)
                confidence *= edge_confidence
        
        return confidence
    
    def get_related_facts(self, entity: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """获取与实体相关的事实
        
        Args:
            entity: 实体名称
            max_distance: 最大关系距离
            
        Returns:
            List[Dict[str, Any]]: 相关事实列表
        """
        if not self.graph.has_node(entity):
            return []
        
        related_facts = []
        
        # 获取出边（实体作为主体）
        for neighbor in self.graph.successors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            related_facts.append({
                "subject": entity,
                "relation": edge_data.get("relation", "related_to"),
                "object": neighbor,
                "confidence": edge_data.get("confidence", 0.5),
                "source": edge_data.get("source", "unknown")
            })
        
        # 获取入边（实体作为客体）
        for neighbor in self.graph.predecessors(entity):
            edge_data = self.graph.get_edge_data(neighbor, entity)
            related_facts.append({
                "subject": neighbor,
                "relation": edge_data.get("relation", "related_to"),
                "object": entity,
                "confidence": edge_data.get("confidence", 0.5),
                "source": edge_data.get("source", "unknown")
            })
        
        # 如果需要更远的关系，可以使用BFS或DFS进一步扩展
        
        return related_facts
    
    def save_to_file(self, file_path: str):
        """保存知识图谱到文件
        
        Args:
            file_path: 文件路径
        """
        data = nx.node_link_data(self.graph)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"知识图谱已保存到 {file_path}")
        except Exception as e:
            print(f"保存知识图谱失败: {e}")
    
    def load_from_file(self, file_path: str):
        """从文件加载知识图谱
        
        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
            print(f"已从 {file_path} 加载知识图谱")
        except Exception as e:
            print(f"加载知识图谱失败: {e}")