import json
import re
from typing import Dict, List, Any, Tuple, Optional
from difflib import SequenceMatcher
from collections import Counter
from core.fact_verifier import FactVerifier

class ConflictResolver:
    """冲突解决器，负责解决多个Agent响应之间的冲突"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化冲突解决器
        
        Args:
            config: 配置参数
        """
        self.config = config or {
            "similarity_threshold": 0.7,
            "contradiction_threshold": 0.3,
            "factual_weight": 0.6,
            "opinion_weight": 0.4,
            "resolution_strategies": ["knowledge_graph", "majority_voting", "confidence_weighted", "semantic_analysis", "source_reliability"],
            "default_strategy": "knowledge_graph"
        }
        
        # 初始化事实验证器
        self.fact_verifier = FactVerifier()
    
    def detect_conflicts(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测响应之间的冲突
        
        Args:
            responses: Agent响应列表
            
        Returns:
            List[Dict[str, Any]]: 检测到的冲突列表
        """
        conflicts = []
        
        # 提取关键信息
        extracted_info = [self._extract_key_information(resp) for resp in responses]
        
        # 检测事实性冲突
        factual_conflicts = self._detect_factual_conflicts(extracted_info, responses)
        conflicts.extend(factual_conflicts)
        
        # 检测观点冲突
        opinion_conflicts = self._detect_opinion_conflicts(extracted_info, responses)
        conflicts.extend(opinion_conflicts)
        
        # 检测推荐冲突
        recommendation_conflicts = self._detect_recommendation_conflicts(extracted_info, responses)
        conflicts.extend(recommendation_conflicts)
        
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解决检测到的冲突
        
        Args:
            conflicts: 检测到的冲突列表
            responses: 原始响应列表
            
        Returns:
            Dict[str, Any]: 解决冲突后的统一响应
        """
        if not conflicts:
            # 如果没有冲突，返回置信度最高的响应
            return max(responses, key=lambda x: x.get("confidence", 0.5))
        
        # 按冲突类型分组
        factual_conflicts = [c for c in conflicts if c["type"] == "factual"]
        opinion_conflicts = [c for c in conflicts if c["type"] == "opinion"]
        recommendation_conflicts = [c for c in conflicts if c["type"] == "recommendation"]
        
        # 解决事实性冲突
        factual_resolutions = {}
        for conflict in factual_conflicts:
            resolution = self._resolve_factual_conflict(conflict, responses)
            factual_resolutions[conflict["key"]] = resolution
        
        # 解决观点冲突
        opinion_resolutions = {}
        for conflict in opinion_conflicts:
            resolution = self._resolve_opinion_conflict(conflict, responses)
            opinion_resolutions[conflict["key"]] = resolution
        
        # 解决推荐冲突
        recommendation_resolutions = {}
        for conflict in recommendation_conflicts:
            resolution = self._resolve_recommendation_conflict(conflict, responses)
            recommendation_resolutions[conflict["key"]] = resolution
        
        # 构建统一响应
        unified_response = self._build_unified_response(
            responses, 
            factual_resolutions, 
            opinion_resolutions, 
            recommendation_resolutions
        )
        
        return unified_response
    
    def _extract_key_information(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """从响应中提取关键信息
        
        Args:
            response: Agent响应
            
        Returns:
            Dict[str, Any]: 提取的关键信息
        """
        content = response.get("content", "")
        
        # 提取事实性陈述
        facts = self._extract_facts(content)
        
        # 提取观点
        opinions = self._extract_opinions(content)
        
        # 提取推荐
        recommendations = self._extract_recommendations(content)
        
        return {
            "facts": facts,
            "opinions": opinions,
            "recommendations": recommendations,
            "agent_id": response.get("agent_id", ""),
            "confidence": response.get("confidence", 0.5)
        }
    
    def _extract_facts(self, content: str) -> List[str]:
        """提取事实性陈述
        
        Args:
            content: 响应内容
            
        Returns:
            List[str]: 提取的事实性陈述列表
        """
        # 简单实现：提取包含事实性关键词的句子
        fact_patterns = [
            r"([^.!?]*(?:是|为|有|包含|等于|大于|小于|包括|由|构成)[^.!?]*[.!?])",
            r"([^.!?]*(?:定义为|表示|指的是|意味着)[^.!?]*[.!?])",
            r"([^.!?]*(?:在|当|如果)[^.!?]*(?:时|情况下)[^.!?]*[.!?])"
        ]
        
        facts = []
        for pattern in fact_patterns:
            matches = re.findall(pattern, content)
            facts.extend(matches)
        
        return [fact.strip() for fact in facts if fact.strip()]
    
    def _extract_opinions(self, content: str) -> List[str]:
        """提取观点
        
        Args:
            content: 响应内容
            
        Returns:
            List[str]: 提取的观点列表
        """
        # 简单实现：提取包含观点性关键词的句子
        opinion_patterns = [
            r"([^.!?]*(?:认为|觉得|相信|建议|推荐|可能|或许|也许|应该)[^.!?]*[.!?])",
            r"([^.!?]*(?:更好|最好|优先|优势|劣势|不足|问题)[^.!?]*[.!?])",
            r"([^.!?]*(?:我的|个人|主观)[^.!?]*(?:看法|观点|意见)[^.!?]*[.!?])"
        ]
        
        opinions = []
        for pattern in opinion_patterns:
            matches = re.findall(pattern, content)
            opinions.extend(matches)
        
        return [opinion.strip() for opinion in opinions if opinion.strip()]
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """提取推荐
        
        Args:
            content: 响应内容
            
        Returns:
            List[str]: 提取的推荐列表
        """
        # 简单实现：提取包含推荐性关键词的句子
        recommendation_patterns = [
            r"([^.!?]*(?:建议|推荐|应该|最好|可以|不妨|试试)[^.!?]*[.!?])",
            r"([^.!?]*(?:步骤|方法|做法|流程|过程)[^.!?]*[.!?])",
            r"([^.!?]*(?:首先|然后|接着|最后|第一|第二)[^.!?]*[.!?])"
        ]
        
        recommendations = []
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, content)
            recommendations.extend(matches)
        
        return [recommendation.strip() for recommendation in recommendations if recommendation.strip()]
    
    def _detect_factual_conflicts(self, extracted_info: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测事实性冲突
        
        Args:
            extracted_info: 提取的关键信息列表
            responses: 原始响应列表
            
        Returns:
            List[Dict[str, Any]]: 检测到的事实性冲突列表
        """
        conflicts = []
        
        # 收集所有事实性陈述
        all_facts = []
        for info in extracted_info:
            all_facts.extend([(fact, info["agent_id"]) for fact in info["facts"]])
        
        # 检测冲突
        for i, (fact1, agent1) in enumerate(all_facts):
            for j, (fact2, agent2) in enumerate(all_facts[i+1:], i+1):
                if agent1 == agent2:
                    continue  # 跳过同一Agent的陈述
                
                # 计算相似度
                similarity = SequenceMatcher(None, fact1, fact2).ratio()
                
                # 如果相似度高但不完全相同，可能存在冲突
                if similarity > self.config["similarity_threshold"] and similarity < 0.95:
                    conflicts.append({
                        "type": "factual",
                        "key": f"fact_{i}_{j}",
                        "statements": [fact1, fact2],
                        "agents": [agent1, agent2],
                        "similarity": similarity
                    })
        
        return conflicts
    
    def _detect_opinion_conflicts(self, extracted_info: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测观点冲突
        
        Args:
            extracted_info: 提取的关键信息列表
            responses: 原始响应列表
            
        Returns:
            List[Dict[str, Any]]: 检测到的观点冲突列表
        """
        conflicts = []
        
        # 收集所有观点
        all_opinions = []
        for info in extracted_info:
            all_opinions.extend([(opinion, info["agent_id"]) for opinion in info["opinions"]])
        
        # 检测冲突
        for i, (opinion1, agent1) in enumerate(all_opinions):
            for j, (opinion2, agent2) in enumerate(all_opinions[i+1:], i+1):
                if agent1 == agent2:
                    continue  # 跳过同一Agent的观点
                
                # 计算相似度
                similarity = SequenceMatcher(None, opinion1, opinion2).ratio()
                
                # 如果相似度适中，可能存在观点冲突
                if similarity > self.config["contradiction_threshold"] and similarity < self.config["similarity_threshold"]:
                    conflicts.append({
                        "type": "opinion",
                        "key": f"opinion_{i}_{j}",
                        "statements": [opinion1, opinion2],
                        "agents": [agent1, agent2],
                        "similarity": similarity
                    })
        
        return conflicts
    
    def _detect_recommendation_conflicts(self, extracted_info: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测推荐冲突
        
        Args:
            extracted_info: 提取的关键信息列表
            responses: 原始响应列表
            
        Returns:
            List[Dict[str, Any]]: 检测到的推荐冲突列表
        """
        conflicts = []
        
        # 收集所有推荐
        all_recommendations = []
        for info in extracted_info:
            all_recommendations.extend([(rec, info["agent_id"]) for rec in info["recommendations"]])
        
        # 检测冲突
        for i, (rec1, agent1) in enumerate(all_recommendations):
            for j, (rec2, agent2) in enumerate(all_recommendations[i+1:], i+1):
                if agent1 == agent2:
                    continue  # 跳过同一Agent的推荐
                
                # 计算相似度
                similarity = SequenceMatcher(None, rec1, rec2).ratio()
                
                # 如果相似度适中，可能存在推荐冲突
                if similarity > self.config["contradiction_threshold"] and similarity < self.config["similarity_threshold"]:
                    conflicts.append({
                        "type": "recommendation",
                        "key": f"recommendation_{i}_{j}",
                        "statements": [rec1, rec2],
                        "agents": [agent1, agent2],
                        "similarity": similarity
                    })
        
        return conflicts
    
    def _resolve_factual_conflict(self, conflict: Dict[str, Any], responses: List[Dict[str, Any]]) -> str:
        """解决事实性冲突
        
        Args:
            conflict: 冲突信息
            responses: 原始响应列表
            
        Returns:
            str: 解决后的事实陈述
        """
        # 获取冲突的陈述和对应的Agent
        statements = conflict["statements"]
        agents = conflict["agents"]
        
        # 选择解决策略
        strategy = self.config.get("default_strategy")
        
        # 使用知识图谱验证
        if strategy == "knowledge_graph":
            verification_result = self.fact_verifier.verify_conflict(statements)
            
            if verification_result["verified"]:
                # 知识图谱验证成功
                return f"{verification_result['resolution']} (置信度: {verification_result['confidence']:.2f}, 证据: {verification_result['evidence']})"
        
        # 如果知识图谱验证失败或不使用知识图谱，回退到置信度策略
        # 获取对应Agent的置信度
        confidences = []
        for agent_id in agents:
            for resp in responses:
                if resp.get("agent_id") == agent_id:
                    confidences.append(resp.get("confidence", 0.5))
                    break
            else:
                confidences.append(0.5)  # 默认置信度
        
        # 使用置信度加权选择
        if confidences[0] > confidences[1] * 1.2:  # 如果第一个Agent的置信度明显更高
            # 尝试从验证成功的陈述中学习
            if strategy == "knowledge_graph":
                self.fact_verifier.learn_from_statement(statements[0], confidences[0], agents[0])
            return statements[0]
        elif confidences[1] > confidences[0] * 1.2:  # 如果第二个Agent的置信度明显更高
            # 尝试从验证成功的陈述中学习
            if strategy == "knowledge_graph":
                self.fact_verifier.learn_from_statement(statements[1], confidences[1], agents[1])
            return statements[1]
        else:  # 置信度相近，合并陈述
            return f"有不同观点：一方面，{statements[0]}另一方面，{statements[1]}"
    
    def _resolve_opinion_conflict(self, conflict: Dict[str, Any], responses: List[Dict[str, Any]]) -> str:
        """解决观点冲突
        
        Args:
            conflict: 冲突信息
            responses: 原始响应列表
            
        Returns:
            str: 解决后的观点陈述
        """
        # 对于观点冲突，通常保留多种观点
        statements = conflict["statements"]
        return f"存在不同观点：一种观点认为，{statements[0]}另一种观点认为，{statements[1]}"
    
    def _resolve_recommendation_conflict(self, conflict: Dict[str, Any], responses: List[Dict[str, Any]]) -> str:
        """解决推荐冲突
        
        Args:
            conflict: 冲突信息
            responses: 原始响应列表
            
        Returns:
            str: 解决后的推荐陈述
        """
        # 获取冲突的推荐和对应的Agent
        statements = conflict["statements"]
        agents = conflict["agents"]
        
        # 获取对应Agent的置信度
        confidences = []
        for agent_id in agents:
            for resp in responses:
                if resp.get("agent_id") == agent_id:
                    confidences.append(resp.get("confidence", 0.5))
                    break
            else:
                confidences.append(0.5)  # 默认置信度
        
        # 使用置信度加权选择
        if confidences[0] > confidences[1] * 1.5:  # 如果第一个Agent的置信度明显更高
            return f"推荐：{statements[0]}"
        elif confidences[1] > confidences[0] * 1.5:  # 如果第二个Agent的置信度明显更高
            return f"推荐：{statements[1]}"
        else:  # 置信度相近，提供多个选项
            return f"有多种建议：\n- {statements[0]}\n- {statements[1]}"
    
    def _build_unified_response(self, 
                               responses: List[Dict[str, Any]], 
                               factual_resolutions: Dict[str, str], 
                               opinion_resolutions: Dict[str, str], 
                               recommendation_resolutions: Dict[str, str]) -> Dict[str, Any]:
        """构建统一响应
        
        Args:
            responses: 原始响应列表
            factual_resolutions: 事实性冲突的解决结果
            opinion_resolutions: 观点冲突的解决结果
            recommendation_resolutions: 推荐冲突的解决结果
            
        Returns:
            Dict[str, Any]: 统一响应
        """
        # 选择置信度最高的响应作为基础
        base_response = max(responses, key=lambda x: x.get("confidence", 0.5))
        base_content = base_response.get("content", "")
        
        # 构建新的内容，替换冲突部分
        new_content = base_content
        
        # 替换事实性冲突
        for key, resolution in factual_resolutions.items():
            # 这里简化处理，实际应用中需要更精确的替换
            for conflict_statement in key.split("_")[1:]:  # 从key中提取冲突ID
                for response in responses:
                    content = response.get("content", "")
                    if conflict_statement in content:
                        new_content = new_content.replace(conflict_statement, resolution)
        
        # 添加观点和推荐的解决结果
        if opinion_resolutions or recommendation_resolutions:
            new_content += "\n\n补充信息：\n"
            
            for resolution in opinion_resolutions.values():
                new_content += f"\n{resolution}"
            
            for resolution in recommendation_resolutions.values():
                new_content += f"\n{resolution}"
        
        # 计算平均置信度
        avg_confidence = sum(resp.get("confidence", 0.5) for resp in responses) / len(responses)
        
        return {
            "content": new_content,
            "confidence": avg_confidence,
            "sources": [resp.get("agent_id") for resp in responses],
            "has_conflicts": bool(factual_resolutions or opinion_resolutions or recommendation_resolutions),
            "conflict_count": len(factual_resolutions) + len(opinion_resolutions) + len(recommendation_resolutions)
        }
    
    def apply_majority_voting(self, statements: List[str], confidences: List[float] = None) -> str:
        """应用多数投票策略解决冲突
        
        Args:
            statements: 冲突的陈述列表
            confidences: 对应的置信度列表
            
        Returns:
            str: 解决后的陈述
        """
        # 如果没有提供置信度，假设所有陈述权重相等
        if not confidences:
            confidences = [1.0] * len(statements)
        
        # 计算每个陈述的相似组
        similarity_groups = []
        statement_groups = []
        
        for i, stmt1 in enumerate(statements):
            found_group = False
            for j, group in enumerate(statement_groups):
                # 检查与组中第一个陈述的相似度
                similarity = SequenceMatcher(None, stmt1, group[0]).ratio()
                if similarity > self.config["similarity_threshold"]:
                    statement_groups[j].append(stmt1)
                    similarity_groups[j].append((stmt1, confidences[i]))
                    found_group = True
                    break
            
            if not found_group:
                statement_groups.append([stmt1])
                similarity_groups.append([(stmt1, confidences[i])])
        
        # 找出权重最高的组
        max_weight = 0
        max_group_idx = 0
        
        for i, group in enumerate(similarity_groups):
            weight = sum(conf for _, conf in group)
            if weight > max_weight:
                max_weight = weight
                max_group_idx = i
        
        # 返回权重最高组中置信度最高的陈述
        max_group = similarity_groups[max_group_idx]
        return max(max_group, key=lambda x: x[1])[0]
    
    def apply_confidence_weighted(self, statements: List[str], confidences: List[float]) -> str:
        """应用置信度加权策略解决冲突
        
        Args:
            statements: 冲突的陈述列表
            confidences: 对应的置信度列表
            
        Returns:
            str: 解决后的陈述
        """
        if not statements:
            return ""
        
        # 找出置信度最高的陈述
        max_confidence_idx = confidences.index(max(confidences))
        return statements[max_confidence_idx]
    
    def apply_semantic_analysis(self, statements: List[str], confidences: List[float] = None) -> str:
        """应用语义分析策略解决冲突
        
        Args:
            statements: 冲突的陈述列表
            confidences: 对应的置信度列表
            
        Returns:
            str: 解决后的陈述
        """
        # 简化实现：提取共同部分和差异部分
        if len(statements) < 2:
            return statements[0] if statements else ""
        
        # 找出最相似的两个陈述
        max_similarity = 0
        similar_pair = (0, 1)
        
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                similarity = SequenceMatcher(None, stmt1, stmt2).ratio()
                if similarity > max_similarity:
                    max_similarity = similarity
                    similar_pair = (i, j)
        
        # 提取共同部分
        stmt1 = statements[similar_pair[0]]
        stmt2 = statements[similar_pair[1]]
        
        # 使用difflib找出共同部分和差异部分
        matcher = SequenceMatcher(None, stmt1, stmt2)
        common_parts = []
        
        for block in matcher.get_matching_blocks():
            if block.size > 5:  # 只考虑较长的共同部分
                common_parts.append(stmt1[block.a:block.a+block.size])
        
        if common_parts:
            return " ".join(common_parts)
        else:
            # 如果没有明显的共同部分，返回置信度最高的陈述
            if confidences:
                return self.apply_confidence_weighted(statements, confidences)
            else:
                return statements[0]
    
    def apply_source_reliability(self, statements: List[str], agents: List[str], responses: List[Dict[str, Any]]) -> str:
        """应用源可靠性策略解决冲突
        
        Args:
            statements: 冲突的陈述列表
            agents: 对应的Agent列表
            responses: 原始响应列表
            
        Returns:
            str: 解决后的陈述
        """
        # 简化实现：根据Agent的历史可靠性选择陈述
        # 这里假设所有Agent可靠性相同，使用置信度代替
        confidences = []
        
        for agent_id in agents:
            for resp in responses:
                if resp.get("agent_id") == agent_id:
                    confidences.append(resp.get("confidence", 0.5))
                    break
            else:
                confidences.append(0.5)  # 默认置信度
        
        return self.apply_confidence_weighted(statements, confidences)