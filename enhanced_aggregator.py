import json
import os
from typing import Dict, List, Any, Optional
from memory.memory_manager import MemoryManager
from core.conflict_resolver import ConflictResolver

class EnhancedAggregator:
    """增强型聚合器，负责跨Agent信息融合、冲突解决和统一响应生成"""
    def __init__(self, config_path: Optional[str] = None):
        """初始化增强型聚合器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.memory_manager = MemoryManager()
        self.conflict_resolver = ConflictResolver()
        self.load_config(config_path)
    
    def load_config(self, config_path: Optional[str] = None):
        """加载聚合器配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        # 默认配置
        self.config = {
            "fusion_strategy": "weighted_merge",
            "confidence_threshold": 0.7,
            "max_context_length": 4000,
            "conflict_resolution": {
                "enabled": True,
                "strategies": ["majority_voting", "confidence_weighted", "semantic_analysis"],
                "default_strategy": "confidence_weighted",
                "similarity_threshold": 0.7,
                "contradiction_threshold": 0.3
            },
            "response_format": {
                "include_confidence": False,
                "include_sources": True,
                "include_conflict_info": True
            }
        }
        
        # 如果提供了配置文件路径，从文件加载配置
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # 更新配置，保留默认值
                    self._update_config_recursive(self.config, file_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
        
        # 更新冲突解决器配置
        if "conflict_resolution" in self.config:
            conflict_config = self.config["conflict_resolution"]
            self.conflict_resolver.config.update({
                "similarity_threshold": conflict_config.get("similarity_threshold", 0.7),
                "contradiction_threshold": conflict_config.get("contradiction_threshold", 0.3),
                "resolution_strategies": conflict_config.get("strategies", ["majority_voting", "confidence_weighted"]),
                "default_strategy": conflict_config.get("default_strategy", "confidence_weighted")
            })
    
    def _update_config_recursive(self, target: Dict, source: Dict):
        """递归更新配置
        
        Args:
            target: 目标配置字典
            source: 源配置字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config_recursive(target[key], value)
            else:
                target[key] = value
    
    def fuse_responses(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """融合多个Agent的响应

        Args:
            agent_responses: Agent响应列表

        Returns:
            Dict[str, Any]: 融合后的统一响应
        """
        if not agent_responses:
            return {"content": "", "confidence": 0.0}
        
        # 如果只有一个响应，直接返回
        if len(agent_responses) == 1:
            return agent_responses[0]
        
        # 检测并解决冲突
        if self.config["conflict_resolution"]["enabled"]:
            conflicts = self.conflict_resolver.detect_conflicts(agent_responses)
            
            if conflicts:
                # 有冲突，使用冲突解决器处理
                resolved_response = self.conflict_resolver.resolve_conflicts(conflicts, agent_responses)
                resolved_response["conflict_info"] = {
                    "detected": True,
                    "count": len(conflicts),
                    "types": list(set(conflict["type"] for conflict in conflicts))
                }
                return resolved_response
        
        # 无冲突或冲突解决未启用，使用融合策略
        strategy = self.config["fusion_strategy"]
        
        if strategy == "weighted_merge":
            return self._weighted_merge(agent_responses)
        elif strategy == "confidence_based":
            return self._confidence_based_selection(agent_responses)
        else:
            # 默认简单合并
            return self._simple_merge(agent_responses)
    
    def _weighted_merge(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于权重的响应融合

        Args:
            agent_responses: Agent响应列表

        Returns:
            Dict[str, Any]: 融合后的响应
        """
        # 计算总权重
        total_weight = sum(resp.get("confidence", 0.5) for resp in agent_responses)
        
        if total_weight == 0:
            # 如果总权重为0，使用简单合并
            return self._simple_merge(agent_responses)
        
        # 加权合并内容
        fused_content = ""
        for resp in agent_responses:
            weight = resp.get("confidence", 0.5) / total_weight
            content = resp.get("content", "")
            fused_content += f"{content} "
        
        # 计算融合后的置信度
        avg_confidence = sum(resp.get("confidence", 0.5) for resp in agent_responses) / len(agent_responses)
        
        return {
            "content": fused_content.strip(),
            "confidence": avg_confidence,
            "sources": [resp.get("agent_id") for resp in agent_responses],
            "conflict_info": {"detected": False}
        }
    
    def _confidence_based_selection(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于置信度的选择

        Args:
            agent_responses: Agent响应列表

        Returns:
            Dict[str, Any]: 选择的响应
        """
        # 选择置信度最高的响应
        best_response = max(agent_responses, key=lambda x: x.get("confidence", 0.5))
        
        # 如果最高置信度低于阈值，使用加权合并
        if best_response.get("confidence", 0.5) < self.config["confidence_threshold"]:
            return self._weighted_merge(agent_responses)
        
        return {
            "content": best_response.get("content", ""),
            "confidence": best_response.get("confidence", 0.5),
            "source": best_response.get("agent_id"),
            "conflict_info": {"detected": False}
        }
    
    def _simple_merge(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """简单的响应合并

        Args:
            agent_responses: Agent响应列表

        Returns:
            Dict[str, Any]: 合并后的响应
        """
        merged_content = "\n\n".join([f"来自 {resp.get('agent_id', '未知Agent')} 的响应:\n{resp.get('content', '')}" 
                              for resp in agent_responses])
        avg_confidence = sum(resp.get("confidence", 0.5) for resp in agent_responses) / len(agent_responses)
        
        return {
            "content": merged_content,
            "confidence": avg_confidence,
            "sources": [resp.get("agent_id") for resp in agent_responses],
            "conflict_info": {"detected": False}
        }
    
    def generate_final_response(self, fused_response: Dict[str, Any], query: str) -> str:
        """生成最终响应

        Args:
            fused_response: 融合后的响应
            query: 用户查询

        Returns:
            str: 最终响应文本
        """
        # 从记忆中检索相关信息
        relevant_memories = self.memory_manager.retrieve_memory(query, top_k=3)
        
        # 构建上下文
        context_parts = []
        
        # 添加融合的响应内容
        context_parts.append(fused_response.get("content", ""))
        
        # 添加冲突信息（如果启用）
        if self.config["response_format"]["include_conflict_info"] and fused_response.get("conflict_info", {}).get("detected", False):
            conflict_info = fused_response.get("conflict_info", {})
            context_parts.append(f"\n\n注意：在生成此回答时，系统检测到了{conflict_info.get('count', 0)}个信息冲突，并已尝试解决。")
        
        # 添加来源信息（如果启用）
        if self.config["response_format"]["include_sources"] and "sources" in fused_response:
            sources = fused_response.get("sources", [])
            if len(sources) > 1:
                context_parts.append(f"\n\n此回答综合了以下Agent的信息：{', '.join(sources)}")
        
        # 添加置信度信息（如果启用）
        if self.config["response_format"]["include_confidence"] and "confidence" in fused_response:
            confidence = fused_response.get("confidence", 0.5)
            context_parts.append(f"\n\n响应置信度：{confidence:.2f}")
        
        # 如果有相关记忆，添加到响应中
        if relevant_memories:
            memory_parts = ["\n\n相关历史信息:"]
            for memory in relevant_memories:
                memory_parts.append(f"- {memory['content']}")
            context_parts.append("\n".join(memory_parts))
        
        # 构建最终响应
        final_response = "\n".join(context_parts)
        
        # 保存交互到记忆
        interaction = {
            "query": query,
            "response": final_response,
            "confidence": fused_response.get("confidence", 0.5)
        }
        self.memory_manager.store_memory("interaction", json.dumps(interaction))
        
        return final_response
    
    def aggregate_responses(self, agent_responses: List[Dict[str, Any]], query: str) -> str:
        """聚合多个Agent的响应并生成最终响应
        
        Args:
            agent_responses: Agent响应列表
            query: 用户查询
            
        Returns:
            str: 最终响应文本
        """
        # 融合响应
        fused_response = self.fuse_responses(agent_responses)
        
        # 生成最终响应
        return self.generate_final_response(fused_response, query)
    
    def provide_feedback(self, query: str, response: str, feedback: Dict[str, Any]):
        """提供反馈以改进聚合器
        
        Args:
            query: 用户查询
            response: 系统响应
            feedback: 反馈信息，包含评分、评论等
        """
        # 保存反馈到记忆
        feedback_data = {
            "query": query,
            "response": response,
            "rating": feedback.get("rating", 0),
            "comment": feedback.get("comment", ""),
            "conflict_detected": feedback.get("conflict_detected", False),
            "conflict_resolved": feedback.get("conflict_resolved", False)
        }
        
        self.memory_manager.store_memory("feedback", json.dumps(feedback_data))
        
        # 如果反馈中包含冲突信息，可以用于调整冲突解决策略
        if "conflict_detected" in feedback and "conflict_resolved" in feedback:
            # 这里可以实现更复杂的策略调整逻辑
            pass