import json
import os
from typing import Dict, List, Any
from memory.memory_manager import MemoryManager

class Aggregator:
    """聚合器，负责跨Agent信息融合和统一响应生成"""
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.load_config()

    def load_config(self):
        """加载聚合器配置"""
        # 当前版本使用默认配置
        self.config = {
            "fusion_strategy": "weighted_merge",
            "confidence_threshold": 0.7,
            "max_context_length": 4000
        }

    def fuse_responses(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """融合多个Agent的响应

        Args:
            agent_responses: Agent响应列表

        Returns:
            Dict[str, Any]: 融合后的统一响应
        """
        if not agent_responses:
            return {"content": "", "confidence": 0.0}

        # 根据融合策略选择处理方式
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
            "sources": [resp.get("agent_id") for resp in agent_responses]
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
            "source": best_response.get("agent_id")
        }

    def _simple_merge(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """简单的响应合并

        Args:
            agent_responses: Agent响应列表

        Returns:
            Dict[str, Any]: 合并后的响应
        """
        merged_content = "\n".join([resp.get("content", "") for resp in agent_responses])
        avg_confidence = sum(resp.get("confidence", 0.5) for resp in agent_responses) / len(agent_responses)
        
        return {
            "content": merged_content,
            "confidence": avg_confidence,
            "sources": [resp.get("agent_id") for resp in agent_responses]
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
        if relevant_memories:
            context_parts.append("相关历史信息:")
            for memory in relevant_memories:
                context_parts.append(f"- {memory['content']}")
        
        # 添加融合的响应内容
        context_parts.append("响应内容:")
        context_parts.append(fused_response.get("content", ""))
        
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

# 单例模式
aggregator = Aggregator()