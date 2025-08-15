from typing import Dict, List, Any, Optional, Tuple
import json
import time
import numpy as np
from collections import Counter

class RoutingStrategy:
    """路由策略基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化路由策略
        
        Args:
            config: 策略配置
        """
        self.config = config or {}
    
    def select_agent(self, query: str, agent_profiles: Dict[str, Dict[str, Any]], 
                    context: Optional[List[Dict[str, str]]] = None) -> str:
        """选择合适的Agent处理查询
        
        Args:
            query: 用户查询
            agent_profiles: Agent配置信息
            context: 对话上下文
            
        Returns:
            选定的Agent ID
        """
        raise NotImplementedError("子类必须实现select_agent方法")


class HistoryBasedStrategy(RoutingStrategy):
    """基于历史交互的路由策略
    
    根据用户历史交互记录，分析用户偏好和查询模式，选择最合适的Agent
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.interaction_history = []
        self.agent_performance = {}
        self.user_preferences = {}
        self.history_file = self.config.get("history_file", "interaction_history.json")
        self.load_history()
    
    def load_history(self):
        """加载历史交互记录"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.interaction_history = data.get("interactions", [])
                self.agent_performance = data.get("performance", {})
                self.user_preferences = data.get("preferences", {})
        except (FileNotFoundError, json.JSONDecodeError):
            # 文件不存在或格式错误，使用空记录
            pass
    
    def save_history(self):
        """保存历史交互记录"""
        data = {
            "interactions": self.interaction_history,
            "performance": self.agent_performance,
            "preferences": self.user_preferences
        }
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def update_history(self, query: str, agent_id: str, success: bool, feedback: Optional[int] = None):
        """更新历史交互记录
        
        Args:
            query: 用户查询
            agent_id: 处理查询的Agent ID
            success: 处理是否成功
            feedback: 用户反馈评分（1-5）
        """
        # 添加交互记录
        interaction = {
            "query": query,
            "agent_id": agent_id,
            "timestamp": time.time(),
            "success": success
        }
        if feedback is not None:
            interaction["feedback"] = feedback
        
        self.interaction_history.append(interaction)
        
        # 限制历史记录大小
        max_history = self.config.get("max_history_size", 1000)
        if len(self.interaction_history) > max_history:
            self.interaction_history = self.interaction_history[-max_history:]
        
        # 更新Agent性能记录
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {"success": 0, "total": 0, "feedback_sum": 0, "feedback_count": 0}
        
        self.agent_performance[agent_id]["total"] += 1
        if success:
            self.agent_performance[agent_id]["success"] += 1
        
        if feedback is not None:
            self.agent_performance[agent_id]["feedback_sum"] += feedback
            self.agent_performance[agent_id]["feedback_count"] += 1
        
        # 保存更新后的历史记录
        self.save_history()
    
    def get_user_preference(self, query: str) -> Dict[str, float]:
        """分析用户对不同Agent的偏好
        
        Args:
            query: 用户查询
            
        Returns:
            各Agent的偏好得分
        """
        # 简单实现：基于历史交互中用户对各Agent的反馈评分
        preferences = {}
        
        # 默认所有Agent的偏好得分为0
        for agent_id in self.agent_performance:
            preferences[agent_id] = 0.0
        
        # 计算每个Agent的平均反馈评分
        for agent_id, perf in self.agent_performance.items():
            if perf["feedback_count"] > 0:
                avg_feedback = perf["feedback_sum"] / perf["feedback_count"]
                preferences[agent_id] = avg_feedback
        
        return preferences
    
    def get_query_similarity(self, query: str) -> Dict[str, float]:
        """计算当前查询与历史查询的相似度
        
        Args:
            query: 用户查询
            
        Returns:
            各Agent的查询相似度得分
        """
        # 简单实现：基于关键词匹配的相似度计算
        query_words = set(query.lower().split())
        similarity_scores = {}
        
        # 默认所有Agent的相似度得分为0
        for agent_id in self.agent_performance:
            similarity_scores[agent_id] = 0.0
        
        # 计算当前查询与历史查询的相似度
        for interaction in self.interaction_history:
            hist_query = interaction["agent_id"]
            hist_words = set(hist_query.lower().split())
            
            # 计算Jaccard相似度
            if hist_words and query_words:
                intersection = len(query_words.intersection(hist_words))
                union = len(query_words.union(hist_words))
                similarity = intersection / union if union > 0 else 0
                
                # 更新相似度得分
                agent_id = interaction["agent_id"]
                similarity_scores[agent_id] = max(similarity_scores.get(agent_id, 0), similarity)
        
        return similarity_scores
    
    def select_agent(self, query: str, agent_profiles: Dict[str, Dict[str, Any]], 
                    context: Optional[List[Dict[str, str]]] = None) -> str:
        """基于历史交互选择合适的Agent
        
        Args:
            query: 用户查询
            agent_profiles: Agent配置信息
            context: 对话上下文
            
        Returns:
            选定的Agent ID
        """
        if not agent_profiles:
            return "default_agent"
        
        # 获取各项评分
        preferences = self.get_user_preference(query)
        similarities = self.get_query_similarity(query)
        
        # 计算Agent的历史成功率
        success_rates = {}
        for agent_id, perf in self.agent_performance.items():
            if perf["total"] > 0:
                success_rates[agent_id] = perf["success"] / perf["total"]
            else:
                success_rates[agent_id] = 0.0
        
        # 计算综合得分
        scores = {}
        for agent_id in agent_profiles:
            # 如果是新Agent，给予一定的探索机会
            if agent_id not in self.agent_performance:
                scores[agent_id] = self.config.get("exploration_score", 0.5)
                continue
            
            # 综合考虑用户偏好、查询相似度和历史成功率
            pref_weight = self.config.get("preference_weight", 0.4)
            sim_weight = self.config.get("similarity_weight", 0.3)
            success_weight = self.config.get("success_weight", 0.3)
            
            score = (
                pref_weight * preferences.get(agent_id, 0) / 5.0 +  # 归一化到0-1
                sim_weight * similarities.get(agent_id, 0) +
                success_weight * success_rates.get(agent_id, 0)
            )
            
            scores[agent_id] = score
        
        # 选择得分最高的Agent
        if not scores:
            return "default_agent"
        
        return max(scores.items(), key=lambda x: x[1])[0]


class SemanticRoutingStrategy(RoutingStrategy):
    """基于语义理解的路由策略
    
    使用语义分析技术，理解查询意图和内容，选择最合适的Agent
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.embedding_cache = {}
        self.intent_patterns = self.config.get("intent_patterns", {})
    
    def get_query_embedding(self, query: str) -> List[float]:
        """获取查询的向量表示
        
        Args:
            query: 用户查询
            
        Returns:
            查询的向量表示
        """
        # 实际应用中应使用适当的embedding模型
        # 这里使用简化的实现
        if query in self.embedding_cache:
            return self.embedding_cache[query]
        
        try:
            from adapters.llm import get_embedding
            embedding = get_embedding(query)
            self.embedding_cache[query] = embedding
            return embedding
        except ImportError:
            # 如果没有embedding模型，返回随机向量
            import random
            random_embedding = [random.random() for _ in range(128)]
            self.embedding_cache[query] = random_embedding
            return random_embedding
    
    def detect_intent(self, query: str) -> Dict[str, float]:
        """检测查询意图
        
        Args:
            query: 用户查询
            
        Returns:
            各意图的置信度
        """
        # 简单实现：基于关键词匹配的意图检测
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern.lower() in query_lower:
                    score += 1.0
            
            if patterns:  # 避免除零错误
                score /= len(patterns)
            
            intent_scores[intent] = score
        
        return intent_scores
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度
        """
        # 确保向量长度相同
        if len(vec1) != len(vec2):
            return 0.0
        
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def select_agent(self, query: str, agent_profiles: Dict[str, Dict[str, Any]], 
                    context: Optional[List[Dict[str, str]]] = None) -> str:
        """基于语义理解选择合适的Agent
        
        Args:
            query: 用户查询
            agent_profiles: Agent配置信息
            context: 对话上下文
            
        Returns:
            选定的Agent ID
        """
        if not agent_profiles:
            return "default_agent"
        
        # 获取查询的向量表示
        query_embedding = self.get_query_embedding(query)
        
        # 检测查询意图
        intent_scores = self.detect_intent(query)
        
        # 计算查询与各Agent描述的语义相似度
        similarity_scores = {}
        for agent_id, profile in agent_profiles.items():
            # 获取Agent描述的向量表示
            description = profile.get("description", "")
            if description:
                desc_embedding = self.get_query_embedding(description)
                similarity = self.calculate_similarity(query_embedding, desc_embedding)
                similarity_scores[agent_id] = similarity
            else:
                similarity_scores[agent_id] = 0.0
        
        # 计算综合得分
        scores = {}
        for agent_id, profile in agent_profiles.items():
            # 获取Agent支持的意图
            supported_intents = profile.get("supported_intents", [])
            
            # 计算意图匹配得分
            intent_match_score = 0.0
            for intent in supported_intents:
                intent_match_score += intent_scores.get(intent, 0.0)
            
            if supported_intents:  # 避免除零错误
                intent_match_score /= len(supported_intents)
            
            # 综合考虑语义相似度和意图匹配度
            sim_weight = self.config.get("similarity_weight", 0.6)
            intent_weight = self.config.get("intent_weight", 0.4)
            
            score = (
                sim_weight * similarity_scores.get(agent_id, 0.0) +
                intent_weight * intent_match_score
            )
            
            scores[agent_id] = score
        
        # 选择得分最高的Agent
        if not scores:
            return "default_agent"
        
        return max(scores.items(), key=lambda x: x[1])[0]


class AdaptiveRoutingStrategy(RoutingStrategy):
    """自适应路由策略
    
    根据系统状态和查询特性，动态选择最合适的路由策略
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.strategies = {}
        self.strategy_performance = {}
        self.initialize_strategies()
    
    def initialize_strategies(self):
        """初始化各种路由策略"""
        # 创建历史交互策略
        history_config = self.config.get("history_strategy_config", {})
        self.strategies["history"] = HistoryBasedStrategy(history_config)
        
        # 创建语义路由策略
        semantic_config = self.config.get("semantic_strategy_config", {})
        self.strategies["semantic"] = SemanticRoutingStrategy(semantic_config)
        
        # 初始化策略性能记录
        for strategy_name in self.strategies:
            self.strategy_performance[strategy_name] = {
                "success": 0,
                "total": 0,
                "recent_success": []
            }
    
    def update_strategy_performance(self, strategy_name: str, success: bool):
        """更新策略性能记录
        
        Args:
            strategy_name: 策略名称
            success: 是否成功
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "success": 0,
                "total": 0,
                "recent_success": []
            }
        
        # 更新总体统计
        self.strategy_performance[strategy_name]["total"] += 1
        if success:
            self.strategy_performance[strategy_name]["success"] += 1
        
        # 更新最近成功记录
        recent_window = self.config.get("recent_window_size", 10)
        recent_success = self.strategy_performance[strategy_name]["recent_success"]
        recent_success.append(1 if success else 0)
        
        # 保持窗口大小
        if len(recent_success) > recent_window:
            recent_success = recent_success[-recent_window:]
        
        self.strategy_performance[strategy_name]["recent_success"] = recent_success
    
    def select_strategy(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """选择合适的路由策略
        
        Args:
            query: 用户查询
            context: 对话上下文
            
        Returns:
            选定的策略名称
        """
        # 计算各策略的性能得分
        strategy_scores = {}
        
        for strategy_name, perf in self.strategy_performance.items():
            # 计算总体成功率
            overall_success_rate = 0.0
            if perf["total"] > 0:
                overall_success_rate = perf["success"] / perf["total"]
            
            # 计算最近成功率
            recent_success_rate = 0.0
            recent_success = perf["recent_success"]
            if recent_success:
                recent_success_rate = sum(recent_success) / len(recent_success)
            
            # 综合考虑总体成功率和最近成功率
            overall_weight = self.config.get("overall_weight", 0.3)
            recent_weight = self.config.get("recent_weight", 0.7)
            
            score = (
                overall_weight * overall_success_rate +
                recent_weight * recent_success_rate
            )
            
            strategy_scores[strategy_name] = score
        
        # 如果没有足够的性能记录，使用探索-利用策略
        min_samples = self.config.get("min_strategy_samples", 5)
        exploration_rate = self.config.get("exploration_rate", 0.2)
        
        for strategy_name, perf in self.strategy_performance.items():
            if perf["total"] < min_samples:
                # 给予更多探索机会
                strategy_scores[strategy_name] = 0.5 + 0.5 * (perf["total"] / min_samples)
        
        # 随机探索
        if np.random.random() < exploration_rate:
            return np.random.choice(list(self.strategies.keys()))
        
        # 选择得分最高的策略
        if not strategy_scores:
            return "semantic"  # 默认使用语义路由策略
        
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def select_agent(self, query: str, agent_profiles: Dict[str, Dict[str, Any]], 
                    context: Optional[List[Dict[str, str]]] = None) -> str:
        """自适应选择合适的Agent
        
        Args:
            query: 用户查询
            agent_profiles: Agent配置信息
            context: 对话上下文
            
        Returns:
            选定的Agent ID
        """
        if not agent_profiles:
            return "default_agent"
        
        # 选择路由策略
        strategy_name = self.select_strategy(query, context)
        strategy = self.strategies.get(strategy_name)
        
        if not strategy:
            # 如果策略不存在，使用语义路由策略
            strategy_name = "semantic"
            strategy = self.strategies.get(strategy_name, SemanticRoutingStrategy())
        
        # 使用选定的策略选择Agent
        agent_id = strategy.select_agent(query, agent_profiles, context)
        
        return agent_id
    
    def feedback(self, strategy_name: str, success: bool):
        """提供策略执行结果的反馈
        
        Args:
            strategy_name: 策略名称
            success: 是否成功
        """
        self.update_strategy_performance(strategy_name, success)


class MultiAgentRoutingStrategy(RoutingStrategy):
    """多Agent协作路由策略
    
    将查询分发给多个Agent处理，然后聚合结果
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.task_decomposer = None
        self.result_aggregator = None
    
    def decompose_task(self, query: str) -> List[Dict[str, Any]]:
        """将查询分解为多个子任务
        
        Args:
            query: 用户查询
            
        Returns:
            子任务列表
        """
        # 简单实现：根据查询类型和复杂度分解任务
        # 实际应用中应使用更复杂的任务分解算法
        
        # 检测查询是否需要分解
        if len(query.split()) < 10:  # 简单查询不分解
            return [{
                "sub_query": query,
                "type": "simple"
            }]
        
        # 分解复杂查询
        sentences = query.split('. ')
        sub_tasks = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            sub_tasks.append({
                "sub_query": sentence,
                "type": "decomposed",
                "order": i
            })
        
        return sub_tasks
    
    def select_agents_for_subtasks(self, sub_tasks: List[Dict[str, Any]], 
                                agent_profiles: Dict[str, Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
        """为每个子任务选择合适的Agent
        
        Args:
            sub_tasks: 子任务列表
            agent_profiles: Agent配置信息
            
        Returns:
            子任务和对应Agent的列表
        """
        # 创建语义路由策略用于子任务分配
        semantic_strategy = SemanticRoutingStrategy(self.config.get("semantic_strategy_config", {}))
        
        # 为每个子任务分配Agent
        task_agent_pairs = []
        
        for task in sub_tasks:
            sub_query = task["sub_query"]
            agent_id = semantic_strategy.select_agent(sub_query, agent_profiles)
            task_agent_pairs.append((task, agent_id))
        
        return task_agent_pairs
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多个Agent的处理结果
        
        Args:
            results: 各Agent的处理结果
            
        Returns:
            聚合后的结果
        """
        # 简单实现：按子任务顺序拼接结果
        # 实际应用中应使用更复杂的结果聚合算法
        
        # 按顺序排序结果
        sorted_results = sorted(results, key=lambda x: x.get("task", {}).get("order", 0))
        
        # 拼接响应
        aggregated_response = ""
        used_agents = set()
        
        for result in sorted_results:
            response = result.get("response", "")
            agent_id = result.get("agent_id", "unknown")
            
            if response:
                aggregated_response += response + "\n\n"
                used_agents.add(agent_id)
        
        return {
            "response": aggregated_response.strip(),
            "agents_used": list(used_agents),
            "success": True
        }
    
    def select_agent(self, query: str, agent_profiles: Dict[str, Dict[str, Any]], 
                    context: Optional[List[Dict[str, str]]] = None) -> str:
        """多Agent协作处理查询
        
        注意：此方法在多Agent协作策略中的行为与其他策略不同，
        它不仅选择Agent，还会处理查询并返回结果。
        
        Args:
            query: 用户查询
            agent_profiles: Agent配置信息
            context: 对话上下文
            
        Returns:
            主要处理Agent的ID（用于兼容接口）
        """
        # 在实际应用中，此方法应返回主要Agent的ID，
        # 并通过其他机制处理多Agent协作
        
        # 分解任务
        sub_tasks = self.decompose_task(query)
        
        # 如果只有一个子任务，使用语义路由策略
        if len(sub_tasks) == 1:
            semantic_strategy = SemanticRoutingStrategy(self.config.get("semantic_strategy_config", {}))
            return semantic_strategy.select_agent(query, agent_profiles, context)
        
        # 为子任务分配Agent
        task_agent_pairs = self.select_agents_for_subtasks(sub_tasks, agent_profiles)
        
        # 统计各Agent分配的任务数
        agent_counts = Counter([agent_id for _, agent_id in task_agent_pairs])
        
        # 返回分配任务最多的Agent的ID
        if not agent_counts:
            return "default_agent"
        
        return agent_counts.most_common(1)[0][0]


class StrategyFactory:
    """路由策略工厂类"""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any] = None) -> RoutingStrategy:
        """创建指定类型的路由策略
        
        Args:
            strategy_type: 策略类型
            config: 策略配置
            
        Returns:
            路由策略实例
        """
        if strategy_type == "history":
            return HistoryBasedStrategy(config)
        elif strategy_type == "semantic":
            return SemanticRoutingStrategy(config)
        elif strategy_type == "adaptive":
            return AdaptiveRoutingStrategy(config)
        elif strategy_type == "multi_agent":
            return MultiAgentRoutingStrategy(config)
        else:
            # 默认使用语义路由策略
            return SemanticRoutingStrategy(config)