from adapters.llm import call_model, lite_llm_infer
import json
import os
from typing import Dict, List, Any, Optional
from core.routing_strategies import StrategyFactory

class EnhancedRouter:
    def __init__(self, config_path: str = "configs/router_config.json"):
        """初始化增强型路由系统

        Args:
            config_path: 路由配置文件路径
        """
        self.config = self._load_config(config_path)
        self.agent_profiles = self.config.get("agent_profiles", {})
        self.routing_strategies_config = self.config.get("routing_strategies", {})
        self.agent_strategy_mapping = self.config.get("agent_strategy_mapping", {})
        self.default_strategy_name = self.config.get("default_strategy", "direct")
        self.routing_threshold = self.config.get("routing_threshold", 0.75)
        self.fallback_agent = self.config.get("fallback_agent", "default_agent")
        
        # 初始化路由策略实例
        self.strategy_instances = {}
        self._initialize_strategies()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载路由配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 配置文件 {config_path} 未找到，使用默认配置")
            return {
                "agent_profiles": {},
                "routing_strategies": {},
                "agent_strategy_mapping": {},
                "default_strategy": "direct"
            }
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {config_path} 格式无效，使用默认配置")
            return {
                "agent_profiles": {},
                "routing_strategies": {},
                "agent_strategy_mapping": {},
                "default_strategy": "direct"
            }
    
    def _initialize_strategies(self):
        """初始化路由策略实例"""
        for strategy_name, strategy_config in self.routing_strategies_config.items():
            if strategy_config.get("enabled", True):
                try:
                    self.strategy_instances[strategy_name] = StrategyFactory.create_strategy(
                        strategy_name, 
                        strategy_config.get("config", {})
                    )
                except Exception as e:
                    print(f"警告: 初始化策略 {strategy_name} 失败: {str(e)}")
    
    def determine_agent(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """根据查询和上下文确定最适合的Agent

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            选定的Agent ID
        """
        # 获取默认策略
        strategy_name = self.default_strategy_name
        strategy = self.strategy_instances.get(strategy_name)
        
        if not strategy:
            # 如果默认策略不可用，回退到基于LLM的决策
            return self._llm_based_routing(query, context)
        
        try:
            # 使用策略选择Agent
            agent_id = strategy.select_agent(query, self.agent_profiles, context)
            return agent_id
        except Exception as e:
            print(f"警告: 使用策略 {strategy_name} 路由失败: {str(e)}")
            return self._llm_based_routing(query, context)
    
    def _llm_based_routing(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """基于LLM的路由决策

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            选定的Agent ID
        """
        if not self.agent_profiles:
            return self.fallback_agent

        # 构建提示词
        agent_descriptions = "\n".join([f"{agent_id}: {profile['description']}" for agent_id, profile in self.agent_profiles.items()])

        prompt = f"""
        你是一个智能路由系统，需要根据用户查询和可用Agent的描述，选择最适合处理该查询的Agent。

        可用Agent:
        {agent_descriptions}

        用户查询: {query}

        请仅返回最适合的Agent ID，不要添加任何解释。
        """

        # 使用轻量级LLM推理
        response = lite_llm_infer(prompt)
        response = response.strip()

        # 验证响应是否为有效的Agent ID
        if response in self.agent_profiles:
            return response
        else:
            print(f"警告: 无法确定合适的Agent，使用默认Agent。LLM响应: {response}")
            return self.fallback_agent
    
    def route_query(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """路由查询到合适的Agent

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            路由结果，包含选定的Agent和处理策略
        """
        agent_id = self.determine_agent(query, context)
        strategy_name = self.agent_strategy_mapping.get(agent_id, self.default_strategy_name)

        return {
            "agent_id": agent_id,
            "strategy": strategy_name,
            "timestamp": "",  # 实际应用中添加时间戳
            "query": query
        }
    
    def execute_route(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """执行路由并获取结果

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            处理结果
        """
        route_info = self.route_query(query, context)
        agent_id = route_info["agent_id"]
        strategy_name = route_info["strategy"]
        
        # 获取Agent配置
        agent_config = self.agent_profiles.get(agent_id, {})
        model = agent_config.get("model", "gpt-3.5-turbo")
        temperature = agent_config.get("temperature", 0.7)
        
        # 根据策略处理查询
        if strategy_name == "direct":
            return self._execute_direct_strategy(query, agent_id, model, temperature)
        elif strategy_name == "memory_enhanced":
            return self._execute_memory_enhanced_strategy(query, context, agent_id, model, temperature)
        elif strategy_name in self.strategy_instances:
            # 使用高级策略处理
            return self._execute_advanced_strategy(query, context, agent_id, strategy_name, model, temperature)
        else:
            # 默认使用直接策略
            return self._execute_direct_strategy(query, agent_id, model, temperature)
    
    def _execute_direct_strategy(self, query: str, agent_id: str, model: str, temperature: float) -> Dict[str, Any]:
        """执行直接路由策略

        Args:
            query: 用户查询
            agent_id: Agent ID
            model: 使用的模型
            temperature: 温度参数

        Returns:
            处理结果
        """
        agent_desc = self.agent_profiles.get(agent_id, {}).get("description", "通用助手")
        
        prompt = f"""
        你是一个{agent_desc}。请回答以下问题：

        {query}
        """
        
        response = call_model(prompt, model=model, temperature=temperature)
        
        return {
            "agent_id": agent_id,
            "response": response,
            "strategy": "direct",
            "success": True
        }
    
    def _execute_memory_enhanced_strategy(self, query: str, context: Optional[List[Dict[str, str]]], 
                                        agent_id: str, model: str, temperature: float) -> Dict[str, Any]:
        """执行记忆增强路由策略

        Args:
            query: 用户查询
            context: 对话上下文
            agent_id: Agent ID
            model: 使用的模型
            temperature: 温度参数

        Returns:
            处理结果
        """
        agent_desc = self.agent_profiles.get(agent_id, {}).get("description", "通用助手")
        
        # 构建上下文提示
        context_prompt = ""
        if context:
            context_prompt = "对话历史：\n"
            for i, message in enumerate(context[-5:]):  # 只使用最近5条消息
                role = message.get("role", "user")
                content = message.get("content", "")
                context_prompt += f"{role}: {content}\n"
        
        prompt = f"""
        你是一个{agent_desc}。

        {context_prompt}

        请回答用户的最新问题：
        {query}
        """
        
        response = call_model(prompt, model=model, temperature=temperature)
        
        return {
            "agent_id": agent_id,
            "response": response,
            "strategy": "memory_enhanced",
            "success": True
        }
    
    def _execute_advanced_strategy(self, query: str, context: Optional[List[Dict[str, str]]], 
                                agent_id: str, strategy_name: str, model: str, temperature: float) -> Dict[str, Any]:
        """执行高级路由策略

        Args:
            query: 用户查询
            context: 对话上下文
            agent_id: Agent ID
            strategy_name: 策略名称
            model: 使用的模型
            temperature: 温度参数

        Returns:
            处理结果
        """
        agent_desc = self.agent_profiles.get(agent_id, {}).get("description", "通用助手")
        strategy = self.strategy_instances.get(strategy_name)
        
        if not strategy:
            # 如果策略不可用，回退到直接策略
            return self._execute_direct_strategy(query, agent_id, model, temperature)
        
        # 对于多Agent协作策略，需要特殊处理
        if strategy_name == "multi_agent":
            return self._execute_multi_agent_strategy(query, context, model, temperature)
        
        # 构建上下文提示
        context_prompt = ""
        if context:
            context_prompt = "对话历史：\n"
            for i, message in enumerate(context[-5:]):  # 只使用最近5条消息
                role = message.get("role", "user")
                content = message.get("content", "")
                context_prompt += f"{role}: {content}\n"
        
        prompt = f"""
        你是一个{agent_desc}。

        {context_prompt}

        请回答用户的最新问题：
        {query}
        """
        
        response = call_model(prompt, model=model, temperature=temperature)
        
        # 更新策略性能记录（如果策略支持反馈）
        if hasattr(strategy, "update_history") and callable(getattr(strategy, "update_history")):
            strategy.update_history(query, agent_id, True)  # 假设处理成功
        
        return {
            "agent_id": agent_id,
            "response": response,
            "strategy": strategy_name,
            "success": True
        }
    
    def _execute_multi_agent_strategy(self, query: str, context: Optional[List[Dict[str, str]]], 
                                    model: str, temperature: float) -> Dict[str, Any]:
        """执行多Agent协作策略

        Args:
            query: 用户查询
            context: 对话上下文
            model: 默认模型
            temperature: 默认温度参数

        Returns:
            处理结果
        """
        strategy = self.strategy_instances.get("multi_agent")
        
        if not strategy:
            # 如果策略不可用，回退到直接策略
            return self._execute_direct_strategy(query, self.fallback_agent, model, temperature)
        
        # 分解任务
        sub_tasks = strategy.decompose_task(query)
        
        # 为子任务分配Agent
        task_agent_pairs = strategy.select_agents_for_subtasks(sub_tasks, self.agent_profiles)
        
        # 处理每个子任务
        results = []
        
        for task, agent_id in task_agent_pairs:
            # 获取Agent配置
            agent_config = self.agent_profiles.get(agent_id, {})
            agent_model = agent_config.get("model", model)
            agent_temp = agent_config.get("temperature", temperature)
            agent_desc = agent_config.get("description", "通用助手")
            
            # 构建提示
            sub_query = task["sub_query"]
            prompt = f"""
            你是一个{agent_desc}。

            请回答以下问题：
            {sub_query}
            """
            
            # 调用模型
            response = call_model(prompt, model=agent_model, temperature=agent_temp)
            
            # 记录结果
            results.append({
                "task": task,
                "agent_id": agent_id,
                "response": response
            })
        
        # 聚合结果
        aggregated_result = strategy.aggregate_results(results)
        
        return aggregated_result
    
    def provide_feedback(self, query: str, agent_id: str, strategy_name: str, success: bool, feedback: Optional[int] = None):
        """为路由决策提供反馈

        Args:
            query: 用户查询
            agent_id: 使用的Agent ID
            strategy_name: 使用的策略名称
            success: 处理是否成功
            feedback: 用户反馈评分（1-5）
        """
        strategy = self.strategy_instances.get(strategy_name)
        
        if not strategy:
            return
        
        # 更新策略性能记录
        if hasattr(strategy, "update_history") and callable(getattr(strategy, "update_history")):
            strategy.update_history(query, agent_id, success, feedback)
        
        # 对于自适应策略，还需要更新策略选择的性能记录
        adaptive_strategy = self.strategy_instances.get("adaptive")
        if adaptive_strategy and hasattr(adaptive_strategy, "feedback") and callable(getattr(adaptive_strategy, "feedback")):
            adaptive_strategy.feedback(strategy_name, success)