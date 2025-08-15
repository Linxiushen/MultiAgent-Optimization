from adapters.llm import call_model, lite_llm_infer
import json
from typing import Dict, List, Any, Optional

class Router:
    def __init__(self, config_path: str = "configs/router_config.json"):
        """初始化路由系统

        Args:
            config_path: 路由配置文件路径
        """
        self.config = self._load_config(config_path)
        self.agent_profiles = self.config.get("agent_profiles", {})
        self.routing_strategies = self.config.get("routing_strategies", {})
        self.default_strategy = self.config.get("default_strategy", "direct")

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
                "default_strategy": "direct"
            }
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {config_path} 格式无效，使用默认配置")
            return {
                "agent_profiles": {},
                "routing_strategies": {},
                "default_strategy": "direct"
            }

    def determine_agent(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """根据查询和上下文确定最适合的Agent

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            选定的Agent ID
        """
        # 简单实现：使用LLM来决定路由
        if not self.agent_profiles:
            return "default_agent"

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
            return "default_agent"

    def route_query(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """路由查询到合适的Agent

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            路由结果，包含选定的Agent和处理策略
        """
        agent_id = self.determine_agent(query, context)
        strategy = self.routing_strategies.get(agent_id, self.default_strategy)

        return {
            "agent_id": agent_id,
            "strategy": strategy,
            "timestamp": "",  # 实际应用中添加时间戳
            "query": query
        }

    def execute_route(self, route_result: Dict[str, Any], context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """执行路由结果

        Args:
            route_result: 路由结果
            context: 对话上下文

        Returns:
            执行结果
        """
        agent_id = route_result["agent_id"]
        query = route_result["query"]
        strategy = route_result["strategy"]

        # 根据不同的策略执行路由
        if strategy == "direct":
            # 直接调用对应Agent的处理逻辑
            agent_config = self.agent_profiles.get(agent_id, {})
            model = agent_config.get("model", "gpt-3.5-turbo")
            temperature = agent_config.get("temperature", 0.7)

            # 构建完整提示
            if context:
                messages = context + [{"role": "user", "content": query}]
            else:
                messages = [{"role": "user", "content": query}]

            # 调用LLM
            response = call_model(
                model=model,
                messages=messages,
                temperature=temperature
            )

            return {
                "agent_id": agent_id,
                "response": response,
                "model_used": model,
                "success": True
            }
        elif strategy == "memory_enhanced":
            # 记忆增强策略
            try:
                from memory import MemoryManager
                memory_manager = MemoryManager()

                # 检索相关记忆
                relevant_memories = memory_manager.retrieve_memory(query)
                memory_content = "\n".join([mem["content"] for mem in relevant_memories])

                # 构建完整提示
                agent_config = self.agent_profiles.get(agent_id, {})
                model = agent_config.get("model", "gpt-3.5-turbo")
                temperature = agent_config.get("temperature", 0.7)

                if context:
                    messages = context + [{
                        "role": "system",
                        "content": f"以下是相关记忆信息，可以帮助你回答用户问题：\n{memory_content}"
                    }, {
                        "role": "user",
                        "content": query
                    }]
                else:
                    messages = [{
                        "role": "system",
                        "content": f"以下是相关记忆信息，可以帮助你回答用户问题：\n{memory_content}"
                    }, {
                        "role": "user",
                        "content": query
                    }]

                # 调用LLM
                response = call_model(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )

                # 存储新记忆
                memory_manager.store_memory(f"用户问题: {query}\nAI回答: {response}", {
                    "agent_id": agent_id,
                    "query_type": "memory_enhanced"
                })

                return {
                    "agent_id": agent_id,
                    "response": response,
                    "model_used": model,
                    "memory_count": len(relevant_memories),
                    "success": True
                }
            except Exception as e:
                print(f"记忆增强策略执行失败: {str(e)}")
                return {
                    "agent_id": agent_id,
                    "response": f"记忆增强策略执行失败: {str(e)}",
                    "success": False
                }
        else:
            # 未知策略
            return {
                "agent_id": agent_id,
                "response": f"未知策略: {strategy}",
                "success": False
            }