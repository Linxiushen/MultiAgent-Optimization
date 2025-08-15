#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多Agent智能分流 + 长程记忆优化系统演示脚本

这个脚本提供了系统核心功能的快速演示，包括：
1. 智能路由功能
2. 记忆管理与优化
3. 多Agent响应聚合与冲突解决
4. 分布式协调

使用方法：python quick_demo.py
"""

import os
import sys
import json
import time
from datetime import datetime
import random

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心组件
from core.router import Router, EnhancedRouter
from core.aggregator import Aggregator, EnhancedAggregator
from memory.memory_manager import MemoryManager
from distributed.coordinator import Coordinator
from recovery.recovery_manager import RecoveryManager

# 模拟LLM调用
def simulate_llm_call(prompt, agent_type):
    """模拟对LLM的调用"""
    responses = {
        "tech": [
            "Python是一种解释型高级编程语言，以其简洁的语法和强大的库而闻名。",
            "Python的主要特点包括动态类型系统、自动内存管理和丰富的标准库。",
            "在AI和数据科学领域，Python因其易用性和强大的生态系统而成为首选语言。"
        ],
        "creative": [
            "想象一个未来的世界，AI与人类共同创造艺术和音乐，开启创新的新纪元。",
            "创意思维需要打破常规，尝试从不同角度看待问题，寻找独特的解决方案。",
            "艺术创作是表达内心世界的方式，通过色彩、形状和声音传递情感和思想。"
        ],
        "memory": [
            "根据您的历史交互，您对Python编程和AI技术特别感兴趣。",
            "您上次询问了关于机器学习模型优化的问题，我们讨论了梯度下降算法。",
            "您之前提到您正在开发一个自然语言处理项目，需要改进文本分类准确率。"
        ]
    }
    
    # 模拟延迟
    time.sleep(random.uniform(0.5, 1.5))
    
    return random.choice(responses[agent_type])

# 演示配置
class DemoConfig:
    def __init__(self):
        self.show_memory = True
        self.show_routing = True
        self.show_aggregation = True
        self.show_conflict_resolution = True
        self.show_distributed = True

# 控制台颜色
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# 演示类
class SystemDemo:
    def __init__(self, config=None):
        self.config = config or DemoConfig()
        
        # 初始化组件
        print(f"{Colors.HEADER}{Colors.BOLD}正在初始化系统组件...{Colors.ENDC}")
        self.memory_manager = MemoryManager()
        self.router = EnhancedRouter(self.memory_manager)
        self.aggregator = EnhancedAggregator(self.memory_manager)
        self.coordinator = Coordinator()
        self.recovery_manager = RecoveryManager()
        
        # 注册Agent
        self.coordinator.register_agent("tech_agent", {"type": "tech", "status": "active"})
        self.coordinator.register_agent("creative_agent", {"type": "creative", "status": "active"})
        self.coordinator.register_agent("memory_agent", {"type": "memory", "status": "active"})
        
        print(f"{Colors.GREEN}系统初始化完成!{Colors.ENDC}\n")
    
    def demonstrate_routing(self, query):
        """演示路由功能"""
        if not self.config.show_routing:
            return None
            
        print(f"{Colors.BLUE}{Colors.BOLD}===== 智能路由演示 ====={Colors.ENDC}")
        print(f"用户查询: {query}")
        
        # 路由决策
        print("\n正在进行路由决策...")
        time.sleep(1)
        
        route_result = self.router.route(query)
        selected_agents = route_result["selected_agents"]
        routing_explanation = route_result["explanation"]
        
        print(f"\n{Colors.GREEN}路由决策完成!{Colors.ENDC}")
        print(f"选择的Agent: {', '.join([agent['type'] for agent in selected_agents])}")
        print(f"路由解释: {routing_explanation}")
        
        return selected_agents
    
    def demonstrate_memory(self, query):
        """演示记忆功能"""
        if not self.config.show_memory:
            return
            
        print(f"\n{Colors.BLUE}{Colors.BOLD}===== 记忆管理演示 ====={Colors.ENDC}")
        
        # 检索相关记忆
        print("正在检索相关记忆...")
        time.sleep(1)
        
        memories = self.memory_manager.retrieve_relevant(query, limit=2)
        
        if memories:
            print(f"\n{Colors.GREEN}找到{len(memories)}条相关记忆:{Colors.ENDC}")
            for i, memory in enumerate(memories):
                print(f"  {i+1}. {memory.content} (重要性: {memory.importance:.2f})")
        else:
            print(f"\n{Colors.YELLOW}没有找到相关记忆{Colors.ENDC}")
        
        # 存储新记忆
        print("\n正在存储新记忆...")
        time.sleep(0.5)
        
        memory_id = self.memory_manager.store(
            content=f"用户询问: {query}",
            metadata={"timestamp": datetime.now().isoformat(), "type": "user_query"}
        )
        
        print(f"{Colors.GREEN}新记忆已存储! ID: {memory_id}{Colors.ENDC}")
    
    def demonstrate_aggregation(self, query, selected_agents):
        """演示聚合功能"""
        if not self.config.show_aggregation or not selected_agents:
            return
            
        print(f"\n{Colors.BLUE}{Colors.BOLD}===== 响应聚合演示 ====={Colors.ENDC}")
        
        # 获取各Agent的响应
        print("正在收集各Agent的响应...")
        agent_responses = []
        
        for agent in selected_agents:
            agent_type = agent["type"]
            print(f"  请求 {agent_type} Agent...")
            time.sleep(random.uniform(0.5, 1.0))
            
            response = simulate_llm_call(query, agent_type)
            confidence = random.uniform(0.7, 0.95)
            
            agent_responses.append({
                "agent_id": f"{agent_type}_agent",
                "agent_type": agent_type,
                "response": response,
                "confidence": confidence,
                "metadata": {"timestamp": datetime.now().isoformat()}
            })
            
            print(f"  {Colors.GREEN}收到 {agent_type} Agent响应: {Colors.ENDC}\"{response[:50]}...\"")
        
        # 聚合响应
        print("\n正在聚合响应...")
        time.sleep(1)
        
        if self.config.show_conflict_resolution:
            print("检测并解决潜在冲突...")
            time.sleep(0.5)
        
        final_response = self.aggregator.aggregate(query, agent_responses)
        
        print(f"\n{Colors.GREEN}最终聚合响应:{Colors.ENDC}")
        print(f"{Colors.BOLD}{final_response}{Colors.ENDC}")
    
    def demonstrate_distributed(self):
        """演示分布式协调功能"""
        if not self.config.show_distributed:
            return
            
        print(f"\n{Colors.BLUE}{Colors.BOLD}===== 分布式协调演示 ====={Colors.ENDC}")
        
        # 显示当前活跃Agent
        active_agents = self.coordinator.list_active_agents()
        print(f"当前活跃Agent: {len(active_agents)}个")
        for agent_id, info in active_agents.items():
            print(f"  - {agent_id}: {info['type']} (状态: {info['status']})")
        
        # 模拟Agent故障
        print("\n模拟Agent故障场景...")
        time.sleep(1)
        
        failed_agent = "tech_agent"
        print(f"  {Colors.RED}Agent '{failed_agent}' 发生故障!{Colors.ENDC}")
        self.coordinator.update_agent_status(failed_agent, "failed")
        
        # 故障恢复
        print("  触发故障恢复机制...")
        time.sleep(1)
        
        recovery_success = self.recovery_manager.recover_agent(failed_agent)
        if recovery_success:
            print(f"  {Colors.GREEN}Agent '{failed_agent}' 已成功恢复!{Colors.ENDC}")
            self.coordinator.update_agent_status(failed_agent, "active")
        else:
            print(f"  {Colors.RED}Agent '{failed_agent}' 恢复失败!{Colors.ENDC}")
        
        # 显示更新后的Agent状态
        active_agents = self.coordinator.list_active_agents()
        print(f"\n更新后的活跃Agent: {len(active_agents)}个")
        for agent_id, info in active_agents.items():
            status_color = Colors.GREEN if info['status'] == 'active' else Colors.RED
            print(f"  - {agent_id}: {info['type']} (状态: {status_color}{info['status']}{Colors.ENDC})")
    
    def run_demo(self):
        """运行完整演示"""
        print(f"{Colors.HEADER}{Colors.BOLD}欢迎使用多Agent智能分流 + 长程记忆优化系统演示!{Colors.ENDC}\n")
        
        # 示例查询
        queries = [
            "Python有哪些主要特点?",
            "给我一些创意思维的建议",
            "我之前问过什么问题?"
        ]
        
        for i, query in enumerate(queries):
            if i > 0:
                print("\n" + "-"*50 + "\n")
                
            print(f"{Colors.BOLD}演示 {i+1}/{len(queries)}: \"{query}\"{Colors.ENDC}\n")
            
            # 演示记忆功能
            self.demonstrate_memory(query)
            
            # 演示路由功能
            selected_agents = self.demonstrate_routing(query)
            
            # 演示聚合功能
            self.demonstrate_aggregation(query, selected_agents)
            
            # 只在最后一个查询演示分布式协调
            if i == len(queries) - 1:
                self.demonstrate_distributed()
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}演示完成!{Colors.ENDC}")

if __name__ == "__main__":
    # 创建并运行演示
    demo = SystemDemo()
    demo.run_demo()