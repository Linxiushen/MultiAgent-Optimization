#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统演示脚本
展示多Agent智能分流与长程记忆优化系统的核心功能
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.router import Router
from memory.memory_manager import MemoryManager
from distributed.coordinator import DistributedCoordinator
from core.aggregator import Aggregator


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title):
    """打印标题"""
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)


def print_step(step, description):
    """打印步骤"""
    print(f"\n[{step}] {description}")
    print("-" * 40)


def demo_router():
    """演示路由功能"""
    print_header("路由系统演示")
    
    # 创建路由器实例
    router = Router()
    
    # 测试查询
    queries = [
        "请解释什么是机器学习？",
        "帮我写一个Python快速排序算法",
        "创作一首关于春天的诗",
        "计算2的10次方等于多少？"
    ]
    
    for i, query in enumerate(queries, 1):
        print_step(i, f"路由查询: {query}")
        
        # 路由查询
        route_result = router.route_query(query)
        print(f"  分配的Agent: {route_result['agent_id']}")
        print(f"  使用策略: {route_result['strategy']}")
        print(f"  时间戳: {route_result['timestamp']}")
        
        time.sleep(1)  # 暂停1秒以便观察
    
    input("\n按回车键继续...")


def demo_memory():
    """演示记忆功能"""
    print_header("记忆系统演示")
    
    # 创建记忆管理器实例
    memory_manager = MemoryManager()
    
    # 存储一些记忆
    memories = [
        "用户询问了机器学习的定义",
        "用户请求编写快速排序算法",
        "用户要求创作关于春天的诗歌",
        "用户需要计算2的10次方"
    ]
    
    print_step(1, "存储记忆")
    memory_ids = []
    for i, content in enumerate(memories, 1):
        memory_id = memory_manager.store_memory(content, {"source": "demo", "importance": "medium"})
        memory_ids.append(memory_id)
        print(f"  已存储记忆 {i}: {content} (ID: {memory_id})")
        time.sleep(0.5)
    
    # 检索记忆
    print_step(2, "检索相关记忆")
    query = "用户需要什么帮助？"
    retrieved_memories = memory_manager.retrieve_memory(query, top_k=3)
    
    print(f"  查询: {query}")
    for i, memory in enumerate(retrieved_memories, 1):
        print(f"  相关记忆 {i}: {memory['content']}")
        print(f"    相似度分数: {memory.get('similarity_score', 'N/A'):.4f}")
    
    input("\n按回车键继续...")


def demo_coordinator():
    """演示分布式协调功能"""
    print_header("分布式协调系统演示")
    
    # 创建协调器实例
    coordinator = DistributedCoordinator()
    
    # 注册一些Agent
    agents = [
        ("agent_001", "technical", ["coding", "math"], "localhost:8001"),
        ("agent_002", "creative", ["writing", "poetry"], "localhost:8002"),
        ("agent_003", "general", ["qa", "explanation"], "localhost:8003")
    ]
    
    print_step(1, "注册Agent")
    for agent_id, agent_type, capabilities, endpoint in agents:
        success = coordinator.register_agent(agent_id, agent_type, capabilities, endpoint)
        status = "成功" if success else "失败"
        print(f"  注册Agent {agent_id} ({agent_type}): {status}")
        time.sleep(0.5)
    
    # 更新心跳
    print_step(2, "更新Agent心跳")
    for agent_id, _, _, _ in agents:
        success = coordinator.update_agent_heartbeat(agent_id)
        status = "成功" if success else "失败"
        print(f"  更新 {agent_id} 心跳: {status}")
        time.sleep(0.5)
    
    # 分配任务
    print_step(3, "分配任务")
    tasks = [
        ("coding", "编写一个Python快速排序算法"),
        ("writing", "创作一首关于春天的诗"),
        ("qa", "解释什么是机器学习？")
    ]
    
    for task_type, task_data in tasks:
        assigned_agent = coordinator.assign_task(task_type, task_data)
        print(f"  任务类型: {task_type}")
        print(f"  任务内容: {task_data}")
        print(f"  分配的Agent: {assigned_agent}")
        print()
        time.sleep(1)
    
    input("\n按回车键继续...")


def demo_aggregator():
    """演示响应聚合功能"""
    print_header("响应聚合系统演示")
    
    # 创建聚合器实例
    aggregator = Aggregator()
    
    # 模拟Agent响应
    agent_responses = [
        {
            "agent_id": "technical_agent",
            "content": "快速排序是一种高效的排序算法，采用分治法策略。它选择一个元素作为基准，将数组分为两部分，一部分小于基准，一部分大于基准，然后递归地对两部分进行排序。",
            "confidence": 0.95
        },
        {
            "agent_id": "general_agent",
            "content": "快速排序是一种排序算法，通过选择基准元素将数组分割成两个子数组，然后递归地对子数组进行排序。",
            "confidence": 0.85
        },
        {
            "agent_id": "memory_agent",
            "content": "根据历史记录，用户之前询问过算法相关问题。快速排序的平均时间复杂度为O(n log n)。",
            "confidence": 0.90
        }
    ]
    
    print_step(1, "融合多个Agent响应")
    for i, response in enumerate(agent_responses, 1):
        print(f"  Agent {i} ({response['agent_id']}):")
        print(f"    内容: {response['content']}")
        print(f"    置信度: {response['confidence']:.2f}")
        print()
    
    # 融合响应
    fused_response = aggregator.fuse_responses(agent_responses)
    
    print_step(2, "融合后的统一响应")
    print(f"  内容: {fused_response['content']}")
    print(f"  置信度: {fused_response['confidence']:.2f}")
    print(f"  来源: {', '.join(fused_response.get('sources', []))}")
    
    input("\n按回车键继续...")


def main():
    """主函数"""
    clear_screen()
    
    print_header("多Agent智能分流与长程记忆优化系统演示")
    print("\n本演示将展示系统的核心功能模块：")
    print("1. 路由系统 - 智能分配用户查询到合适的Agent")
    print("2. 记忆系统 - 存储和检索历史交互信息")
    print("3. 分布式协调 - 管理多个Agent的注册和任务分配")
    print("4. 响应聚合 - 融合多个Agent的响应生成统一答案")
    
    input("\n按回车键开始演示...")
    
    try:
        # 演示各个模块
        demo_router()
        clear_screen()
        
        demo_memory()
        clear_screen()
        
        demo_coordinator()
        clear_screen()
        
        demo_aggregator()
        clear_screen()
        
        print_header("演示完成")
        print("\n感谢观看多Agent智能分流与长程记忆优化系统的演示！")
        print("\n本系统通过以下方式提升AI交互体验：")
        print("• 智能路由：根据查询内容自动分配最合适的Agent")
        print("• 长程记忆：存储和利用历史交互信息提升响应质量")
        print("• 分布式架构：支持多个Agent协同工作")
        print("• 响应聚合：融合多个Agent的响应生成更全面的答案")
        
    except KeyboardInterrupt:
        print("\n\n演示被用户中断。")
    except Exception as e:
        print(f"\n\n演示过程中发生错误: {e}")
    finally:
        print("\n演示结束。")


if __name__ == "__main__":
    main()