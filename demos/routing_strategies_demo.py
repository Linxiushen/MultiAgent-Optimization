import os
import sys
import json
import time
from typing import Dict, List, Any, Optional

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.enhanced_router import EnhancedRouter
from core.routing_strategies import StrategyFactory
from adapters.llm import call_model

# 模拟LLM调用，避免实际API调用
def mock_call_model(prompt, model="gpt-3.5-turbo", temperature=0.7):
    """模拟LLM调用"""
    print(f"\n[模拟调用 {model}, 温度={temperature}]")
    print(f"提示词: {prompt[:100]}...")
    
    # 根据提示词内容生成模拟响应
    if "技术" in prompt or "代码" in prompt or "编程" in prompt:
        return "这是技术Agent的响应。我可以帮助你解决技术问题和编程任务。"
    elif "创意" in prompt or "设计" in prompt or "写作" in prompt:
        return "这是创意Agent的响应。我可以帮助你进行创意设计和文案写作。"
    elif "记忆" in prompt or "之前" in prompt or "历史" in prompt:
        return "这是记忆Agent的响应。我记得你之前问过的问题，可以基于历史上下文回答。"
    else:
        return "这是默认Agent的响应。我是一个通用助手，可以回答各种问题。"

# 替换实际的LLM调用函数
from adapters import llm
llm.call_model = mock_call_model
llm.lite_llm_infer = lambda prompt: "technical_agent" if "代码" in prompt or "编程" in prompt else "creative_agent" if "创意" in prompt or "设计" in prompt else "default_agent"

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_section(title):
    """打印小节标题"""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80)

def print_result(result):
    """打印路由结果"""
    print(f"\n选择的Agent: {result['agent_id']}")
    print(f"使用的策略: {result['strategy']}")
    print(f"响应: {result['response']}")
    print(f"成功: {result['success']}")

def demo_direct_strategy(router):
    """演示直接路由策略"""
    print_section("直接路由策略演示")
    
    query = "什么是人工智能？"
    print(f"用户查询: {query}")
    
    result = router.execute_route(query)
    print_result(result)

def demo_memory_enhanced_strategy(router):
    """演示记忆增强路由策略"""
    print_section("记忆增强路由策略演示")
    
    # 创建对话上下文
    context = [
        {"role": "user", "content": "Python和Java有什么区别？"},
        {"role": "assistant", "content": "Python是一种解释型语言，而Java是编译型语言..."},
        {"role": "user", "content": "哪个更适合初学者？"},
        {"role": "assistant", "content": "对于初学者来说，Python通常被认为更容易上手..."},
    ]
    
    query = "你能给我一些学习编程的建议吗？"
    print(f"用户查询: {query}")
    print(f"上下文: {len(context)}条对话历史")
    
    # 修改Agent策略映射，使用记忆增强策略
    original_mapping = router.agent_strategy_mapping.copy()
    router.agent_strategy_mapping["technical_agent"] = "memory_enhanced"
    
    result = router.execute_route(query, context)
    print_result(result)
    
    # 恢复原始映射
    router.agent_strategy_mapping = original_mapping

def demo_semantic_strategy(router):
    """演示语义路由策略"""
    print_section("语义路由策略演示")
    
    queries = [
        "如何编写Python代码来处理JSON数据？",
        "帮我设计一个创意广告文案",
        "什么是机器学习？"
    ]
    
    # 设置默认策略为语义路由
    original_strategy = router.default_strategy_name
    router.default_strategy_name = "semantic"
    
    for query in queries:
        print(f"\n用户查询: {query}")
        result = router.execute_route(query)
        print_result(result)
    
    # 恢复原始默认策略
    router.default_strategy_name = original_strategy

def demo_history_based_strategy(router):
    """演示基于历史交互的路由策略"""
    print_section("基于历史交互的路由策略演示")
    
    # 确保历史策略已初始化
    if "history" not in router.strategy_instances:
        router.strategy_instances["history"] = StrategyFactory.create_strategy("history")
    
    history_strategy = router.strategy_instances["history"]
    
    # 添加一些历史交互记录
    history_strategy.update_history("如何编写Python代码？", "technical_agent", True, 5)
    history_strategy.update_history("Python和Java有什么区别？", "technical_agent", True, 4)
    history_strategy.update_history("帮我写一个故事", "creative_agent", True, 5)
    history_strategy.update_history("设计一个广告文案", "creative_agent", True, 3)
    
    # 设置默认策略为历史交互路由
    original_strategy = router.default_strategy_name
    router.default_strategy_name = "history"
    
    queries = [
        "如何学习Java编程？",
        "帮我写一首诗",
        "什么是人工智能？"
    ]
    
    for query in queries:
        print(f"\n用户查询: {query}")
        result = router.execute_route(query)
        print_result(result)
        
        # 提供反馈
        router.provide_feedback(query, result["agent_id"], "history", True, 4)
    
    # 恢复原始默认策略
    router.default_strategy_name = original_strategy

def demo_adaptive_strategy(router):
    """演示自适应路由策略"""
    print_section("自适应路由策略演示")
    
    # 确保自适应策略已初始化
    if "adaptive" not in router.strategy_instances:
        router.strategy_instances["adaptive"] = StrategyFactory.create_strategy("adaptive")
    
    adaptive_strategy = router.strategy_instances["adaptive"]
    
    # 添加一些策略性能记录
    adaptive_strategy.update_strategy_performance("semantic", True)
    adaptive_strategy.update_strategy_performance("semantic", True)
    adaptive_strategy.update_strategy_performance("history", False)
    
    # 设置默认策略为自适应路由
    original_strategy = router.default_strategy_name
    router.default_strategy_name = "adaptive"
    
    queries = [
        "如何使用Python处理大数据？",
        "设计一个网站首页",
        "解释一下量子计算"
    ]
    
    for query in queries:
        print(f"\n用户查询: {query}")
        
        # 打印策略选择
        selected_strategy = adaptive_strategy.select_strategy(query)
        print(f"自适应策略选择: {selected_strategy}")
        
        result = router.execute_route(query)
        print_result(result)
        
        # 提供反馈
        success = True  # 假设所有响应都成功
        adaptive_strategy.feedback(selected_strategy, success)
    
    # 恢复原始默认策略
    router.default_strategy_name = original_strategy

def demo_multi_agent_strategy(router):
    """演示多Agent协作路由策略"""
    print_section("多Agent协作路由策略演示")
    
    # 确保多Agent协作策略已初始化
    if "multi_agent" not in router.strategy_instances:
        router.strategy_instances["multi_agent"] = StrategyFactory.create_strategy("multi_agent")
    
    # 设置默认策略为多Agent协作
    original_strategy = router.default_strategy_name
    router.default_strategy_name = "multi_agent"
    
    query = "请帮我编写一个Python函数来计算斐波那契数列。然后解释一下这个算法的时间复杂度。最后，给我一些优化建议。"
    print(f"用户查询: {query}")
    
    # 打印任务分解
    multi_agent_strategy = router.strategy_instances["multi_agent"]
    sub_tasks = multi_agent_strategy.decompose_task(query)
    print(f"\n任务分解为{len(sub_tasks)}个子任务:")
    for i, task in enumerate(sub_tasks):
        print(f"  {i+1}. {task['sub_query']}")
    
    result = router.execute_route(query)
    print_result(result)
    
    # 恢复原始默认策略
    router.default_strategy_name = original_strategy

def main():
    """主函数"""
    print_header("智能路由策略演示")
    
    # 加载配置
    config_path = "configs/router_config.json"
    print(f"加载配置: {config_path}")
    
    # 初始化路由器
    router = EnhancedRouter(config_path)
    print(f"路由器初始化完成，默认策略: {router.default_strategy_name}")
    print(f"已加载策略: {', '.join(router.strategy_instances.keys())}")
    
    # 演示各种路由策略
    demo_direct_strategy(router)
    demo_memory_enhanced_strategy(router)
    demo_semantic_strategy(router)
    demo_history_based_strategy(router)
    demo_adaptive_strategy(router)
    demo_multi_agent_strategy(router)
    
    print_header("演示结束")

if __name__ == "__main__":
    main()