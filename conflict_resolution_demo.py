import os
import sys
import json
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.enhanced_aggregator import EnhancedAggregator
from core.conflict_resolver import ConflictResolver

# 模拟LLM调用
def mock_llm_call(prompt, model="gpt-3.5-turbo", temperature=0.7):
    """模拟LLM调用，返回预定义的响应"""
    print(f"\n[模拟LLM调用] 模型: {model}, 温度: {temperature}")
    print(f"提示: {prompt[:100]}...")
    
    # 根据提示返回不同的响应
    if "Python" in prompt:
        return {
            "content": "Python是一种解释型高级编程语言，以简洁、易读的语法著称。",
            "confidence": 0.85
        }
    elif "人工智能" in prompt:
        return {
            "content": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
            "confidence": 0.8
        }
    else:
        return {
            "content": "我不确定如何回答这个问题。",
            "confidence": 0.5
        }

# 模拟Agent响应
def get_mock_responses(query):
    """获取模拟的Agent响应"""
    if "Python" in query:
        return [
            {
                "agent_id": "technical_agent",
                "content": "Python是一种解释型语言，执行速度较慢。它的优点是简单易学，适合初学者。Python的主要应用领域包括数据分析、人工智能和Web开发。",
                "confidence": 0.8
            },
            {
                "agent_id": "creative_agent",
                "content": "Python是一种编译型语言，执行速度很快。它的优点是简单易学，适合初学者。Python主要用于游戏开发和系统编程。",
                "confidence": 0.6
            },
            {
                "agent_id": "memory_agent",
                "content": "根据历史记录，Python是一种解释型语言，执行速度中等。它适合初学者学习，主要用于数据分析和Web开发。我建议你先学习Python基础语法，然后专注于特定领域的应用。",
                "confidence": 0.7
            }
        ]
    elif "人工智能" in query:
        return [
            {
                "agent_id": "technical_agent",
                "content": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。它包括机器学习、深度学习、自然语言处理等子领域。",
                "confidence": 0.85
            },
            {
                "agent_id": "creative_agent",
                "content": "人工智能是模拟人类思维过程的技术，它可以创造性地解决问题，产生艺术作品，甚至可以超越人类的能力。我认为AI将彻底改变我们的生活方式。",
                "confidence": 0.7
            },
            {
                "agent_id": "memory_agent",
                "content": "根据您之前的问题，您似乎对AI的技术细节很感兴趣。人工智能是让机器模拟人类智能的科学，包括学习、推理和自我修正。目前最热门的AI应用是大型语言模型和生成式AI。",
                "confidence": 0.75
            }
        ]
    else:
        return [
            {
                "agent_id": "default_agent",
                "content": "我不确定如何回答这个问题。请提供更多信息。",
                "confidence": 0.5
            }
        ]

# 演示函数
def demonstrate_conflict_resolution():
    """演示冲突解决功能"""
    print("\n===== 增强型聚合器冲突解决演示 =====\n")
    
    # 初始化增强型聚合器
    aggregator = EnhancedAggregator()
    
    # 确保冲突解决已启用
    aggregator.config["conflict_resolution"]["enabled"] = True
    
    # 显示当前配置
    print("当前聚合器配置:")
    print(json.dumps(aggregator.config, indent=2, ensure_ascii=False))
    
    # 演示1: Python语言特性冲突
    print("\n\n===== 演示1: Python语言特性冲突 =====\n")
    query1 = "Python是什么类型的语言，它的执行速度如何？"
    print(f"用户查询: {query1}")
    
    # 获取模拟响应
    responses1 = get_mock_responses(query1)
    print("\n各Agent的响应:")
    for resp in responses1:
        print(f"\n{resp['agent_id']} (置信度: {resp['confidence']}):\n{resp['content']}")
    
    # 聚合响应
    print("\n\n聚合器处理中...")
    final_response1 = aggregator.aggregate_responses(responses1, query1)
    
    print("\n最终响应:")
    print(final_response1)
    
    # 演示2: 人工智能定义和应用冲突
    print("\n\n===== 演示2: 人工智能定义和应用 =====\n")
    query2 = "什么是人工智能，它有哪些主要应用？"
    print(f"用户查询: {query2}")
    
    # 获取模拟响应
    responses2 = get_mock_responses(query2)
    print("\n各Agent的响应:")
    for resp in responses2:
        print(f"\n{resp['agent_id']} (置信度: {resp['confidence']}):\n{resp['content']}")
    
    # 聚合响应
    print("\n\n聚合器处理中...")
    final_response2 = aggregator.aggregate_responses(responses2, query2)
    
    print("\n最终响应:")
    print(final_response2)
    
    # 演示3: 禁用冲突解决
    print("\n\n===== 演示3: 禁用冲突解决 =====\n")
    
    # 禁用冲突解决
    aggregator.config["conflict_resolution"]["enabled"] = False
    print("已禁用冲突解决")
    
    # 使用相同的查询和响应
    print(f"\n用户查询: {query1}")
    
    # 聚合响应
    print("\n聚合器处理中...")
    final_response3 = aggregator.aggregate_responses(responses1, query1)
    
    print("\n最终响应 (无冲突解决):")
    print(final_response3)
    
    # 演示4: 不同冲突解决策略
    print("\n\n===== 演示4: 不同冲突解决策略 =====\n")
    
    # 启用冲突解决
    aggregator.config["conflict_resolution"]["enabled"] = True
    
    strategies = ["majority_voting", "confidence_weighted", "semantic_analysis", "source_reliability"]
    
    for strategy in strategies:
        print(f"\n--- 策略: {strategy} ---")
        
        # 设置策略
        aggregator.config["conflict_resolution"]["strategy"] = strategy
        
        # 聚合响应
        print("聚合器处理中...")
        strategy_response = aggregator.aggregate_responses(responses1, query1)
        
        print(f"\n使用 {strategy} 策略的最终响应:")
        print(strategy_response)
    
    # 演示5: 用户反馈
    print("\n\n===== 演示5: 用户反馈 =====\n")
    
    # 模拟用户反馈
    feedback = {
        "rating": 4,
        "comment": "回答很好，但关于Python执行速度的描述有些不准确",
        "conflict_detected": True,
        "conflict_resolved": True
    }
    
    print("用户反馈:")
    print(json.dumps(feedback, indent=2, ensure_ascii=False))
    
    # 提供反馈
    print("\n处理反馈...")
    aggregator.provide_feedback(query1, final_response1, feedback)
    
    print("\n反馈已处理并存储到记忆中")

# 主函数
def main():
    """主函数"""
    try:
        demonstrate_conflict_resolution()
    except Exception as e:
        print(f"\n演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()