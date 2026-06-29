"""
聚合器演示脚本
展示如何使用聚合器融合多个Agent的响应
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.aggregator import aggregator

def main():
    print("=== 聚合器演示 ===")
    
    # 模拟多个Agent的响应
    agent_responses = [
        {
            "agent_id": "technical_agent",
            "content": "根据技术分析，这个问题可以通过优化算法复杂度来解决。建议使用动态规划方法，时间复杂度可以降低到O(n^2)。",
            "confidence": 0.85
        },
        {
            "agent_id": "creative_agent",
            "content": "从创意角度，这个问题可以看作是一个寻宝游戏。我们可以设计一个有趣的故事情节，让用户在解决问题的过程中获得成就感。",
            "confidence": 0.75
        },
        {
            "agent_id": "default_agent",
            "content": "这个问题需要综合考虑技术可行性和用户体验。建议先进行原型设计，然后收集用户反馈进行迭代优化。",
            "confidence": 0.65
        }
    ]
    
    print("\nAgent响应:")
    for i, resp in enumerate(agent_responses, 1):
        print(f"{i}. {resp['agent_id']}: {resp['content']} (置信度: {resp['confidence']})")
    
    # 使用聚合器融合响应
    print("\n=== 融合响应 ===")
    fused_response = aggregator.fuse_responses(agent_responses)
    print(f"融合后的内容: {fused_response['content']}")
    print(f"融合置信度: {fused_response['confidence']:.2f}")
    if 'sources' in fused_response:
        print(f"来源: {', '.join(fused_response['sources'])}")
    
    # 生成最终响应
    query = "如何解决这个复杂的技术问题？"
    print("\n=== 最终响应 ===")
    final_response = aggregator.generate_final_response(fused_response, query)
    print(final_response)

if __name__ == "__main__":
    main()