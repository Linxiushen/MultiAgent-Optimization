"""
综合演示脚本
展示多Agent智能分流与长程记忆优化系统的各项功能
"""

import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保NLTK数据已下载
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
except ImportError:
    print("NLTK未安装，某些功能可能无法正常工作")

from core.router import Router
from core.aggregator import Aggregator
from memory.memory_manager import MemoryManager
from memory.memory_optimizer_enhanced import MemoryOptimizerEnhanced


def demo_router_functionality():
    """演示路由功能"""
    print("=== 路由功能演示 ===")
    router = Router()
    
    # 示例查询
    queries = [
        "什么是机器学习？",
        "帮我写一首关于春天的诗",
        "计算2的10次方等于多少",
        "解释量子力学的基本原理"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        route_result = router.route_query(query)
        print(f"路由结果: Agent={route_result['agent_id']}, 策略={route_result['strategy']}")
        
        # 执行路由
        execution_result = router.execute_route(route_result)
        if execution_result['success']:
            print(f"执行成功，响应: {execution_result['response'][:100]}...")
        else:
            print(f"执行失败: {execution_result['response']}")


def demo_memory_functionality():
    """演示记忆功能"""
    print("\n=== 记忆功能演示 ===")
    memory_manager = MemoryManager()
    
    # 存储一些记忆
    memories = [
        "用户喜欢Python编程语言",
        "用户对机器学习很感兴趣",
        "用户正在学习自然语言处理",
        "用户最喜欢的算法是决策树"
    ]
    
    print("存储记忆...")
    for i, content in enumerate(memories):
        memory_id = memory_manager.store_memory(content, {"category": "user_preference", "importance": 0.8})
        print(f"  存储记忆 {i+1}: {content} (ID: {memory_id})")
    
    # 检索记忆
    print("\n检索相关记忆...")
    query = "用户的技术兴趣是什么？"
    relevant_memories = memory_manager.retrieve_memory(query, top_k=3)
    
    for i, memory in enumerate(relevant_memories):
        print(f"  相关记忆 {i+1}: {memory['content']} (相似度: {memory.get('similarity_score', 0):.2f})")


def demo_enhanced_memory_optimizer():
    """演示增强版记忆优化器功能"""
    print("\n=== 增强版记忆优化器演示 ===")
    optimizer = MemoryOptimizerEnhanced()
    
    # 创建一些测试记忆
    test_memories = {
        'mem1': {
            'content': 'Python是一种高级编程语言，具有简洁易读的语法。',
            'timestamp': time.time() - 3600,
            'importance': 0.8,
            'access_count': 5,
            'tags': ['programming', 'python']
        },
        'mem2': {
            'content': 'Java是一种面向对象的编程语言，广泛应用于企业级开发。',
            'timestamp': time.time() - 1800,
            'importance': 0.7,
            'access_count': 3,
            'tags': ['programming', 'java']
        },
        'mem3': {
            'content': 'Python语言因其简洁性和强大的库支持而受到数据科学家的喜爱。',
            'timestamp': time.time() - 1200,
            'importance': 0.9,
            'access_count': 4,
            'tags': ['programming', 'python', 'data_science']
        }
    }
    
    # 计算重要性
    print("计算记忆重要性...")
    for mid, memory in test_memories.items():
        importance = optimizer.calculate_importance(memory)
        print(f"  记忆 {mid}: 重要性 = {importance:.2f}")
    
    # 聚类记忆
    print("\n聚类记忆...")
    clusters = optimizer.cluster_memories(test_memories)
    for cluster_id, mem_ids in clusters.items():
        print(f"  聚类 {cluster_id}: {mem_ids}")
    
    # 生成摘要
    print("\n生成记忆摘要...")
    long_text = "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出预测或决策。机器学习算法可以分为监督学习、无监督学习和强化学习等几大类。每种类型都有其特定的应用场景和优势。"
    summary = optimizer.generate_summary(long_text)
    print(f"  原文: {long_text}")
    print(f"  摘要: {summary}")


def demo_aggregator_functionality():
    """演示聚合器功能"""
    print("\n=== 聚合器功能演示 ===")
    aggregator = Aggregator()
    
    # 模拟多个Agent的响应
    agent_responses = [
        {
            "agent_id": "tech_agent",
            "content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
            "confidence": 0.9
        },
        {
            "agent_id": "academic_agent",
            "content": "机器学习是一种数据分析方法，通过算法构建分析模型。",
            "confidence": 0.8
        },
        {
            "agent_id": "application_agent",
            "content": "机器学习广泛应用于推荐系统、图像识别和自然语言处理等领域。",
            "confidence": 0.85
        }
    ]
    
    # 融合响应
    print("融合多个Agent的响应...")
    fused_response = aggregator.fuse_responses(agent_responses)
    print(f"  融合后内容: {fused_response['content']}")
    print(f"  置信度: {fused_response['confidence']:.2f}")
    print(f"  来源: {fused_response.get('sources', fused_response.get('source', 'N/A'))}")
    
    # 生成最终响应
    print("\n生成最终响应...")
    query = "什么是机器学习？"
    final_response = aggregator.generate_final_response(fused_response, query)
    print(f"  最终响应: {final_response}")


def main():
    """主函数"""
    print("多Agent智能分流与长程记忆优化系统综合演示")
    print("=" * 50)
    
    demo_router_functionality()
    demo_memory_functionality()
    demo_enhanced_memory_optimizer()
    demo_aggregator_functionality()
    
    print("\n演示完成！")


if __name__ == "__main__":
    main()