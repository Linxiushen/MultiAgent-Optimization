import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_optimizer import MemoryOptimizer

def test_semantic_similarity():
    """测试记忆优化器的语义相似度功能"""
    # 初始化记忆优化器
    optimizer = MemoryOptimizer()
    
    # 创建测试记忆
    memories = {
        "mem1": {
            "content": "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。",
            "timestamp": 1630000000,
            "tags": ["AI", "科技"],
            "access_count": 5
        },
        "mem2": {
            "content": "机器学习是人工智能的一个子领域，专注于让计算机系统从数据中学习。",
            "timestamp": 1630001000,
            "tags": ["机器学习", "AI"],
            "access_count": 3
        },
        "mem3": {
            "content": "深度学习是机器学习的一种方法，使用神经网络进行学习。",
            "timestamp": 1630002000,
            "tags": ["深度学习", "神经网络"],
            "access_count": 2
        },
        "mem4": {
            "content": "自然语言处理是AI的一个应用领域，专注于让计算机理解和生成人类语言。",
            "timestamp": 1630003000,
            "tags": ["NLP", "语言处理"],
            "access_count": 4
        },
        "mem5": {
            "content": "今天的天气真好，阳光明媚，适合出门散步。",
            "timestamp": 1630004000,
            "tags": ["天气", "生活"],
            "access_count": 1
        },
        "mem6": {
            "content": "人工智能技术正在快速发展，影响着各个行业。",
            "timestamp": 1630005000,
            "tags": ["AI", "发展"],
            "access_count": 3
        },
        "mem7": {
            "content": "神经网络是一种模拟人脑结构的计算模型，是深度学习的基础。",
            "timestamp": 1630006000,
            "tags": ["神经网络", "深度学习"],
            "access_count": 2
        }
    }
    
    # 测试压缩记忆功能
    print("测试压缩记忆功能...")
    compressed_memories, removed_ids = optimizer.compress_memories(memories)
    print(f"压缩前记忆数量: {len(memories)}")
    print(f"压缩后记忆数量: {len(compressed_memories)}")
    print(f"被移除的记忆ID: {removed_ids}")
    print()
    
    # 测试检索相关记忆功能
    print("测试检索相关记忆功能...")
    queries = [
        "人工智能技术的发展",
        "机器学习和深度学习的区别",
        "今天天气如何？",
        "计算机如何理解人类语言"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        relevant_memories = optimizer.retrieve_relevant_memories(query, memories, top_k=2)
        print(f"找到 {len(relevant_memories)} 条相关记忆:")
        for i, memory in enumerate(relevant_memories):
            print(f"  {i+1}. {memory['content'][:50]}...")
    
    # 测试高级记忆压缩算法
    print("\n测试高级记忆压缩算法...")
    advanced_compressed = optimizer.advanced_memory_compression(memories)
    print(f"压缩前记忆数量: {len(memories)}")
    print(f"高级压缩后记忆数量: {len(advanced_compressed)}")
    
    # 显示压缩后的记忆内容
    print("\n压缩后的记忆内容:")
    for i, (mid, memory) in enumerate(advanced_compressed.items()):
        print(f"\n记忆 {i+1} (ID: {mid}):")
        print(f"内容: {memory['content'][:100]}..." if len(memory['content']) > 100 else f"内容: {memory['content']}")
        if 'source_memories' in memory:
            print(f"源记忆: {memory['source_memories']}")
        print(f"标签: {memory['tags']}")

def test_memory_optimization_pipeline():
    """测试完整的记忆优化流程"""
    print("\n\n测试完整的记忆优化流程...")
    
    # 初始化记忆优化器
    optimizer = MemoryOptimizer()
    
    # 创建大量测试记忆
    memories = {}
    for i in range(20):
        category = i % 4
        if category == 0:
            content = f"人工智能研究报告 {i}: 探讨AI技术在{['医疗', '金融', '教育', '交通', '安防'][i % 5]}领域的应用前景。"
            tags = ["AI", "研究", ["医疗", "金融", "教育", "交通", "安防"][i % 5]]
        elif category == 1:
            content = f"机器学习算法分析 {i}: 比较{['决策树', '随机森林', 'SVM', '神经网络', '强化学习'][i % 5]}的优缺点和适用场景。"
            tags = ["机器学习", "算法", ["决策树", "随机森林", "SVM", "神经网络", "强化学习"][i % 5]]
        elif category == 2:
            content = f"数据科学项目 {i}: 使用{['Python', 'R', 'Julia', 'MATLAB', 'Scala'][i % 5]}进行大数据分析和可视化。"
            tags = ["数据科学", "编程", ["Python", "R", "Julia", "MATLAB", "Scala"][i % 5]]
        else:
            content = f"日常笔记 {i}: 今天{['参加了会议', '学习了新技术', '完成了项目', '解决了问题', '计划了行程'][i % 5]}，感觉很充实。"
            tags = ["日常", "笔记", ["会议", "学习", "项目", "问题", "计划"][i % 5]]
        
        memories[f"mem{i}"] = {
            "content": content,
            "timestamp": 1630000000 + i * 10000,
            "tags": tags,
            "access_count": i % 10
        }
    
    # 1. 先进行高级压缩
    print("\n1. 执行高级记忆压缩...")
    compressed_memories = optimizer.advanced_memory_compression(memories)
    print(f"压缩前记忆数量: {len(memories)}")
    print(f"压缩后记忆数量: {len(compressed_memories)}")
    
    # 2. 然后进行优先级排序
    print("\n2. 执行记忆优先级排序...")
    priorities = optimizer.prioritize_memories(compressed_memories)
    print(f"前5个高优先级记忆:")
    for i, (mid, score) in enumerate(priorities[:5]):
        print(f"  {i+1}. ID: {mid}, 分数: {score:.4f}, 内容: {compressed_memories[mid]['content'][:50]}...")
    
    # 3. 最后进行遗忘操作
    print("\n3. 执行低优先级记忆遗忘...")
    kept_memories, forgotten_ids = optimizer.forget_low_priority_memories(compressed_memories, keep_ratio=0.6)
    print(f"遗忘前记忆数量: {len(compressed_memories)}")
    print(f"遗忘后记忆数量: {len(kept_memories)}")
    print(f"被遗忘的记忆数量: {len(forgotten_ids)}")
    
    # 4. 测试查询
    print("\n4. 测试优化后的记忆检索...")
    queries = ["人工智能在医疗领域的应用", "机器学习算法比较", "Python数据分析", "日常笔记"]
    
    for query in queries:
        print(f"\n查询: {query}")
        relevant_memories = optimizer.retrieve_relevant_memories(query, kept_memories, top_k=2)
        print(f"找到 {len(relevant_memories)} 条相关记忆:")
        for i, memory in enumerate(relevant_memories):
            print(f"  {i+1}. {memory['content'][:50]}...")

if __name__ == "__main__":
    test_semantic_similarity()
    test_memory_optimization_pipeline()