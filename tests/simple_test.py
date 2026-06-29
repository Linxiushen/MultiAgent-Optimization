"""
简化测试脚本
逐步测试每个功能模块
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_router():
    """测试路由功能"""
    print("测试路由功能...")
    try:
        from core.router import Router
        router = Router()
        print("✓ Router模块导入和实例化成功")
        
        # 测试路由功能
        query = "什么是机器学习？"
        result = router.route_query(query)
        print(f"✓ 路由查询成功: {result}")
        
        return True
    except Exception as e:
        print(f"✗ 路由功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory():
    """测试记忆功能"""
    print("\n测试记忆功能...")
    try:
        from memory.memory_manager import MemoryManager
        memory_manager = MemoryManager()
        print("✓ MemoryManager模块导入和实例化成功")
        
        # 测试存储记忆
        memory_id = memory_manager.store_memory("测试记忆内容", {"category": "test"})
        print(f"✓ 记忆存储成功: {memory_id}")
        
        # 测试检索记忆
        memories = memory_manager.retrieve_memory("测试", top_k=1)
        print(f"✓ 记忆检索成功: {len(memories)} 条记录")
        
        return True
    except Exception as e:
        print(f"✗ 记忆功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_optimizer():
    """测试记忆优化器功能"""
    print("\n测试记忆优化器功能...")
    try:
        from memory.memory_optimizer_enhanced import MemoryOptimizerEnhanced
        optimizer = MemoryOptimizerEnhanced()
        print("✓ MemoryOptimizerEnhanced模块导入和实例化成功")
        
        # 测试重要性计算
        test_memory = {
            'content': '这是一个测试记忆',
            'timestamp': 1000,
            'importance': 0.8,
            'access_count': 5
        }
        importance = optimizer.calculate_importance(test_memory)
        print(f"✓ 重要性计算成功: {importance}")
        
        return True
    except Exception as e:
        print(f"✗ 记忆优化器功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregator():
    """测试聚合器功能"""
    print("\n测试聚合器功能...")
    try:
        from core.aggregator import Aggregator
        aggregator = Aggregator()
        print("✓ Aggregator模块导入和实例化成功")
        
        # 测试响应融合
        responses = [
            {"agent_id": "test1", "content": "测试内容1", "confidence": 0.8},
            {"agent_id": "test2", "content": "测试内容2", "confidence": 0.7}
        ]
        fused = aggregator.fuse_responses(responses)
        print(f"✓ 响应融合成功: {fused}")
        
        return True
    except Exception as e:
        print(f"✗ 聚合器功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("简化功能测试")
    print("=" * 30)
    
    # 逐个测试各个模块
    tests = [
        test_router,
        test_memory,
        test_memory_optimizer,
        test_aggregator
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试 {test.__name__} 发生异常: {e}")
            results.append(False)
    
    # 汇总结果
    passed = sum(results)
    total = len(results)
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("所有测试通过！")
        return True
    else:
        print("部分测试失败！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)