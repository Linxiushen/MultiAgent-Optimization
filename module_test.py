"""
模块测试脚本
检查各个模块是否能正常导入和运行
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """测试模块导入"""
    print("开始测试模块导入...")
    
    try:
        from core.router import Router
        print("✓ Router模块导入成功")
    except Exception as e:
        print(f"✗ Router模块导入失败: {e}")
        return False
    
    try:
        from core.aggregator import Aggregator
        print("✓ Aggregator模块导入成功")
    except Exception as e:
        print(f"✗ Aggregator模块导入失败: {e}")
        return False
    
    try:
        from memory.memory_manager import MemoryManager
        print("✓ MemoryManager模块导入成功")
    except Exception as e:
        print(f"✗ MemoryManager模块导入失败: {e}")
        return False
    
    try:
        from memory.memory_optimizer_enhanced import MemoryOptimizerEnhanced
        print("✓ MemoryOptimizerEnhanced模块导入成功")
    except Exception as e:
        print(f"✗ MemoryOptimizerEnhanced模块导入失败: {e}")
        return False
    
    print("所有模块导入测试通过！")
    return True


def test_basic_functionality():
    """测试基本功能"""
    print("\n开始测试基本功能...")
    
    try:
        # 测试Router
        from core.router import Router
        router = Router()
        print("✓ Router实例化成功")
        
        # 测试Aggregator
        from core.aggregator import Aggregator
        aggregator = Aggregator()
        print("✓ Aggregator实例化成功")
        
        # 测试MemoryManager
        from memory.memory_manager import MemoryManager
        memory_manager = MemoryManager()
        print("✓ MemoryManager实例化成功")
        
        # 测试MemoryOptimizerEnhanced
        from memory.memory_optimizer_enhanced import MemoryOptimizerEnhanced
        memory_optimizer = MemoryOptimizerEnhanced()
        print("✓ MemoryOptimizerEnhanced实例化成功")
        
        print("所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("模块和功能测试")
    print("=" * 30)
    
    import_success = test_imports()
    if not import_success:
        return False
    
    functionality_success = test_basic_functionality()
    return functionality_success


if __name__ == "__main__":
    success = main()
    if success:
        print("\n所有测试通过！")
    else:
        print("\n测试失败！")
    sys.exit(0 if success else 1)