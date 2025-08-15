"""
测试综合演示脚本
验证comprehensive_demo.py是否能正常运行
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_comprehensive_demo():
    """测试综合演示脚本"""
    try:
        # 导入并运行综合演示脚本的主要功能
        from demos.comprehensive_demo import (
            demo_router_functionality,
            demo_memory_functionality,
            demo_enhanced_memory_optimizer,
            demo_aggregator_functionality
        )
        
        print("开始测试综合演示脚本...")
        
        # 测试路由功能
        demo_router_functionality()
        
        # 测试记忆功能
        demo_memory_functionality()
        
        # 测试增强版记忆优化器功能
        demo_enhanced_memory_optimizer()
        
        # 测试聚合器功能
        demo_aggregator_functionality()
        
        print("综合演示脚本测试完成！所有功能正常运行。")
        return True
        
    except Exception as e:
        print(f"综合演示脚本测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive_demo()
    sys.exit(0 if success else 1)