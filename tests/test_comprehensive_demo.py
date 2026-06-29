"""
测试综合演示脚本
验证comprehensive_demo.py是否能正常运行
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _run_comprehensive_demo():
    """依次运行综合演示脚本的主要功能，出错时抛出异常。"""
    from demos.comprehensive_demo import (
        demo_router_functionality,
        demo_memory_functionality,
        demo_enhanced_memory_optimizer,
        demo_aggregator_functionality
    )

    print("开始测试综合演示脚本...")
    demo_router_functionality()
    demo_memory_functionality()
    demo_enhanced_memory_optimizer()
    demo_aggregator_functionality()
    print("综合演示脚本测试完成！所有功能正常运行。")


def test_comprehensive_demo():
    """测试综合演示脚本可正常运行（出错时异常会使测试失败）。"""
    _run_comprehensive_demo()


if __name__ == "__main__":
    try:
        _run_comprehensive_demo()
        sys.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)