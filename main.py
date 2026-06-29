#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多Agent智能分流 + 长程记忆优化方案 —— 主程序入口。

启动各核心组件，注册示例Agent，并对若干示例查询完成"路由 → 执行"的完整流程。
运行方式（在仓库根目录执行）：

    python main.py
"""

import os
import sys

# 确保以仓库根目录为导入根，便于 `core/`、`memory/` 等包被正确解析
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.enhanced_router import EnhancedRouter
from memory.memory_manager import MemoryManager
from distributed.coordinator import DistributedCoordinator
from recovery.recovery_manager import RecoveryManager


# 示例查询，覆盖技术、创意与记忆增强等不同类型
SAMPLE_QUERIES = [
    "用 Python 实现一个二分查找",
    "写一首关于秋天的短诗",
    "我们之前讨论的方案进展如何？",
]


def main():
    """初始化系统、处理示例查询并安全关闭。"""
    print("启动多Agent智能分流 + 长程记忆优化方案...")

    # 初始化各核心组件
    memory_manager = MemoryManager()
    coordinator = DistributedCoordinator()
    recovery_manager = RecoveryManager()
    router = EnhancedRouter()

    # 注册示例Agent，供协调器进行任务分配
    coordinator.register_agent(
        "tech_agent", "worker", ["technical"], endpoint="local://tech"
    )
    coordinator.register_agent(
        "creative_agent", "worker", ["creative"], endpoint="local://creative"
    )

    # 启动故障恢复监控
    recovery_manager.start_monitoring()
    print("系统启动完成，开始处理示例查询...\n")

    try:
        for query in SAMPLE_QUERIES:
            print(f"用户查询: {query}")
            route_result = router.route_query(query)
            response = router.execute_route(query)
            print(f"路由结果: {route_result}")
            print(f"响应: {response}\n")
            # 将交互写入长程记忆，供后续检索增强
            memory_manager.store_memory(query, {"type": "interaction"})
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭系统...")
    finally:
        # 清理：停止后台监控线程
        recovery_manager.stop_monitoring()
        print("系统已关闭。")


if __name__ == "__main__":
    main()
