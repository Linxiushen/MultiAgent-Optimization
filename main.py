#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多Agent智能分流 + 长程记忆优化方案
主程序入口
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.router import Router
from memory.memory_manager import MemoryManager
from distributed.coordinator import Coordinator
from recovery.recovery_manager import RecoveryManager


def main():
    """主函数"""
    print("启动多Agent智能分流 + 长程记忆优化方案...")
    
    # 初始化各核心组件
    memory_manager = MemoryManager()
    coordinator = Coordinator()
    recovery_manager = RecoveryManager()
    router = Router(memory_manager, coordinator, recovery_manager)
    
    # 启动分布式协调器
    coordinator.start()
    
    # 启动故障恢复管理器
    recovery_manager.start_monitoring()
    
    # 启动路由系统
    router.start()
    
    print("系统启动完成，等待请求...")
    
    # 这里可以添加主循环或事件监听器
    # 例如：监听用户输入、网络请求等
    try:
        # 模拟主循环
        while True:
            # 处理请求的逻辑
            pass
    except KeyboardInterrupt:
        print("\n正在关闭系统...")
        # 执行清理操作
        coordinator.stop()
        recovery_manager.stop_monitoring()
        router.stop()
        print("系统已关闭。")


if __name__ == "__main__":
    main()