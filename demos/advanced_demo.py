"""高级演示：展示路由与长程记忆的组合使用。

运行方式（在仓库根目录执行）：
    python demos/advanced_demo.py
"""

import os
from core.router import Router
from memory import MemoryManager
from adapters.llm import load_environment


def clear_screen():
    """清除终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_with_color(text, color_code=32):  # 默认绿色
    """带颜色打印文本"""
    print(f"\033[{color_code}m{text}\033[0m")


def print_separator():
    """打印分隔线"""
    print_with_color("=" * 60)


def main():
    """演示主流程：初始化路由与记忆系统，处理若干示例查询。"""
    load_environment()
    print_separator()
    print_with_color("高级演示：多Agent智能分流 + 长程记忆")
    print_separator()

    router = Router()
    memory_manager = MemoryManager()

    sample_queries = [
        "帮我写一个快速排序算法",
        "讲个关于宇宙的小故事",
        "我们上次聊到的项目进展如何？",
    ]

    for query in sample_queries:
        print_with_color(f"\n用户查询: {query}", color_code=36)
        decision = router.route_query(query)
        print(f"路由结果: {decision}")
        # 将查询写入长程记忆，供后续检索增强使用
        memory_manager.store_memory(query, {"type": "interaction"})

    print_separator()
    print_with_color("演示结束")
    print_separator()


if __name__ == "__main__":
    main()
