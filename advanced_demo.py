import os
import sys
import time
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
    print_with_color("=