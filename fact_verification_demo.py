import sys
import os
import json
from colorama import init, Fore, Style

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fact_verifier import FactVerifier
from core.conflict_resolver import ConflictResolver

# 初始化colorama
init()

def print_colored(text, color=Fore.WHITE, style=Style.NORMAL):
    """打印彩色文本"""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 80)
    print_colored(text.center(80), Fore.CYAN, Style.BRIGHT)
    print("=" * 80)

def print_result(result, prefix="结果"):
    """打印验证结果"""
    print(f"\n{prefix}:")
    if isinstance(result, dict):
        for key, value in result.items():
            if key == "verified":
                color = Fore.GREEN if value else Fore.RED
                print_colored(f"  {key}: {value}", color)
            elif key == "confidence":
                # 根据置信度选择颜色
                if value >= 0.8:
                    color = Fore.GREEN
                elif value >= 0.5:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
                print_colored(f"  {key}: {value:.2f}", color)
            elif key == "evidence" or key == "resolution":
                print_colored(f"  {key}: {value}", Fore.CYAN)
            elif key == "related_facts" and isinstance(value, list):
                print_colored(f"  相关事实:", Fore.MAGENTA)
                for i, fact in enumerate(value, 1):
                    print_colored(f"    {i}. {fact}", Fore.MAGENTA)
            else:
                print(f"  {key}: {value}")
    else:
        print(result)

def demo_fact_verification():
    """演示事实验证功能"""
    print_header("基于知识图谱的事实验证演示")
    
    # 初始化事实验证器
    fact_verifier = FactVerifier()
    
    # 添加一些测试知识
    print_colored("\n添加初始知识...", Fore.YELLOW, Style.BRIGHT)
    test_facts = [
        ("地球", "is_a", "行星", 0.99, "system"),
        ("地球", "part_of", "太阳系", 0.99, "system"),
        ("地球", "has_property", "第三颗行星", 0.99, "system"),
        ("太阳", "is_a", "恒星", 0.99, "system"),
        ("太阳", "part_of", "太阳系", 0.99, "system"),
        ("行星", "related_to", "围绕恒星运行", 0.95, "system"),
        ("水", "has_property", "化学式H2O", 0.99, "system"),
        ("水", "has_property", "沸点100摄氏度", 0.95, "system"),
        ("100摄氏度", "related_to", "212华氏度", 0.99, "system"),
    ]
    
    for fact in test_facts:
        fact_verifier.knowledge_graph.add_fact(*fact)
        print(f"  添加: {fact[0]} {fact[1]} {fact[2]}")
    
    # 测试事实验证
    print_header("事实验证测试")
    
    test_statements = [
        "地球是行星",
        "地球是太阳系的第三颗行星",
        "水的化学式是H2O",
        "水的沸点是100摄氏度",
        "水的沸点是212华氏度",
        "地球围绕太阳运行",
        "火星是红色的"
    ]
    
    for statement in test_statements:
        print_colored(f"\n验证陈述: '{statement}'", Fore.YELLOW)
        result = fact_verifier.verify_statement(statement)
        print_result(result)
    
    # 测试从陈述中学习
    print_header("从陈述中学习新知识")
    
    new_statements = [
        "火星是行星",
        "火星是红色的",
        "火星是太阳系的第四颗行星"
    ]
    
    for statement in new_statements:
        print_colored(f"\n学习陈述: '{statement}'", Fore.YELLOW)
        success = fact_verifier.learn_from_statement(statement, 0.9, "demo")
        print(f"  学习{'成功' if success else '失败'}")
    
    # 验证学习效果
    print_colored("\n验证学习效果:", Fore.YELLOW, Style.BRIGHT)
    for statement in new_statements:
        result = fact_verifier.verify_statement(statement)
        print_colored(f"  '{statement}': {'✓' if result['verified'] else '✗'}", 
                     Fore.GREEN if result['verified'] else Fore.RED)
    
    # 测试冲突验证
    print_header("冲突验证测试")
    
    conflict_pairs = [
        ["水的沸点是100摄氏度", "水的沸点是212华氏度"],
        ["地球是太阳系的第三颗行星", "地球是太阳系的第四颗行星"],
        ["火星是红色的", "火星是蓝色的"]
    ]
    
    for pair in conflict_pairs:
        print_colored(f"\n验证冲突: '{pair[0]}' vs '{pair[1]}'", Fore.YELLOW)
        result = fact_verifier.verify_conflict(pair)
        print_result(result)

def demo_conflict_resolver():
    """演示冲突解决器功能"""
    print_header("冲突解决器演示")
    
    # 初始化冲突解决器
    resolver = ConflictResolver()
    
    # 准备测试响应
    responses = [
        {
            "agent_id": "agent1",
            "response": "地球是太阳系中的第三颗行星，距离太阳约1.5亿公里。",
            "confidence": 0.9
        },
        {
            "agent_id": "agent2",
            "response": "地球是太阳系中的第四颗行星，位于火星和金星之间。",
            "confidence": 0.7
        },
        {
            "agent_id": "agent3",
            "response": "水的沸点是100摄氏度（在标准大气压下）。",
            "confidence": 0.95
        },
        {
            "agent_id": "agent4",
            "response": "水的沸点是212华氏度，这是标准大气压下的沸腾温度。",
            "confidence": 0.92
        }
    ]
    
    # 检测冲突
    print_colored("\n检测冲突...", Fore.YELLOW, Style.BRIGHT)
    conflicts = resolver.detect_conflicts(responses)
    
    if not conflicts:
        print_colored("  未检测到冲突", Fore.GREEN)
    else:
        print_colored(f"  检测到 {len(conflicts)} 个冲突", Fore.RED)
        
        # 解决冲突
        print_colored("\n解决冲突...", Fore.YELLOW, Style.BRIGHT)
        for i, conflict in enumerate(conflicts, 1):
            print_colored(f"\n冲突 {i}:", Fore.CYAN)
            print(f"  类型: {conflict['type']}")
            print(f"  陈述: {conflict['statements']}")
            print(f"  Agent: {conflict['agents']}")
            
            resolution = resolver._resolve_factual_conflict(conflict, responses)
            print_colored(f"\n解决方案:", Fore.GREEN)
            print(f"  {resolution}")

def main():
    """主函数"""
    demo_fact_verification()
    demo_conflict_resolver()

if __name__ == "__main__":
    main()