import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adapters.llm import call_model

def cli_assistant():
    """
    简单的命令行助手，支持连续对话
    """
    print("欢迎使用DARS-M命令行助手！输入'退出'结束对话。")
    conversation_history = []

    while True:
        user_input = input("用户: ")

        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("再见！")
            break

        # 构建完整的对话历史
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        prompt += f"\n用户: {user_input}\n助手:"

        # 调用模型
        response = call_model(prompt)

        if response:
            print(f"助手: {response}")
            # 更新对话历史
            conversation_history.append({"role": "用户", "content": user_input})
            conversation_history.append({"role": "助手", "content": response})
        else:
            print("抱歉，我无法回答这个问题。请稍后再试。")

if __name__ == "__main__":
    cli_assistant()