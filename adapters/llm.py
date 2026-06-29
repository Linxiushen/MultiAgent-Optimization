import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 加载环境变量
def load_environment():
    """加载 .env 中的环境变量。

    缺少 OPENAI_API_KEY 时仅记录警告并返回 False，而不是抛出异常——
    这样在客户端被 mock 或仅做离线测试时，调用方不会因为缺少密钥而直接崩溃。
    实际发起请求时若密钥无效，会由 OpenAI 客户端自行报错。

    Returns:
        bool: 是否检测到 OPENAI_API_KEY
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("未在环境变量中找到 OPENAI_API_KEY，涉及真实 LLM 调用的功能将不可用")
        return False
    return True

# 初始化OpenAI客户端
def init_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 调用模型的通用接口
def call_model(prompt, model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1000):
    """
    调用指定的LLM模型生成回答

    参数:
        prompt (str): 输入提示
        model_name (str): 模型名称
        temperature (float): 温度参数
        max_tokens (int): 最大生成token数

    返回:
        str: 模型生成的回答
    """
    try:
        # 确保环境已加载
        load_environment()

        # 初始化客户端
        client = init_openai_client()

        # 调用模型
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling model {model_name}: {e}")
        return None

# 轻量级LLM调用
def lite_llm_infer(prompt):
    return call_model(prompt, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=500)

# 重量级LLM调用
def heavy_llm_infer(prompt):
    return call_model(prompt, model_name="gpt-4", temperature=0.7, max_tokens=2000)