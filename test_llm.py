import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adapters.llm import call_model, lite_llm_infer, heavy_llm_infer

class TestLLMAdapter(unittest.TestCase):
    @patch('adapters.llm.OpenAI')
    def test_call_model_success(self, mock_openai):
        # 配置mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "测试响应"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # 测试函数
        result = call_model("测试提示")

        # 验证结果
        self.assertEqual(result, "测试响应")
        mock_client.chat.completions.create.assert_called_once()

    @patch('adapters.llm.OpenAI')
    def test_call_model_error(self, mock_openai):
        # 配置mock抛出异常
        mock_openai.side_effect = Exception("测试异常")

        # 测试函数
        result = call_model("测试提示")

        # 验证结果
        self.assertIsNone(result)

    def test_lite_llm_infer(self):
        # 测试轻量级LLM调用
        with patch('adapters.llm.call_model') as mock_call_model:
            mock_call_model.return_value = "轻量级响应"
            result = lite_llm_infer("测试提示")
            self.assertEqual(result, "轻量级响应")
            mock_call_model.assert_called_once_with(
                "测试提示", model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=500
            )

    def test_heavy_llm_infer(self):
        # 测试重量级LLM调用
        with patch('adapters.llm.call_model') as mock_call_model:
            mock_call_model.return_value = "重量级响应"
            result = heavy_llm_infer("测试提示")
            self.assertEqual(result, "重量级响应")
            mock_call_model.assert_called_once_with(
                "测试提示", model_name="gpt-4", temperature=0.7, max_tokens=2000
            )

if __name__ == '__main__':
    unittest.main()