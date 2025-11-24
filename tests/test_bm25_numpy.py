import sys
import os
import unittest

# Add src to path so we can import mcp_jieba
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from mcp_jieba.engine import JiebaEngine

class TestBM25Numpy(unittest.TestCase):
    def setUp(self):
        self.engine = JiebaEngine()

    def test_single_string_input(self):
        text = "这是一个测试文本。我们需要提取关键词。关键词提取很重要。"
        print(f"Testing single string input: {text}")
        result = self.engine.extract_keywords_bm25(text, top_k=3)
        print(f"Result: {result}")

        self.assertIsInstance(result, dict)
        self.assertIn('0', result)
        self.assertIsInstance(result['0'], list)
        # Check if we got some keywords
        self.assertTrue(len(result['0']) > 0)

    def test_list_input(self):
        texts = [
            "机器学习是人工智能的一个分支。",
            "深度学习是机器学习的一种方法。"
        ]
        print(f"\nTesting list input: {texts}")
        result = self.engine.extract_keywords_bm25(texts, top_k=3)
        print(f"Result: {result}")

        self.assertIsInstance(result, dict)
        self.assertIn('0', result)
        self.assertIn('1', result)
        self.assertIsInstance(result['0'], list)
        self.assertIsInstance(result['1'], list)

if __name__ == "__main__":
    unittest.main()
