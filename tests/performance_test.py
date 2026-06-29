import unittest
import time
import uuid
import threading
import random
from memory.memory_manager import memory_manager
from core.router import router
from distributed.coordinator import coordinator

class PerformanceTest(unittest.TestCase):
    def setUp(self):
        # 重置记忆管理器
        memory_manager.clear_all_memories()
        # 重置协调器
        coordinator.agents = {}
        # 注册测试Agent
        for i in range(5):
            coordinator.register_agent(
                agent_id=f'agent_{i}',
                agent_type='llm',
                capabilities=['text_generation', 'translation', 'summarization'],
                endpoint=f'localhost:800{i}'
            )

    def test_memory_storage_performance(self):
        """测试记忆存储性能"""
        print("开始测试记忆存储性能...")
        start_time = time.time()
        memory_ids = []

        # 存储1000条记忆
        for i in range(1000):
            content = f"这是测试记忆内容 #{i}，包含一些随机信息 {uuid.uuid4()}"
            tags = [f'tag_{i%10}', 'performance_test']
            importance = random.uniform(0.1, 1.0)
            memory_id = memory_manager.store_memory(content, tags, importance)
            memory_ids.append(memory_id)

        end_time = time.time()
        duration = end_time - start_time
        print(f"存储1000条记忆耗时: {duration:.2f}秒")
        print(f"平均每条记忆存储时间: {duration/1000*1000:.2f}毫秒")

        # 验证所有记忆都被存储
        self.assertEqual(len(memory_manager.memories), 1000, "存储的记忆数量不正确")

    def test_memory_retrieval_performance(self):
        """测试记忆检索性能"""
        # 首先存储一些测试记忆
        for i in range(1000):
            content = f"这是测试记忆内容 #{i}，关于 {['科技', '体育', '文化', '历史', '艺术'][i%5]}"
            memory_manager.store_memory(content, [f'topic_{i%5}'])

        print("开始测试记忆检索性能...")
        start_time = time.time()

        # 执行100次检索
        for i in range(100):
            query = f"关于 {['科技', '体育', '文化', '历史', '艺术'][i%5]} 的内容"
            results = memory_manager.retrieve_memory(query, top_k=5)

        end_time = time.time()
        duration = end_time - start_time
        print(f"执行100次记忆检索耗时: {duration:.2f}秒")
        print(f"平均每次检索时间: {duration/100*1000:.2f}毫秒")

    def test_router_performance(self):
        """测试路由系统性能"""
        print("开始测试路由系统性能...")
        start_time = time.time()

        # 执行100次路由查询
        for i in range(100):
            query = f"{['生成', '翻译', '总结', '回答', '分析'][i%5]}关于 {uuid.uuid4()} 的内容"
            response = router.route_query(query)

        end_time = time.time()
        duration = end_time - start_time
        print(f"执行100次路由查询耗时: {duration:.2f}秒")
        print(f"平均每次查询时间: {duration/100*1000:.2f}毫秒")

    def test_concurrent_requests(self):
        """测试并发请求处理能力"""
        print("开始测试并发请求处理能力...")
        num_threads = 10
        requests_per_thread = 20

        def worker():
            for _ in range(requests_per_thread):
                # 随机选择执行记忆操作或路由查询
                if random.random() > 0.5:
                    # 记忆操作
                    content = f"并发测试记忆 {uuid.uuid4()}"
                    memory_manager.store_memory(content)
                else:
                    # 路由查询
                    query = f"并发测试查询 {uuid.uuid4()}"
                    router.route_query(query)

        threads = []
        start_time = time.time()

        # 创建并启动线程
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        end_time = time.time()
        duration = end_time - start_time
        total_requests = num_threads * requests_per_thread
        print(f"处理 {total_requests} 个并发请求耗时: {duration:.2f}秒")
        print(f"平均请求处理时间: {duration/total_requests*1000:.2f}毫秒")
        print(f"吞吐量: {total_requests/duration:.2f} 请求/秒")

    def test_memory_optimization_performance(self):
        """测试记忆优化性能"""
        # 首先存储一些测试记忆
        for i in range(1000):
            content = f"这是测试记忆内容 #{i}，{['重复内容', '相似内容', '不同内容'][i%3]} {uuid.uuid4()}"
            memory_manager.store_memory(content)

        print("开始测试记忆优化性能...")
        start_time = time.time()

        # 执行记忆优化
        optimized_memories, deleted = memory_manager.optimize_memories()

        end_time = time.time()
        duration = end_time - start_time
        print(f"优化1000条记忆耗时: {duration:.2f}秒")
        print(f"优化后保留记忆数: {len(optimized_memories)}")
        print(f"优化后删除记忆数: {len(deleted)}")

if __name__ == '__main__':
    unittest.main()