import json
import os
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import faiss
import numpy as np
from adapters.llm import lite_llm_infer
from .memory_optimizer import MemoryOptimizer

class MemoryManager:
    def __init__(self, memory_dir: str = "memory/store", index_path: str = "memory/index.faiss"):
        """初始化记忆管理器

        Args:
            memory_dir: 记忆存储目录
            index_path: FAISS索引文件路径
        """
        self.memory_dir = memory_dir
        self.index_path = index_path
        self.dimension = 768  # 假设使用的嵌入维度是768
        self.optimizer = MemoryOptimizer()

        # 确保目录存在
        os.makedirs(memory_dir, exist_ok=True)

        # 初始化或加载FAISS索引
        self._init_index()

        # 启动定期优化记忆的线程
        self.optimize_thread = threading.Thread(target=self._periodic_optimize, daemon=True)
        self.optimize_thread.start()

    def _periodic_optimize(self):
        """定期优化记忆的后台线程"""
        while True:
            time.sleep(3600)  # 每小时优化一次
            self.optimize_memories()

    def _init_index(self) -> None:
        """初始化或加载FAISS索引"""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"加载现有索引: {self.index_path}")
            except:
                print(f"索引文件 {self.index_path} 损坏，创建新索引")
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            print(f"创建新索引: {self.index_path}")
            self.index = faiss.IndexFlatL2(self.dimension)

    def _save_index(self) -> None:
        """保存FAISS索引"""
        faiss.write_index(self.index, self.index_path)
        print(f"索引已保存到: {self.index_path}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """生成文本的嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        # 这里使用简化的嵌入生成方法，实际应用中可能需要使用专门的嵌入模型
        prompt = f"生成以下文本的嵌入向量:\n{text}\n\n请以JSON格式返回一个长度为{self.dimension}的数组，不要添加其他解释。"
        response = lite_llm_infer(prompt)

        try:
            embedding = json.loads(response)
            return np.array(embedding, dtype=np.float32).reshape(1, -1)
        except json.JSONDecodeError:
            print(f"生成嵌入失败，使用随机向量。LLM响应: {response}")
            return np.random.rand(1, self.dimension).astype(np.float32)

    def store_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储记忆

        Args:
            content: 记忆内容
            metadata: 相关元数据

        Returns:
            记忆ID
        """
        memory_id = f"mem_{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()

        # 生成嵌入
        embedding = self._generate_embedding(content)

        # 添加到索引
        self.index.add(embedding)
        self._save_index()

        # 保存内容和元数据
        memory_data = {
            "id": memory_id,
            "content": content,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }

        memory_file = os.path.join(self.memory_dir, f"{memory_id}.json")
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

        print(f"记忆已存储: {memory_id}")
        return memory_id

    def retrieve_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索与查询相关的记忆

        Args:
            query: 查询文本
            top_k: 返回的最大记忆数量

        Returns:
            相关记忆列表
        """
        if self.index.ntotal == 0:
            return []

        # 生成查询嵌入
        query_embedding = self._generate_embedding(query)

        # 搜索索引
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # 检索记忆内容
        memories = []
        memory_dict = {}
        for i, idx in enumerate(indices[0]):
            memory_file = os.path.join(self.memory_dir, f"mem_{idx}.json")
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    memory_data["similarity_score"] = 1.0 / (1.0 + distances[0][i])  # 转换为相似性分数
                    memories.append(memory_data)
                    memory_dict[memory_data['id']] = memory_data

        # 使用优化器进一步优化检索结果
        memory_ids = [mem['id'] for mem in memories]
        optimized_results = self.optimizer.rank_retrieved_memories(query, memory_dict, memory_ids, top_k)

        # 按优化后的分数排序
        optimized_memories = []
        for mem_id, score in optimized_results:
            memory = memory_dict[mem_id]
            memory['optimized_score'] = score
            optimized_memories.append(memory)

        optimized_memories.sort(key=lambda x: x["optimized_score"], reverse=True)

        return optimized_memories

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """通过ID获取记忆

        Args:
            memory_id: 记忆ID

        Returns:
            记忆数据，如果不存在则返回None
        """
        memory_file = os.path.join(self.memory_dir, f"{memory_id}.json")
        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否成功删除
        """
        memory_file = os.path.join(self.memory_dir, f"{memory_id}.json")
        if os.path.exists(memory_file):
            os.remove(memory_file)
            print(f"记忆已删除: {memory_id}")
            # 注意：这里没有从FAISS索引中删除，实际应用中可能需要重建索引或使用其他方法
            return True
        return False

    def clear_all_memories(self) -> bool:
        """清除所有记忆

        Returns:
            是否成功清除
        """
        try:
            # 删除所有记忆文件
            for filename in os.listdir(self.memory_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.memory_dir, filename))

            # 重置索引
            self.index = faiss.IndexFlatL2(self.dimension)
            self._save_index()

            print("所有记忆已清除")
            return True
        except Exception as e:
            print(f"清除记忆失败: {str(e)}")
            return False

    def optimize_memories(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """优化记忆存储

        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: 优化后的记忆列表和被删除的记忆ID列表
        """
        print("开始优化记忆...")

        # 加载所有记忆
        all_memories = {}
        for filename in os.listdir(self.memory_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.memory_dir, filename), 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                    all_memories[memory['id']] = memory

        # 使用优化器优化记忆
        optimized_memories, deleted_ids = self.optimizer.optimize_memory_storage(all_memories)

        # 删除标记为删除的记忆
        for mem_id in deleted_ids:
            memory_file = os.path.join(self.memory_dir, f"{mem_id}.json")
            if os.path.exists(memory_file):
                os.remove(memory_file)

        # 重建索引
        self.index = faiss.IndexFlatL2(self.dimension)
        for memory in optimized_memories.values():
            # 重新生成嵌入并添加到索引
            embedding = self._generate_embedding(memory['content'])
            self.index.add(embedding)

        self._save_index()

        print(f"记忆优化完成，删除了 {len(deleted_ids)} 条记忆")
        return list(optimized_memories.values()), deleted_ids

    def get_memory_stats(self) -> Dict:
        """获取记忆统计信息

        Returns:
            Dict: 记忆统计信息
        """
        # 加载所有记忆
        all_memories = {}
        for filename in os.listdir(self.memory_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.memory_dir, filename), 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                    all_memories[memory['id']] = memory

        total_count = len(all_memories)
        if total_count == 0:
            return {
                'total_count': 0,
                'index_size': self.index.ntotal,
                'avg_importance': 0,
                'avg_age_days': 0
            }

        # 计算平均重要性 (如果有)
        has_importance = any('importance' in mem for mem in all_memories.values())
        avg_importance = 0
        if has_importance:
            avg_importance = sum(mem.get('importance', 0.5) for mem in all_memories.values()) / total_count

        # 计算平均年龄(天)
        current_time = time.time()
        avg_age_seconds = sum(current_time - datetime.fromisoformat(mem['timestamp']).timestamp() for mem in all_memories.values()) / total_count
        avg_age_days = avg_age_seconds / (24 * 3600)

        return {
            'total_count': total_count,
            'index_size': self.index.ntotal,
            'avg_importance': round(avg_importance, 2) if has_importance else None,
            'avg_age_days': round(avg_age_days, 2)
        }