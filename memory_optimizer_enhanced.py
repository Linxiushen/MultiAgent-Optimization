import time
import heapq
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json
import os
import re
from textblob import TextBlob
import nltk

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

class MemoryOptimizerEnhanced:
    """增强版记忆优化器，扩展了基础版的功能，增加了聚类、摘要和情感分析"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.memory_vectors = {}
        self.load_config()
        self.sia = SentimentIntensityAnalyzer()
        self.cluster_model = None

    def load_config(self):
        """加载记忆优化配置"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'memory_optimizer_config.json')
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # 使用默认配置
            self.config = {
                "compression_threshold": 0.7,
                "importance_decay_rate": 0.01,
                "max_similar_vectors": 5,
                "relevance_threshold": 0.5,
                "cluster_count": 5,
                "summary_length": 3
            }

    def calculate_importance(self, memory: Dict) -> float:
        """计算记忆的重要性分数，考虑情感因素和用户反馈"""
        current_time = time.time()
        creation_time = memory.get('timestamp', current_time)
        time_diff = current_time - creation_time

        # 基础重要性分数
        base_importance = memory.get('importance', 0.5)

        # 计算时间衰减
        decay_factor = np.exp(-self.config['importance_decay_rate'] * time_diff)

        # 考虑访问频率
        access_count = memory.get('access_count', 1)
        frequency_factor = min(1.5, 1 + np.log10(access_count))

        # 考虑情感因素
        content = memory.get('content', '')
        if content:
            sentiment = self.sia.polarity_scores(content)
            # 情感分数映射到0.8-1.2范围
            sentiment_factor = 1 + 0.2 * sentiment['compound']
        else:
            sentiment_factor = 1.0

        # 考虑用户标记的重要性
        user_importance = memory.get('user_importance', 1.0)

        # 综合计算最终重要性
        importance = base_importance * decay_factor * frequency_factor * sentiment_factor * user_importance
        return importance

    def compress_memories(self, memories: Dict[str, Dict]) -> Tuple[Dict[str, Dict], List[str]]:
        """压缩相似记忆"""
        if not memories:
            return memories, []

        # 提取记忆文本并向量化
        memory_ids = list(memories.keys())
        memory_texts = [memories[mid]['content'] for mid in memory_ids]

        # 更新向量器并转换文本
        try:
            vectors = self.vectorizer.fit_transform(memory_texts)
        except ValueError:
            # 处理空文本情况
            return memories, []

        # 存储向量以便后续检索
        for i, mid in enumerate(memory_ids):
            self.memory_vectors[mid] = vectors[i]

        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(vectors)

        # 记录要删除的记忆ID
        to_delete = set()
        compressed_memories = memories.copy()

        # 查找并合并相似记忆
        threshold = self.config['compression_threshold']
        for i in range(len(memory_ids)):
            if memory_ids[i] in to_delete:
                continue

            similar_indices = [j for j in range(len(memory_ids)) if i != j and similarity_matrix[i][j] > threshold]
            if not similar_indices:
                continue

            # 合并相似记忆
            main_memory = compressed_memories[memory_ids[i]]
            for j in similar_indices:
                if memory_ids[j] in to_delete:
                    continue

                similar_memory = compressed_memories[memory_ids[j]]
                # 合并内容
                main_memory['content'] = f"{main_memory['content']}\n{similar_memory['content']}"
                # 更新时间戳为最新的
                main_memory['timestamp'] = max(main_memory.get('timestamp', 0), similar_memory.get('timestamp', 0))
                # 合并标签
                main_tags = set(main_memory.get('tags', []))
                main_tags.update(similar_memory.get('tags', []))
                main_memory['tags'] = list(main_tags)
                # 增加访问计数
                main_memory['access_count'] = main_memory.get('access_count', 1) + similar_memory.get('access_count', 1)

                to_delete.add(memory_ids[j])
                del compressed_memories[memory_ids[j]]
                if memory_ids[j] in self.memory_vectors:
                    del self.memory_vectors[memory_ids[j]]

        return compressed_memories, list(to_delete)

    def retrieve_relevant_memories(self, query: str, memories: Dict[str, Dict], top_k: int = 5) -> List[Tuple[str, float]]:
        """检索与查询相关的记忆"""
        if not memories:
            return []

        # 确保所有记忆都有向量表示
        memory_ids = list(memories.keys())
        for mid in memory_ids:
            if mid not in self.memory_vectors and 'content' in memories[mid]:
                try:
                    self.memory_vectors[mid] = self.vectorizer.transform([memories[mid]['content']])
                except ValueError:
                    # 忽略空文本
                    continue

        # 对查询进行向量化
        try:
            query_vector = self.vectorizer.transform([query])
        except ValueError:
            # 处理空查询
            return []

        # 计算查询与每个记忆的相似度
        similarities = []
        for mid in memory_ids:
            if mid in self.memory_vectors:
                similarity = cosine_similarity(query_vector, self.memory_vectors[mid])[0][0]
                if similarity >= self.config['relevance_threshold']:
                    # 结合重要性分数进行排序
                    importance = self.calculate_importance(memories[mid])
                    combined_score = similarity * importance
                    similarities.append((mid, combined_score))

        # 按分数降序排序并返回前top_k个结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def prioritize_memories(self, memories: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """对记忆进行优先级排序"""
        # 计算每个记忆的优先级分数
        priorities = []
        for mid, memory in memories.items():
            importance = self.calculate_importance(memory)
            priorities.append((mid, importance))

        # 按优先级降序排序
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def forget_low_priority_memories(self, memories: Dict[str, Dict], keep_ratio: float = 0.8) -> List[str]:
        """遗忘低优先级的记忆

        Args:
            memories: 记忆字典
            keep_ratio: 保留的记忆比例

        Returns:
            List[str]: 被删除的记忆ID列表
        """
        if not memories:
            return []

        # 计算每个记忆的优先级
        priorities = self.prioritize_memories(memories)

        # 确定要保留的记忆数量
        keep_count = max(1, int(len(memories) * keep_ratio))

        # 保留高优先级的记忆
        keep_ids = set([mid for mid, _ in priorities[:keep_count]])

        # 确定要删除的记忆
        to_delete = [mid for mid in memories.keys() if mid not in keep_ids]

        # 删除低优先级记忆
        for mid in to_delete:
            del memories[mid]
            if mid in self.memory_vectors:
                del self.memory_vectors[mid]

        return to_delete

    def save_memories(self, memories: Dict[str, Dict], file_path: str) -> bool:
        """将记忆保存到文件

        Args:
            memories: 要保存的记忆字典
            file_path: 保存文件的路径

        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 转换为可序列化的格式
            serializable_memories = {}
            for mid, memory in memories.items():
                # 复制记忆对象
                serializable_memory = memory.copy()
                # 确保所有值都是可序列化的
                for key, value in serializable_memory.items():
                    if isinstance(value, np.ndarray):
                        serializable_memory[key] = value.tolist()
                    elif isinstance(value, (np.float32, np.float64)):
                        serializable_memory[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        serializable_memory[key] = int(value)
                serializable_memories[mid] = serializable_memory

            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_memories, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"保存记忆失败: {str(e)}")
            return False

    def load_memories(self, file_path: str) -> Dict[str, Dict]:
        """从文件加载记忆

        Args:
            file_path: 记忆文件的路径

        Returns:
            Dict[str, Dict]: 加载的记忆字典
        """
        try:
            if not os.path.exists(file_path):
                print(f"记忆文件不存在: {file_path}")
                return {}

            # 从文件加载
            with open(file_path, 'r', encoding='utf-8') as f:
                memories = json.load(f)

            # 重建向量表示
            self.memory_vectors = {}
            for mid, memory in memories.items():
                if 'content' in memory:
                    try:
                        self.memory_vectors[mid] = self.vectorizer.transform([memory['content']])
                    except ValueError:
                        # 忽略空文本
                        continue

            # 如果有聚类模型，重新训练
            if self.cluster_model is not None and len(memories) > 0:
                self.cluster_memories(memories)

            return memories
        except Exception as e:
            print(f"加载记忆失败: {str(e)}")
            return {}

        # 优先级排序
        prioritized = self.prioritize_memories(memories)

        # 确定要保留的记忆数量
        keep_count = max(1, int(len(prioritized) * keep_ratio))

        # 确定要删除的记忆
        to_forget = [mid for mid, _ in prioritized[keep_count:]]

        return to_forget

    def optimize_memory_storage(self, memories: Dict[str, Dict]) -> Tuple[Dict[str, Dict], List[str]]:
        """全面优化记忆存储"""
        # 第一步：压缩相似记忆
        compressed_memories, deleted_due_to_compression = self.compress_memories(memories)

        # 第二步：遗忘低优先级记忆
        deleted_due_to_priority = self.forget_low_priority_memories(compressed_memories)

        # 合并删除列表
        all_deleted = list(set(deleted_due_to_compression + deleted_due_to_priority))

        # 创建最终的记忆字典
        final_memories = {mid: memory for mid, memory in compressed_memories.items() if mid not in all_deleted}

        return final_memories, all_deleted

    def cluster_memories(self, memories: Dict[str, Dict]) -> Dict[int, List[str]]:
        """对记忆进行聚类

        Args:
            memories: 记忆字典

        Returns:
            Dict[int, List[str]]: 聚类ID到记忆ID列表的映射
        """
        if not memories:
            return {}

        # 确保所有记忆都有向量表示
        memory_ids = list(memories.keys())
        for mid in memory_ids:
            if mid not in self.memory_vectors and 'content' in memories[mid]:
                try:
                    self.memory_vectors[mid] = self.vectorizer.transform([memories[mid]['content']])
                except ValueError:
                    # 忽略空文本
                    continue

        # 提取有效的向量和对应的记忆ID
        valid_vectors = []
        valid_ids = []
        for mid in memory_ids:
            if mid in self.memory_vectors:
                valid_vectors.append(self.memory_vectors[mid].toarray()[0])
                valid_ids.append(mid)

        if not valid_vectors:
            return {}

        # 执行K-means聚类
        n_clusters = min(self.config['cluster_count'], len(valid_vectors))
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.cluster_model.fit_predict(valid_vectors)

        # 构建聚类结果
        clusters = {i: [] for i in range(n_clusters)}
        for i, mid in enumerate(valid_ids):
            clusters[labels[i]].append(mid)

        return clusters

    def generate_summary(self, memory_content: str) -> str:
        """生成记忆内容的摘要

        Args:
            memory_content: 记忆内容文本

        Returns:
            str: 生成的摘要
        """
        if not memory_content:
            return ""

        # 使用TextBlob进行句子分割
        blob = TextBlob(memory_content)
        sentences = list(blob.sentences)

        if not sentences:
            return ""

        # 如果句子数量少于summary_length，直接返回原文
        if len(sentences) <= self.config['summary_length']:
            return memory_content.strip()

        # 简单的摘要生成：取前几个句子
        # 实际应用中可以使用更复杂的算法如TextRank
        summary_sentences = sentences[:self.config['summary_length']]
        summary = ' '.join([str(sent) for sent in summary_sentences])

        return summary

    def auto_optimize(self, memories: Dict[str, Dict], interval: int = 3600) -> Tuple[Dict[str, Dict], List[str]]:
        """定期自动优化记忆

        Args:
            memories: 记忆字典
            interval: 优化间隔（秒）

        Returns:
            Tuple[Dict[str, Dict], List[str]]: 优化后的记忆字典和被删除的记忆ID列表
        """
        last_optimization = getattr(self, 'last_optimization_time', 0)
        current_time = time.time()

        if current_time - last_optimization < interval:
            # 未到优化时间
            return memories, []

        # 执行优化
        optimized_memories, deleted = self.optimize_memory_storage(memories)

        # 更新最后优化时间
        self.last_optimization_time = current_time

        return optimized_memories, deleted

    def get_memory_connections(self, memory_id: str, memories: Dict[str, Dict], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """获取与指定记忆相关的其他记忆

        Args:
            memory_id: 记忆ID
            memories: 记忆字典
            threshold: 相关度阈值

        Returns:
            List[Tuple[str, float]]: 相关记忆ID和相关度分数的列表
        """
        if memory_id not in self.memory_vectors or memory_id not in memories:
            return []

        # 获取当前记忆的向量
        current_vector = self.memory_vectors[memory_id]

        # 计算与其他记忆的相似度
        connections = []
        for mid, vector in self.memory_vectors.items():
            if mid != memory_id and mid in memories:
                similarity = cosine_similarity(current_vector, vector)[0][0]
                if similarity >= threshold:
                    connections.append((mid, similarity))

        # 按相似度降序排序
        connections.sort(key=lambda x: x[1], reverse=True)
        return connections

    def analyze_memory_sentiment(self, memory: Dict) -> Dict[str, float]:
        """分析记忆的情感

        Args:
            memory: 记忆对象

        Returns:
            Dict[str, float]: 情感分析结果
        """
        content = memory.get('content', '')
        if not content:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

        return self.sia.polarity_scores(content)

    def create_memory_index(self, memories: Dict[str, Dict]) -> None:
        """创建记忆索引以加速检索

        Args:
            memories: 记忆字典
        """
        # 提取所有记忆文本
        memory_texts = [memories[mid]['content'] for mid in memories.keys()]

        # 更新向量器
        try:
            self.vectorizer.fit(memory_texts)
        except ValueError:
            # 处理空文本情况
            return

        # 重新计算所有记忆的向量
        self.memory_vectors = {}
        for mid, memory in memories.items():
            try:
                self.memory_vectors[mid] = self.vectorizer.transform([memory['content']])
            except ValueError:
                # 忽略空文本
                continue

    def export_memory_snapshot(self, memories: Dict[str, Dict], file_path: str) -> bool:
        """导出记忆快照

        Args:
            memories: 记忆字典
            file_path: 导出文件路径

        Returns:
            bool: 是否导出成功
        """
        try:
            # 转换为可序列化的格式
            serializable_memories = {}
            for mid, memory in memories.items():
                # 移除不可序列化的对象
                serializable_memory = {k: v for k, v in memory.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                serializable_memories[mid] = serializable_memory

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_memories, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"导出记忆快照失败: {e}")
            return False

    def import_memory_snapshot(self, file_path: str) -> Dict[str, Dict]:
        """导入记忆快照

        Args:
            file_path: 导入文件路径

        Returns:
            Dict[str, Dict]: 导入的记忆字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                memories = json.load(f)
            return memories
        except Exception as e:
            print(f"导入记忆快照失败: {e}")
            return {}