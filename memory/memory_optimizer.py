import time
import heapq
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - 依赖缺失时降级到 TF-IDF
    torch = None
    SentenceTransformer = None

class MemoryOptimizer:
    """记忆优化器，负责记忆的压缩、重要性排序和高效检索"""
    def __init__(self):
        # 中文文本没有空格分词，默认的词级 TF-IDF 会把整段当作单个 token，
        # 导致相似度几乎为 0。使用字符级 3-gram 既能捕捉中文语义重叠，又能让
        # 完全不相关的查询（不共享任何 3-gram）得到 0 相似度。
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
        self.memory_vectors = {}
        self.semantic_vectors = {}
        self.load_config()
        # 初始化语义模型（离线/受限环境下会失败，自动降级到 TF-IDF）
        if SentenceTransformer is None:
            self.use_semantic = False
        else:
            try:
                self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                self.use_semantic = True
            except Exception as e:
                print(f"警告: 无法加载语义模型: {e}")
                self.use_semantic = False

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
                "semantic_weight": 0.6,  # 语义相似度权重
                "tfidf_weight": 0.4,     # TF-IDF相似度权重
                "semantic_threshold": 0.75  # 语义相似度阈值
            }

    def calculate_importance(self, memory: Dict) -> float:
        """计算记忆的重要性分数

        Args:
            memory: 记忆对象

        Returns:
            float: 重要性分数
        """
        # 基于时间衰减的重要性计算
        current_time = time.time()
        creation_time = self._to_timestamp(memory.get('timestamp', current_time), current_time)
        # 时间差以小时为单位（衰减率针对小时尺度标定），避免以秒为单位时
        # 衰减过快、使所有记忆重要性都趋近于 0 而被时间戳完全主导。
        time_diff_hours = max(0.0, (current_time - creation_time) / 3600.0)

        # 基础重要性分数
        base_importance = memory.get('importance', 0.5)

        # 计算时间衰减
        decay_factor = np.exp(-self.config['importance_decay_rate'] * time_diff_hours)

        # 考虑访问频率（access_count 可能为 0，需做下界保护以避免 log10(0)=-inf）
        access_count = max(1, memory.get('access_count', 1))
        frequency_factor = min(1.5, 1 + np.log10(access_count))

        # 综合计算最终重要性
        importance = float(base_importance * decay_factor * frequency_factor)
        return importance

    @staticmethod
    def _to_timestamp(value, default: float) -> float:
        """将时间戳归一化为 float 秒。

        支持数值时间戳（秒）以及 ISO 格式的时间字符串（如 MemoryManager 写入的）。
        无法解析时返回 default。
        """
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(value).timestamp()
            except Exception:
                return default
        return default

    def _compression_similarity_threshold(self) -> float:
        """返回与当前向量空间相适配的压缩相似度阈值。

        使用语义模型时沿用配置中的阈值；降级到字符级 TF-IDF 时相似度整体偏小，
        采用更低的阈值以便正确合并相似记忆。
        """
        if self.use_semantic:
            return self.config.get('compression_threshold', 0.7)
        return self.config.get('tfidf_compression_threshold', 0.2)

    def rank_retrieved_memories(self, query: str, memories: Dict[str, Dict],
                                memory_ids: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """对已检索到的候选记忆按与查询的相关度结合重要性重新排序。

        Args:
            query: 查询文本
            memories: 候选记忆字典（memory_id -> 记忆对象）
            memory_ids: 候选记忆ID列表（保持原始顺序，作为打分相等时的稳定顺序）
            top_k: 返回的最大结果数

        Returns:
            List[Tuple[str, float]]: (记忆ID, 排序分数) 列表，按分数降序
        """
        if not memory_ids:
            return []

        # 计算每个候选记忆与查询的相关度分数
        scored = []
        for mid in memory_ids:
            memory = memories.get(mid)
            if memory is None:
                continue

            relevance = 0.0
            content = memory.get('content', '')
            if content and query:
                try:
                    # 使用临时向量器，避免污染 self.vectorizer 的已拟合状态
                    local_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
                    vecs = local_vec.fit_transform([content, query])
                    relevance = float(cosine_similarity(vecs[0], vecs[1])[0][0])
                except ValueError:
                    relevance = 0.0

            importance = self.calculate_importance(memory)
            # 结合已有的相似度分数（若存在）、相关度与重要性。
            # 四舍五入以消除浮点噪声（如 FAISS 在距离相同的情况下引入的 ~1e-7
            # 级差异），使分数相同的记忆按稳定的传入顺序排列。
            base_similarity = float(memory.get('similarity_score', 0.0))
            score = round(base_similarity + relevance + importance, 4)
            scored.append((mid, score))

        # 稳定排序：分数相同时保持传入顺序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def compute_semantic_vectors(self, texts: List[str]) -> Optional[np.ndarray]:
        """计算文本的语义向量

        Args:
            texts: 文本列表

        Returns:
            Optional[np.ndarray]: 语义向量数组，如果无法计算则返回None
        """
        if not self.use_semantic or not texts:
            return None

        try:
            with torch.no_grad():
                embeddings = self.semantic_model.encode(texts)
            return embeddings
        except Exception as e:
            print(f"警告: 计算语义向量时出错: {e}")
            return None

    def compute_similarity_matrix(self, memory_ids: List[str], memory_texts: List[str]) -> np.ndarray:
        """计算记忆之间的相似度矩阵，结合TF-IDF和语义相似度

        Args:
            memory_ids: 记忆ID列表
            memory_texts: 记忆文本列表

        Returns:
            np.ndarray: 相似度矩阵
        """
        # 计算TF-IDF相似度
        tfidf_vectors = None
        try:
            tfidf_vectors = self.vectorizer.fit_transform(memory_texts)
            tfidf_similarity = cosine_similarity(tfidf_vectors)
        except ValueError:
            # 处理空文本情况
            tfidf_similarity = np.zeros((len(memory_ids), len(memory_ids)))

        # 存储TF-IDF向量以便后续检索
        if tfidf_vectors is not None:
            for i, mid in enumerate(memory_ids):
                self.memory_vectors[mid] = tfidf_vectors[i]

        # 如果启用了语义模型，计算语义相似度并结合
        if self.use_semantic:
            semantic_vectors = self.compute_semantic_vectors(memory_texts)
            if semantic_vectors is not None:
                # 存储语义向量
                for i, mid in enumerate(memory_ids):
                    self.semantic_vectors[mid] = semantic_vectors[i]
                
                # 计算语义相似度
                semantic_similarity = cosine_similarity(semantic_vectors)
                
                # 结合TF-IDF和语义相似度
                tfidf_weight = self.config['tfidf_weight']
                semantic_weight = self.config['semantic_weight']
                combined_similarity = (tfidf_weight * tfidf_similarity + 
                                      semantic_weight * semantic_similarity)
                return combined_similarity

        return tfidf_similarity

    def compress_memories(self, memories: Dict[str, Dict]) -> Tuple[Dict[str, Dict], List[str]]:
        """压缩相似记忆

        Args:
            memories: 记忆字典，key为记忆ID，value为记忆对象

        Returns:
            Tuple[Dict[str, Dict], List[str]]: 压缩后的记忆字典和被删除的记忆ID列表
        """
        if not memories:
            return memories, []

        # 提取记忆文本并向量化
        memory_ids = list(memories.keys())
        memory_texts = [memories[mid]['content'] for mid in memory_ids]

        # 计算相似度矩阵（结合TF-IDF和语义相似度）
        similarity_matrix = self.compute_similarity_matrix(memory_ids, memory_texts)

        # 记录要删除的记忆ID
        to_delete = set()
        compressed_memories = memories.copy()

        # 查找并合并相似记忆。
        # 注意：字符级 3-gram 的 TF-IDF 余弦相似度整体偏小（中文相似文本通常在
        # 0.2~0.5 之间），配置中的 compression_threshold 是针对语义模型空间标定的，
        # 直接套用会导致无法合并任何记忆。这里使用与 TF-IDF 空间相适配的阈值。
        threshold = self._compression_similarity_threshold()
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
        """检索与查询相关的记忆，使用TF-IDF和语义相似度的组合

        Args:
            query: 查询文本
            memories: 记忆字典
            top_k: 返回的最大结果数

        Returns:
            List[Tuple[str, float]]: 记忆ID和相关度分数的列表
        """
        if not memories or not query:
            return []

        memory_ids = [mid for mid in memories.keys() if memories[mid].get('content')]
        if not memory_ids:
            return []

        # 在记忆语料 + 查询上拟合向量器，保证查询与记忆处于同一向量空间
        corpus = [memories[mid]['content'] for mid in memory_ids]
        try:
            self.vectorizer.fit(corpus + [query])
        except ValueError:
            return []

        # 重新计算并缓存所有记忆的TF-IDF向量
        for mid in memory_ids:
            try:
                self.memory_vectors[mid] = self.vectorizer.transform([memories[mid]['content']])
            except ValueError:
                continue

        try:
            query_vector = self.vectorizer.transform([query])
        except ValueError:
            return []

        # 如果启用语义模型，计算查询的语义向量
        semantic_query_vector = None
        if self.use_semantic:
            semantic_query_vector = self.compute_semantic_vectors([query])

        # 相关性下界：字符级 TF-IDF 中，完全不相关的查询（不共享任何 3-gram）
        # 相似度恰好为 0，因此用一个极小正阈值即可可靠地过滤掉无关记忆，
        # 而不会受配置中针对语义空间标定的较高阈值影响。
        epsilon = 1e-9

        similarities = []
        for mid in memory_ids:
            tfidf_similarity = 0.0
            if mid in self.memory_vectors:
                tfidf_similarity = float(cosine_similarity(query_vector, self.memory_vectors[mid])[0][0])

            semantic_similarity = 0.0
            combined_similarity = tfidf_similarity
            if (self.use_semantic and semantic_query_vector is not None
                    and mid in self.semantic_vectors):
                memory_semantic_vector = self.semantic_vectors[mid].reshape(1, -1)
                semantic_similarity = float(
                    cosine_similarity(semantic_query_vector, memory_semantic_vector)[0][0])
                tfidf_weight = self.config['tfidf_weight']
                semantic_weight = self.config['semantic_weight']
                combined_similarity = (tfidf_weight * tfidf_similarity +
                                       semantic_weight * semantic_similarity)

            if combined_similarity > epsilon:
                similarities.append((mid, combined_similarity))

        # 按相似度降序排序并返回前top_k个结果（相似度相同时保持稳定顺序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def prioritize_memories(self, memories: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """对记忆进行优先级排序

        Args:
            memories: 记忆字典

        Returns:
            List[Tuple[str, float]]: 记忆ID和优先级分数的列表
        """
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

        # 确定要保留的记忆数量（向上取整：保留比例应至少覆盖该比例的记忆，
        # 例如 3 条记忆 keep_ratio=0.8 应保留 3 条而非 2 条）
        import math
        keep_count = max(1, math.ceil(len(memories) * keep_ratio))

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
        
    def advanced_memory_compression(self, memories: Dict[str, Dict]) -> Dict[str, Dict]:
        """高级记忆压缩算法，结合语义聚类和信息提取

        Args:
            memories: 记忆字典

        Returns:
            Dict[str, Dict]: 压缩后的记忆字典
        """
        if not memories or len(memories) < 3:
            return memories
            
        # 1. 提取记忆文本和ID
        memory_ids = list(memories.keys())
        memory_texts = [memories[mid]['content'] for mid in memory_ids]
        
        # 2. 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(memory_ids, memory_texts)
        
        # 3. 基于相似度进行层次聚类
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # 将相似度矩阵转换为距离矩阵
        distance_matrix = 1 - similarity_matrix
        # 确保对角线为0
        np.fill_diagonal(distance_matrix, 0)
        
        try:
            # 将距离矩阵转换为压缩形式
            condensed_distance = squareform(distance_matrix)
            # 执行层次聚类
            Z = linkage(condensed_distance, method='ward')
            # 根据距离阈值确定聚类数
            max_dist = 1 - self.config['compression_threshold']
            clusters = fcluster(Z, max_dist, criterion='distance')
        except Exception as e:
            print(f"聚类过程中出错: {e}")
            return memories
        
        # 4. 为每个聚类创建一个压缩记忆
        compressed_memories = {}
        cluster_to_mids = {}
        
        # 将记忆ID按聚类分组
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_to_mids:
                cluster_to_mids[cluster_id] = []
            cluster_to_mids[cluster_id].append(memory_ids[i])
        
        # 处理每个聚类
        for cluster_id, cluster_mids in cluster_to_mids.items():
            if len(cluster_mids) == 1:
                # 单个记忆的聚类，直接保留
                mid = cluster_mids[0]
                compressed_memories[mid] = memories[mid]
            else:
                # 多个记忆的聚类，创建压缩记忆
                cluster_memories = [memories[mid] for mid in cluster_mids]
                
                # 找出最重要的记忆作为基础
                importance_scores = [(mid, self.calculate_importance(memories[mid])) for mid in cluster_mids]
                importance_scores.sort(key=lambda x: x[1], reverse=True)
                base_mid = importance_scores[0][0]
                base_memory = memories[base_mid]
                
                # 合并记忆内容
                merged_content = self._merge_memory_contents(cluster_memories)
                
                # 合并标签
                all_tags = set()
                for memory in cluster_memories:
                    if 'tags' in memory and memory['tags']:
                        all_tags.update(memory['tags'])
                
                # 使用最新的时间戳
                latest_timestamp = max(memory.get('timestamp', 0) for memory in cluster_memories)
                
                # 累加访问计数
                total_access_count = sum(memory.get('access_count', 0) for memory in cluster_memories)
                
                # 创建压缩记忆
                compressed_memory = {
                    'content': merged_content,
                    'timestamp': latest_timestamp,
                    'tags': list(all_tags),
                    'access_count': total_access_count,
                    'source_memories': cluster_mids  # 记录源记忆ID
                }
                
                # 使用最重要记忆的ID作为压缩记忆的ID
                compressed_memories[base_mid] = compressed_memory
        
        return compressed_memories
    
    def _merge_memory_contents(self, memories: List[Dict]) -> str:
        """智能合并多个记忆的内容

        Args:
            memories: 记忆列表

        Returns:
            str: 合并后的内容
        """
        if not memories:
            return ""
        
        if len(memories) == 1:
            return memories[0]['content']
            
        # 简单方法：连接所有内容并去重
        contents = [memory['content'] for memory in memories]
        
        # 如果有语义模型，尝试更智能的合并
        if self.use_semantic:
            try:
                # 提取每个内容的关键句子
                from sklearn.feature_extraction.text import CountVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import re
                
                # 分割成句子
                all_sentences = []
                sentence_to_content_idx = []
                
                for i, content in enumerate(contents):
                    # 简单的句子分割
                    sentences = re.split(r'(?<=[.!?。！？]\s)', content)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    all_sentences.extend(sentences)
                    sentence_to_content_idx.extend([i] * len(sentences))
                
                if not all_sentences:
                    return "\n\n".join(contents)
                    
                # 向量化句子
                sentence_vectors = self.compute_semantic_vectors(all_sentences)
                if sentence_vectors is None:
                    return "\n\n".join(contents)
                    
                # 计算句子间的相似度
                sentence_similarities = cosine_similarity(sentence_vectors)
                
                # 选择代表性句子
                selected_indices = []
                remaining_indices = list(range(len(all_sentences)))
                
                # 贪婪选择算法
                while remaining_indices and len(selected_indices) < min(10, len(all_sentences)):
                    if not selected_indices:
                        # 选择第一个句子（最长的句子）
                        longest_idx = max(remaining_indices, key=lambda i: len(all_sentences[i]))
                        selected_indices.append(longest_idx)
                        remaining_indices.remove(longest_idx)
                    else:
                        # 选择与已选句子最不相似的句子
                        max_min_similarity = -1
                        next_idx = -1
                        
                        for idx in remaining_indices:
                            min_similarity = min(sentence_similarities[idx][sel_idx] for sel_idx in selected_indices)
                            if min_similarity > max_min_similarity:
                                max_min_similarity = min_similarity
                                next_idx = idx
                        
                        if next_idx != -1:
                            selected_indices.append(next_idx)
                            remaining_indices.remove(next_idx)
                        else:
                            break
                
                # 按原始顺序排序选定的句子
                selected_indices.sort()
                
                # 构建合并内容
                merged_content = "\n".join(all_sentences[i] for i in selected_indices)
                return merged_content
                
            except Exception as e:
                print(f"智能合并记忆内容时出错: {e}")
                pass
        
        # 回退到简单合并
        return "\n\n".join(contents)

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

            return memories
        except Exception as e:
            print(f"加载记忆失败: {str(e)}")
            return {}

    def optimize_memory_storage(self, memories: Dict[str, Dict]) -> Tuple[Dict[str, Dict], List[str]]:
        """全面优化记忆存储

        Args:
            memories: 记忆字典

        Returns:
            Tuple[Dict[str, Dict], List[str]]: 优化后的记忆字典和被删除的记忆ID列表
        """
        # 第一步：压缩相似记忆
        compressed_memories, deleted_due_to_compression = self.compress_memories(memories)

        # 第二步：遗忘低优先级记忆
        deleted_due_to_priority = self.forget_low_priority_memories(compressed_memories)

        # 合并删除列表
        all_deleted = list(set(deleted_due_to_compression + deleted_due_to_priority))

        # 创建最终的记忆字典
        final_memories = {mid: memory for mid, memory in compressed_memories.items() if mid not in all_deleted}

        return final_memories, all_deleted