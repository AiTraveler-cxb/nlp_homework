# md_knowledge_base.py
import os
import json
from pathlib import Path
from typing import List, Dict, Optional

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from rank_bm25 import BM25Okapi
import jieba
import numpy as np

from src.rag.vector_store import vector_store


class MarkdownStructureParser:
    """Markdown结构化解析器"""

    def __init__(self):
        self.md = MarkdownIt('commonmark')

    def parse(self, md_text: str) -> List[Dict]:
        """解析Markdown为结构化节点"""
        tokens = self.md.parse(md_text)
        tree = SyntaxTreeNode(tokens)
        nodes = []
        self._traverse(tree, nodes)
        return nodes

    def _traverse(self, node, nodes: List, path: List[str] = None):
        if path is None:
            path = []

        # 如果是标题节点
        if node.type == 'heading':
            level = int(node.tag[1])
            # 提取标题文本
            title = ''
            if node.children:
                for child in node.children:
                    if hasattr(child, 'content'):
                        title += child.content

            # 更新路径
            while len(path) >= level:
                path.pop()
            path.append(title)

            # 创建标题节点（带content字段用于存储后续内容）
            nodes.append({
                'type': 'heading',
                'level': level,
                'title': title,
                'path': path.copy(),
                'path_str': ' > '.join(path),
                'content': '',  # 这个字段会累积子内容
                'children': []
            })

        # 如果是段落或文本节点
        elif node.type in ['paragraph', 'blockquote', 'list_item', 'code_block', 'fence', 'bullet_list',
                           'ordered_list']:
            text = self._extract_text(node)
            if text.strip():
                # 查找最近的标题节点（将内容添加到标题节点下）
                if nodes:
                    # 找到最后一个标题节点（从后往前找）
                    last_heading_idx = None
                    for i in range(len(nodes) - 1, -1, -1):
                        if nodes[i]['type'] == 'heading':
                            last_heading_idx = i
                            break

                    if last_heading_idx is not None:
                        # 添加到最近的标题节点的content中
                        nodes[last_heading_idx]['content'] += text + '\n'
                    else:
                        # 没有标题，创建独立内容节点
                        nodes.append({
                            'type': 'content',
                            'level': 0,
                            'title': '',
                            'path': path.copy() if path else [],
                            'path_str': ' > '.join(path) if path else '根目录',
                            'content': text,
                            'children': []
                        })
                else:
                    # 没有节点，创建独立内容节点
                    nodes.append({
                        'type': 'content',
                        'level': 0,
                        'title': '',
                        'path': path.copy() if path else [],
                        'path_str': ' > '.join(path) if path else '根目录',
                        'content': text,
                        'children': []
                    })

        # 递归遍历子节点
        for child in node.children:
            self._traverse(child, nodes, path.copy())

    def _extract_text(self, node) -> str:
        """提取节点中的纯文本"""
        text = ''

        # 处理不同类型的节点
        if hasattr(node, 'content'):
            text = node.content
        elif hasattr(node, 'children'):
            for child in node.children:
                text += self._extract_text(child)

        # 处理代码块特殊格式
        if hasattr(node, 'info'):
            # fence/code_block 的语言标识
            pass

        return text

    def build_chunks(self, nodes: List[Dict], min_size: int = 200, max_size: int = 1500) -> List[Dict]:
        """构建带层级上下文的分块"""
        chunks = []

        for node in nodes:
            if node['type'] == 'heading':
                # 标题节点的内容可能包含了子内容
                content = node.get('content', '').strip()

                # 如果内容为空，跳过这个标题（可能是空章节）
                if not content:
                    continue

                # 构建增强文本：路径前缀 + 内容
                enhanced_text = f"章节：{node['path_str']}\n{content}"

                # 如果内容太长，需要切割
                if len(enhanced_text) > max_size:
                    # 按段落切割
                    paragraphs = content.split('\n')
                    current_text = ''
                    for para in paragraphs:
                        if len(current_text) + len(para) > max_size:
                            if current_text:
                                chunks.append({
                                    'text': f"章节：{node['path_str']}\n{current_text.strip()}",
                                    'metadata': {
                                        'path': node['path'],
                                        'path_str': node['path_str'],
                                        'titles': node['path'],
                                        'content': current_text.strip(),
                                        'char_count': len(current_text),
                                        'chunk_type': 'heading_split'
                                    }
                                })
                            current_text = para + '\n'
                        else:
                            current_text += para + '\n'

                    if current_text.strip():
                        chunks.append({
                            'text': f"章节：{node['path_str']}\n{current_text.strip()}",
                            'metadata': {
                                'path': node['path'],
                                'path_str': node['path_str'],
                                'titles': node['path'],
                                'content': current_text.strip(),
                                'char_count': len(current_text),
                                'chunk_type': 'heading_split'
                            }
                        })
                else:
                    # 正常大小，直接添加
                    chunks.append({
                        'text': enhanced_text,
                        'metadata': {
                            'path': node['path'],
                            'path_str': node['path_str'],
                            'titles': node['path'],
                            'content': content,
                            'char_count': len(content),
                            'chunk_type': 'heading'
                        }
                    })

            # 独立内容节点（没有标题的段落）
            elif node['type'] == 'content':
                content = node.get('content', '').strip()
                if content:
                    chunks.append({
                        'text': content,
                        'metadata': {
                            'path': node['path'],
                            'path_str': node['path_str'],
                            'titles': node['path'],
                            'content': content,
                            'char_count': len(content),
                            'chunk_type': 'content'
                        }
                    })

        return chunks


class ChineseBM25:
    """中文BM25检索器"""

    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.chunks = []

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        self.corpus = [chunk['text'] for chunk in chunks]
        tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self

    def _tokenize(self, text: str) -> List[str]:
        text = text.replace('章节：', '')
        words = jieba.cut(text)
        return [w for w in words if len(w.strip()) > 0]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.bm25:
            raise ValueError("BM25索引未构建")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'index': idx,
                    'score': float(scores[idx]),
                    'chunk': self.chunks[idx],
                    'text': self.corpus[idx]
                })
        return results


class HybridRetriever:
    """混合检索器"""

    def __init__(self, bm25_weight: float = 0.3, vector_weight: float = 0.7):
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25_retriever = ChineseBM25()
        self.chunks = []

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        self.bm25_retriever.build_index(chunks)
        return self

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """混合检索"""
        # 1. 向量检索
        vector_results = vector_store.search(query, k=top_k * 3)

        vector_dict = {}
        for doc in vector_results:
            key = doc.metadata.get('path_str', doc.page_content[:50])
            vector_dict[key] = {
                'score': doc.metadata.get('retriever_score', 0),
                'chunk': {
                    'text': doc.page_content,
                    'metadata': doc.metadata
                }
            }

        # 2. BM25检索
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 3)
        bm25_dict = {}
        for r in bm25_results:
            key = r['chunk']['metadata'].get('path_str', f"bm25_{r['index']}")
            bm25_dict[key] = {
                'score': r['score'],
                'chunk': r['chunk']
            }

        # 3. 归一化
        def normalize(scores_dict):
            if not scores_dict:
                return {}
            scores = [item['score'] for item in scores_dict.values()]
            max_score = max(scores) if scores else 1
            if max_score == 0:
                return {k: 0 for k in scores_dict}
            return {k: v['score'] / max_score for k, v in scores_dict.items()}

        vector_norm = normalize(vector_dict)
        bm25_norm = normalize(bm25_dict)

        # 4. 加权融合
        all_keys = set(vector_norm.keys()) | set(bm25_norm.keys())
        final_scores = {}

        for key in all_keys:
            vector_score = vector_norm.get(key, 0) * self.vector_weight
            bm25_score = bm25_norm.get(key, 0) * self.bm25_weight
            final_scores[key] = vector_score + bm25_score

        # 5. 排序
        sorted_keys = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:top_k]

        results = []
        for key in sorted_keys:
            if key in vector_dict:
                chunk_data = vector_dict[key]
            else:
                chunk_data = bm25_dict[key]

            results.append({
                'score': final_scores[key],
                'chunk': chunk_data['chunk'],
                'text': chunk_data['chunk']['text']
            })

        return results


class KnowledgeBaseInitializer:
    """知识库初始化器"""

    def __init__(self,
                 docs_root: Path,
                 collection_name: str = "md_knowledge",
                 bm25_weight: float = 0.3,
                 vector_weight: float = 0.7,
                 ignore_vec_init = False):
        self.docs_root = Path(docs_root)
        self.collection_name = collection_name
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.parser = MarkdownStructureParser()
        self.retriever = None
        self.all_chunks = []
        self.ignore_vec_init = ignore_vec_init

    def load_and_chunk(self) -> List[Dict]:
        """加载所有MD文档并分块"""
        all_chunks = []
        md_files = list(self.docs_root.glob('*.md'))

        if not md_files:
            raise ValueError(f"在 {self.docs_root} 中未找到任何 .md 文件")

        print(f"找到 {len(md_files)} 个MD文件")

        for file_path in md_files:
            print(f"  处理: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_text = f.read()

                # 解析
                nodes = self.parser.parse(md_text)

                # 打印调试信息
                print(f"    解析出 {len(nodes)} 个节点")
                for i, node in enumerate(nodes[:3]):  # 只打印前3个
                    print(
                        f"      [{i}] type={node['type']}, title={node.get('title', '')[:30]}, content_len={len(node.get('content', ''))}")

                # 分块
                chunks = self.parser.build_chunks(nodes)
                print(f"    生成 {len(chunks)} 个分块")

                # 添加文档来源
                for chunk in chunks:
                    chunk['metadata']['doc_source'] = file_path.name
                    chunk['metadata']['doc_path'] = str(file_path)
                    chunk['metadata']['chunk_id'] = f"{file_path.stem}_{hash(chunk['text'])}"

                all_chunks.extend(chunks)

            except Exception as e:
                print(f"  处理 {file_path.name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.all_chunks = all_chunks
        print(f"共生成 {len(all_chunks)} 个分块")
        return all_chunks

    def build_index(self, clear_existing: bool = True)-> HybridRetriever:
        """构建索引"""
        if not self.all_chunks:
            self.load_and_chunk()

        # 存入向量库
        if not self.ignore_vec_init:
            print(f"存入向量库: {self.collection_name}")
            for i, chunk in enumerate(self.all_chunks):
                try:
                    vector_store.add_text(
                        text=chunk['text'],
                        metadata={
                            **chunk['metadata'],
                            'collection': self.collection_name
                        }
                    )
                except Exception as e:
                    print(f"  存入第 {i + 1} 个分块时出错: {e}")
                    continue

                if (i + 1) % 10 == 0:
                    print(f"  已存入 {i + 1}/{len(self.all_chunks)} 个分块")
            print(f"向量库存入完成！共 {len(self.all_chunks)} 个分块")

        # 构建BM25
        print("构建BM25索引...")
        self.retriever = HybridRetriever(
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight
        )
        self.retriever.build_index(self.all_chunks)
        print("索引构建完成！")

        return self.retriever

    def save_metadata(self, save_dir: Path):
        """保存元数据"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'chunks_metadata.json', 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': self.all_chunks,
                'config': {
                    'collection_name': self.collection_name,
                    'bm25_weight': self.bm25_weight,
                    'vector_weight': self.vector_weight,
                    'total_chunks': len(self.all_chunks)
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"元数据已保存到: {save_dir}")

    def load_metadata(self, load_dir: Path):
        """加载元数据"""
        load_dir = Path(load_dir)

        with open(load_dir / 'chunks_metadata.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.all_chunks = data['chunks']
        config = data.get('config', {})
        self.collection_name = config.get('collection_name', self.collection_name)
        self.bm25_weight = config.get('bm25_weight', self.bm25_weight)
        self.vector_weight = config.get('vector_weight', self.vector_weight)

        print(f"元数据已从 {load_dir} 加载，共 {len(self.all_chunks)} 个分块")
        return self


if __name__ == '__main__':
    ROOT = Path(__file__).parent.parent
    DOC_ROOT = ROOT / 'docs'
    META_ROOT = ROOT / 'kb_metadata'

    # 初始化
    kb = KnowledgeBaseInitializer(
        docs_root=DOC_ROOT,
        collection_name="md_knowledge",
        bm25_weight=0.3,
        vector_weight=0.7
    )

    # 加载并分块
    kb.load_and_chunk()

    # 构建索引
    retriever = kb.build_index()

    # 保存元数据
    kb.save_metadata(META_ROOT)

    # 测试
    print("\n" + "=" * 60)
    print("测试检索")
    print("=" * 60)

    test_queries = [
        "套包如何创建",
        "套包出库流程"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)

        results = retriever.search(query, top_k=3)

        if not results:
            print("  无结果")
        else:
            for i, result in enumerate(results, 1):
                print(f"  [{i}] 分数: {result['score']:.4f}")
                print(f"      路径: {result['chunk']['metadata'].get('path_str', 'N/A')}")
                print(f"      预览: {result['text'][:150]}...")
                print()