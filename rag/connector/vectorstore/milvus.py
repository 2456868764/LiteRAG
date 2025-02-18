from __future__ import annotations
from typing import List
import uuid
import operator
from pymilvus import MilvusClient
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.milvus import Milvus

from rag.common.utils import md5_encryption
from rag.module.knowledge_file import KnowledgeFile
from rag.connector.vectorstore.base import VectorStore
from rag.common.configuration import config
from comps import CustomLogger

logger = CustomLogger("milvus_vector_store")


class MilvusVectorStore(VectorStore):

    def __init__(self,
                 embedding_model: Embeddings,
                 collection_name: str):
        self.embeddings = embedding_model
        self.knowledge_name = collection_name
        self.collection_name = collection_name
        self.config = config.vector_store.milvus
        self.milvus = None
        self._load_milvus()

    def _load_milvus(self):
        """
        初始化并加载Milvus客户端。

        该方法从self.config中获取连接配置参数，包括主机、端口、用户名、密码和数据库名称，
        并使用这些参数建立与Milvus数据库的连接。此外，它还配置了索引和搜索参数，以便对数据库
        进行高效的查询。通过这种方式，确保了与数据库的稳定连接和高效交互。
        """
        # 构建连接参数字典，用于建立与Milvus数据库的连接
        connection_args = {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "password": self.config.password,
            "secure": False,
            "db_name": self.config.db_name,
        }
        # 获取索引参数，用于数据库的索引构建
        index_params = self.config.kwargs.get("index_params", None)
        # 获取搜索参数，用于数据库的查询操作
        search_params = self.config.kwargs.get("search_params", None)
        # 初始化langchain client，用于与Milvus数据库进行交互
        self.milvus = Milvus(self.embeddings,
                             collection_name=self.collection_name,
                             connection_args=connection_args,
                             index_params=index_params,
                             search_params=search_params,
                             metadata_field="metadata",
                             auto_id=False)
        # 初始化pyclient，提供另一种方式与Milvus数据库进行交互
        self.pyclient = MilvusClient(
            uri="http://"+self.config.host+":"+ str(self.config.port),
            db_name=self.config.db_name,
        )

    def create_vectorstore(self):
        init_kwargs = {"embeddings": self.embeddings.embed_documents(["初始化"]),
                       "metadatas": [{}],
                       "partition_names": None,
                       "replica_number": 1,
                       "timeout": None}
        self.milvus._init(**init_kwargs)

    def drop_vectorstore(self):
        if self.pyclient.has_collection(self.collection_name):
            self.pyclient.release_collection(self.collection_name)
            self.pyclient.drop_collection(self.collection_name)

    def clear_vectorstore(self):
        if self.pyclient.has_collection(self.collection_name):
            self.pyclient.release_collection(self.collection_name)
            self.pyclient.drop_collection(self.collection_name)
            self._load_milvus()

    def delete_doc(self, filename):
        if self.pyclient.has_collection(self.collection_name):
            delete_list = [item.get("pk") for item in
                           self.pyclient.query(collection_name=self.collection_name,
                                               filter=f'metadata["source"] == "{md5_encryption(filename)}"',
                                               output_fields=["pk"])]

            if len(delete_list) > 0:
                self.pyclient.delete(collection_name=self.collection_name,
                                     filter=f'pk in {delete_list}')
                logger.warning(f"成功删除文件 {filename} {str(len(delete_list))} 条记录")
            else:
                logger.warning(f"vs中不存在文件 {filename} 相关的记录，不需要删除")
        else:
            logger.warning(f"vs为空，没有可删除的记录")

    def update_doc(self, file: KnowledgeFile, docs: List[Document]):
        self.delete_doc(file.filename)
        return self.add_doc(file, docs=docs)

    def add_doc(self, file: KnowledgeFile, docs, **kwargs):
        """
        向系统中添加文档。

        将所有信息存储到metadata中，包括源文件的MD5值、文件名等，并确保所有字段都已设置，
        然后移除文本字段和向量字段，最后生成或保留文档的唯一标识符。

        参数:
        - file: KnowledgeFile类型，表示要添加的文件。
        - docs: 待添加的文档集合。
        - **kwargs: 其他可变关键字参数。

        返回:
        - doc_infos: 包含每个文档的id和metadata信息的列表。
        """
        # 初始化文档ID列表
        doc_ids = []
        # 遍历每个文档，为它们设置元数据
        for doc in docs:
            # 设置文档来源为文件名的MD5值
            doc.metadata["source"] = md5_encryption(file.filename)
            # 设置文档的文件名
            doc.metadata["filename"] = file.filename
            # 将所有元数据值转换为字符串类型
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            # 为元数据中不存在的字段设置默认值
            for field in self.milvus.fields:
                doc.metadata.setdefault(field, "")
            # 移除文本字段和向量字段，因为它们不在元数据中
            doc.metadata.pop(self.milvus._text_field, None)
            doc.metadata.pop(self.milvus._vector_field, None)
            # 使用UUID生成文档ID，如果元数据中已提供，则保留
            doc_id = doc.metadata.get("id", str(uuid.uuid4()))
            doc_ids.append(doc_id)

        # Function to yield batches of documents and their corresponding IDs
        def batch_documents(docs, doc_ids, batch_size=16):
            for i in range(0, len(docs), batch_size):
                yield docs[i:i + batch_size], doc_ids[i:i + batch_size]

        # 批量添加文档
        all_doc_infos = []
        for batch_docs, batch_doc_ids in batch_documents(docs, doc_ids):
            # 根据文档ID列表是否为空，调用不同的方法添加文档到Milvus
            ids = self.milvus.add_documents(batch_docs) if len(batch_doc_ids) == 0 else self.milvus.add_documents(
                batch_docs, **{"ids": batch_doc_ids})
            # 构建返回的文档信息列表
            batch_doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, batch_docs)]
            all_doc_infos.extend(batch_doc_infos)

        return all_doc_infos
        # # 根据文档ID列表是否为空，调用不同的方法添加文档到Milvus
        # ids = self.milvus.add_documents(docs) if len(doc_ids) == 0 else self.milvus.add_documents(docs, **{"ids": doc_ids})
        # # 构建返回的文档信息列表
        # doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        #
        # return doc_infos

    def search_docs(self, text, top_k, threshold, **kwargs):
        """
        搜索与给定文本最相似的文档。

        参数:
        - text: 查询文本。
        - top_k: 返回的最相似文档的数量。
        - threshold: 相似度分数阈值，只有分数高于此阈值的文档才会被返回。
        - **kwargs: 其他传递给相似性搜索函数的参数。

        返回:
        - docs: 最相似的文档列表，可能包括原始文档和它们的父文档（如果存在）。
        """
        # 执行相似性搜索，获取最相似的文档和它们的分数
        docs = self.milvus.similarity_search_with_score(query=text,
                                                        k=top_k,
                                                        **kwargs)
        # 如果设置了阈值，则过滤掉分数低于阈值的文档
        if threshold is not None:
            docs = self._score_threshold_process(docs, threshold, top_k)

        docs = self.get_parents(docs)
        return docs

    def search_docs_by_vector(self, embedding, top_k, threshold, **kwargs):
        """
        使用向量相似性搜索文档。

        该方法通过向量相似性搜索找出与给定向量最接近的文档。可接受额外的关键字参数以适应不同的搜索需求。

        参数:
        - embedding (List[float]): 用于搜索的向量。
        - top_k (int): 返回的最相似文档的数量。
        - threshold (float): 可选。相似度分数的阈值，只有分数高于此值的文档才会被返回。
        - **kwargs: 其他传递给相似性搜索方法的参数。

        返回:
        - List: 包含最相似文档的列表。
        """
        # 执行相似性搜索，获取最相似的文档和它们的分数
        docs = self.milvus.similarity_search_by_vector(embedding=embedding,
                                                        k=top_k,
                                                        **kwargs)
        # 如果设置了阈值，则过滤掉分数低于阈值的文档
        if threshold is not None:
            docs = self._score_threshold_process(docs, threshold, top_k)

        # 获取文档的父文档，这可能是为了上下文的完整性或是进一步的文档处理
        docs = self.get_parents(docs)
        return docs

    def search_docs_by_mmr(self, text, top_k, fetch_k, lambda_mult, **kwargs):
        """
        使用最大边际相关性搜索（Maximal Marginal Relevance, MMR）来检索文档。

        该方法旨在处理文档搜索中的多样性问题，通过平衡文档与查询的相关性和文档间的相似性来选择最具代表性的文档。

        参数:
        - text (str): 查询文本，用于搜索相关文档。
        - top_k (int): 最终返回的文档数量。
        - fetch_k (int): 从数据库中获取的候选文档数量，大于等于top_k。
        - lambda_mult (float): MMR公式中的lambda参数，用于调整相关性和多样性的权重。
        - **kwargs: 其他额外参数，传递给搜索算法。

        返回:
        - List: 经过MMR算法筛选后的文档列表。
        """
        # 执行相似性搜索，获取最相似的文档和它们的分数
        docs = self.milvus.max_marginal_relevance_search(query=text,
                                                        k=top_k,
                                                        fetch_k=fetch_k,
                                                        lambda_mult=lambda_mult,
                                                        **kwargs)
        # 获取文档的父文档，具体逻辑未展示，假设在get_parents函数中实现
        docs = self.get_parents(docs)
        return docs

    def get_parents(self, docs):
        """
       召回父文档并替换当前文档列表中的子文档。

        该函数遍历给定的文档列表，查找每个文档的父文档ID，并构建一个映射。
        如果找到了父文档ID，则通过调用外部服务获取父文档，并用父文档替换原始文档列表中的子文档。

        参数:
        - docs: 文档列表，其中每个文档是一个包含文档信息的元组。

        返回:
        - 替换子文档为父文档后的文档列表。
        """
        # 兼容multi_vector，召回父文档
        parent_doc_map = {}
        for i, tp in enumerate(docs):
            parent_id = tp.metadata.get("parent_id")
            if parent_id is not None: parent_doc_map[i] = parent_id

        if len(parent_doc_map) > 0:
            try:
                # 去重后得到所有父文档的ID
                ids = list(set(parent_doc_map.values()))
                # 初始化父文档字典，用于存储父文档ID与文档对象的映射
                parent_docs = {}  # parent_id: parent_doc
                # 通过外部服务获取父文档信息，并构建父文档对象
                for p_doc in self.pyclient.get(collection_name=self.collection_name,
                                               ids=ids,
                                               output_fields=["pk", "text", "metadata"]):
                    parent_docs[p_doc["pk"]] = Document(page_content=p_doc["text"],
                                                        metadata=p_doc["metadata"])
                # 用父文档替换原始文档列表中的子文档
                for doc_index in parent_doc_map:
                    docs[doc_index] = tuple([parent_docs[parent_doc_map[doc_index]], docs[doc_index][1]])

            except Exception as e:
                # 处理查找父文档时可能发生的异常
                msg = f"find parent chunk fail：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e)

        return docs

    def _score_threshold_process(self, docs, score_threshold, k):
        if score_threshold is not None:
            cmp = (
                operator.ge
            )
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]
        return docs[:k]

