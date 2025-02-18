from abc import ABC
from dataclasses import dataclass
import os
import uuid
from typing import List, Union, Tuple, Dict

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from comps import CustomLogger
from rag.common.utils import run_in_thread_pool
from rag.connector.database.service.knowledge_file_service import KnowledgeFileService
from rag.connector.vectorstore.base import VectorStore
from rag.module.knowledge_file import KnowledgeFile
from rag.module.indexing.multi_vector import split_smaller_chunks
from rag.module.indexing.splitter import DOCUMENTS_SPLITER_MAPPING

logger = CustomLogger("Indexing")


@dataclass
class Indexing(ABC):
    vectorstore: VectorStore
    knowledge_file_service: KnowledgeFileService
    chunk_size: int
    chunk_overlap: int
    smaller_chunk_size: int

    def load(self,
             file: KnowledgeFile,
             loader: None):
        if loader is None:
            loader_class = file.document_loader
        else:
            loader_class = loader   
        if file.type == 'file':
            file_path = file.filename if os.path.exists(file.filename) else file.filepath
        else:
            file_path = file.filename
        docs = loader_class(file_path).load()
        return docs

    def split(self,
              docs: List[Document],
              splitter: Union[str, TextSplitter]):
        if isinstance(splitter, str):
            splitter = DOCUMENTS_SPLITER_MAPPING[splitter]
        chunks = splitter(chunk_size=self.chunk_size,
                          chunk_overlap=self.chunk_overlap).split_documents(documents=docs)
        if not chunks: return []
        index = 0
        for chunk in chunks:
            chunk.metadata["id"] = str(uuid.uuid4())
            chunk.metadata["index"] = index
            index += 1

        multi_vector_chunks = []
        if self.smaller_chunk_size is not None and int(self.smaller_chunk_size) > 0:
            multi_vector_chunks.extend(split_smaller_chunks(chunks, self.smaller_chunk_size))

        return chunks + multi_vector_chunks

    def file2chunks(self, file, **kwargs) -> Tuple[bool, Tuple[KnowledgeFile, List[Document]]]:
        try:
            docs = self.load(file=file, loader=None)
            chunks = self.split(docs=docs, splitter=file.text_splitter)
            return True, (file, chunks, docs)
        except Exception as e:
            msg = f"load {file.filename} file error：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}')
            return False, (file, msg, [])

    def store(self,
              file: KnowledgeFile,
              chunks: List[Document]):

        del_status = self.knowledge_file_service.delete_file_from_db(file)
        doc_infos = self.vectorstore.update_doc(file=file, docs=chunks)
        add_db_status = self.knowledge_file_service.add_file_to_db(file, docs_count=len(chunks)) and \
                        self.knowledge_file_service.add_docs_to_db(file.knowledge_name,
                                       file.filename,
                                       doc_infos=doc_infos)
        return del_status and add_db_status

    def index(self,
              files: List[Union[KnowledgeFile, Tuple[str, str], Dict]], ):
        failed_files = {}
        kwargs_list = []
        for i, file in enumerate(files):
            kwargs = {"file": file}
            kwargs_list.append(kwargs)
        for status, result in run_in_thread_pool(func=self.file2chunks, params=kwargs_list):
            if status:
                file, chunks, docs = result
                self.store(file, chunks)
            else:
                file, error = result
                failed_files[file.filename] = error
        return failed_files
